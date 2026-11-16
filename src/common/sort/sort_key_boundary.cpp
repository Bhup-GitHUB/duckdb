#include "duckdb/common/sorting/sort_key_boundary.hpp"

#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/dictionary_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

SortKeyBoundary::SortKeyBoundary(const LogicalType &first_type, OrderModifiers first_modifiers_p,
                                 shared_ptr<DynamicFilterData> dynamic_filter_p)
    : dynamic_filter(std::move(dynamic_filter_p)), first_modifiers(first_modifiers_p), first_value(first_type, 1) {
}

bool SortKeyBoundary::Get(string &result) const {
	const annotated_lock_guard<annotated_mutex> guard {lock};
	if (is_set) {
		result = key;
	}
	return is_set;
}

bool SortKeyBoundary::Update(string_t new_key) {
	Value value;
	{
		const annotated_lock_guard<annotated_mutex> guard {lock};
		if (is_set && !(new_key < string_t(key))) {
			return false;
		}
		key = new_key.GetString();
		is_set = true;
		version.store(version.load(std::memory_order_relaxed) + 1, std::memory_order_release);
		if (!dynamic_filter) {
			return true;
		}
		FlatVector::ValidityMutable(first_value).SetAllValid(1);
		CreateSortKeyHelpers::DecodeSortKey(new_key, first_value, 0, first_modifiers);
		value = first_value.GetValue(0);
		if (value.IsNull() || (!dynamic_value.IsNull() && ValueOperations::NotDistinctFrom(value, dynamic_value))) {
			return true;
		}
		dynamic_value = value;
	}
	dynamic_filter->SetValue(std::move(value));
	return true;
}

void SortKeyBoundary::Contribute(idx_t producer, vector<Quantile> quantiles, idx_t limit) {
	string bound;
	{
		const annotated_lock_guard<annotated_mutex> guard {lock};
		if (producer >= reports.size()) {
			reports.resize(producer + 1);
		}
		reports[producer] = std::move(quantiles);
		points.clear();
		idx_t total = 0;
		for (auto &report : reports) {
			idx_t counted = 0;
			for (auto &quantile : report) {
				if (!is_set || quantile.key < key) {
					points.push_back({&quantile.key, quantile.rows - counted});
					total += quantile.rows - counted;
				}
				counted = quantile.rows;
			}
		}
		if (total < limit) {
			return;
		}
		const auto by_key = [](const Point &l, const Point &r) {
			return *l.key < *r.key;
		};
		auto first = points.begin();
		auto last = points.end();
		auto need = limit;
		while (last - first > 1) {
			const auto mid = first + (last - first) / 2;
			std::nth_element(first, mid, last, by_key);
			idx_t left = 0;
			for (auto it = first; it != mid; ++it) {
				left += it->rows;
			}
			if (left >= need) {
				last = mid;
			} else if (left + mid->rows >= need) {
				first = mid;
				break;
			} else {
				need -= left + mid->rows;
				first = mid + 1;
			}
		}
		bound = *first->key;
	}
	Update(string_t(bound));
}

void SortKeyBoundary::Reset() {
	const annotated_lock_guard<annotated_mutex> guard {lock};
	key.clear();
	is_set = false;
	reports.clear();
	dynamic_value = Value();
	version.store(version.load(std::memory_order_relaxed) + 1, std::memory_order_release);
	if (dynamic_filter) {
		dynamic_filter->Reset();
	}
}

SortKeyBoundaryFilter::SortKeyBoundaryFilter(Allocator &allocator, const vector<LogicalType> &types,
                                             vector<OrderModifiers> modifiers_p)
    : modifiers(std::move(modifiers_p)), comparison_cache(allocator, LogicalType::TINYINT),
      comparison(comparison_cache), ties(STANDARD_VECTOR_SIZE) {
	D_ASSERT(modifiers.size() == types.size());
	compare_chunk.InitializeEmpty(types);
	boundary_values.Initialize(allocator, types);
}

bool SortKeyBoundaryFilter::Refresh(const SortKeyBoundary &boundary) {
	const auto current = boundary.Version();
	if (current == version) {
		return is_set;
	}
	version = current;
	is_set = boundary.Get(key);
	if (is_set) {
		boundary_values.Reset();
		CreateSortKeyHelpers::DecodeSortKey(string_t(key), boundary_values, 0, modifiers);
		for (auto &column : boundary_values.data) {
			column.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
	}
	return is_set;
}

void SortKeyBoundaryFilter::Compare(const Vector &input, idx_t col, Vector &result, idx_t count) {
	const auto &modifier = modifiers[col];
	const bool descending = modifier.order_type == OrderType::DESCENDING;
	if (descending == (modifier.null_type == OrderByNullType::NULLS_FIRST)) {
		VectorOperations::DistinctComparatorFill(input, boundary_values.data[col], result, count);
	} else {
		VectorOperations::DistinctComparatorNullsFirstFill(input, boundary_values.data[col], result, count);
	}
}

const int8_t *SortKeyBoundaryFilter::DictionaryOrders(const Vector &input, idx_t count) {
	static constexpr idx_t MAX_DICTIONARY_SIZE_THRESHOLD = 20000;
	if (input.GetVectorType() != VectorType::DICTIONARY_VECTOR) {
		return nullptr;
	}
	const auto size = DictionaryVector::DictionarySize(input);
	const auto &id = DictionaryVector::DictionaryId(input);
	if (!size.IsValid() || id.empty() || size.GetIndex() >= MAX_DICTIONARY_SIZE_THRESHOLD) {
		return nullptr;
	}
	const auto dictionary_size = size.GetIndex();
	if (id != dictionary_id || version != dictionary_version) {
		if (dictionary_size > STANDARD_VECTOR_SIZE && 2 * count <= STANDARD_VECTOR_SIZE) {
			return nullptr;
		}
		if (dictionary_size > dictionary_capacity) {
			dictionary_orders = make_uniq<Vector>(LogicalType::TINYINT, dictionary_size);
			dictionary_capacity = dictionary_size;
		}
		dictionary_id.clear();
		boundary_values.SetChildCardinality(dictionary_size);
		Compare(DictionaryVector::Child(input), 0, *dictionary_orders, dictionary_size);
		boundary_values.SetChildCardinality(count);
		dictionary_orders->Flatten();
		dictionary_id = id;
		dictionary_version = version;
	}
	return FlatVector::GetData<int8_t>(*dictionary_orders);
}

idx_t SortKeyBoundaryFilter::Admit(DataChunk &keys, SelectionVector &sel) {
	D_ASSERT(is_set);
	const auto count = keys.size();
	boundary_values.SetChildCardinality(count);
	const auto after_order = modifiers[0].order_type == OrderType::DESCENDING ? int8_t(-1) : int8_t(1);
	auto &input = keys.data[0];
	bool any_after = false;
	bool any_tie = false;
	if (const auto orders = DictionaryOrders(input, count)) {
		const auto &rows = DictionaryVector::SelVector(input);
		for (idx_t i = 0; i < count; i++) {
			const auto order = orders[rows.get_index(i)];
			any_after |= order == after_order;
			any_tie |= order == 0;
		}
	} else {
		comparison.ResetFromCache(comparison_cache);
		Compare(input, 0, comparison, count);
		if (comparison.GetVectorType() != VectorType::FLAT_VECTOR) {
			idx_t before;
			return Select(keys, true, sel, before);
		}
		const auto compared = FlatVector::GetData<int8_t>(comparison);
		for (idx_t i = 0; i < count; i++) {
			any_after |= compared[i] == after_order;
			any_tie |= compared[i] == 0;
		}
	}
	if (!any_after && (!any_tie || modifiers.size() == 1)) {
		return count;
	}
	idx_t before;
	return Select(keys, true, sel, before);
}

idx_t SortKeyBoundaryFilter::Select(DataChunk &keys, bool inclusive, SelectionVector &sel, idx_t &before) {
	D_ASSERT(is_set);
	boundary_values.SetChildCardinality(keys.size());
	idx_t selected = 0;
	idx_t remaining = keys.size();
	bool all_rows = true;
	for (idx_t col = 0; col < modifiers.size() && remaining > 0; col++) {
		auto &input = compare_chunk.data[col];
		if (all_rows) {
			input.Reference(keys.data[col]);
		} else {
			input.Slice(keys.data[col], ties, remaining);
		}
		const bool descending = modifiers[col].order_type == OrderType::DESCENDING;
		comparison.ResetFromCache(comparison_cache);
		const auto dictionary = col == 0 ? DictionaryOrders(input, remaining) : nullptr;
		if (dictionary) {
			const auto &rows = DictionaryVector::SelVector(input);
			auto result = FlatVector::GetDataMutable<int8_t>(comparison);
			for (idx_t i = 0; i < remaining; i++) {
				result[i] = dictionary[rows.get_index(i)];
			}
			FlatVector::SetSize(comparison, remaining);
		} else {
			Compare(input, col, comparison, remaining);
		}
		const auto before_order = descending ? int8_t(1) : int8_t(-1);
		auto sel_data = sel.data();
		auto tie_data = ties.data();
		if (comparison.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			const auto order = *ConstantVector::GetData<int8_t>(comparison);
			if (order == 0) {
				continue;
			}
			if (order == before_order) {
				for (idx_t i = 0; i < remaining; i++) {
					sel_data[selected++] = all_rows ? sel_t(i) : tie_data[i];
				}
			}
			remaining = 0;
			break;
		}
		const auto orders = FlatVector::GetData<int8_t>(comparison);
		int8_t any_order = 0;
		for (idx_t i = 0; i < remaining; i++) {
			any_order |= orders[i];
		}
		if (any_order == 0) {
			continue;
		}
		idx_t tie_count = 0;
		for (idx_t i = 0; i < remaining; i++) {
			const auto row = all_rows ? sel_t(i) : tie_data[i];
			sel_data[selected] = row;
			selected += orders[i] == before_order;
			tie_data[tie_count] = row;
			tie_count += orders[i] == 0;
		}
		remaining = tie_count;
		all_rows = false;
	}
	before = selected;
	if (inclusive) {
		auto sel_data = sel.data();
		for (idx_t i = 0; i < remaining; i++) {
			sel_data[selected++] = all_rows ? sel_t(i) : ties.get_index(i);
		}
	}
	return selected;
}

} // namespace duckdb
