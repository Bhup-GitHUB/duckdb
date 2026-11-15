#include "duckdb/common/sorting/sort_key_boundary.hpp"

#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/common/vector/constant_vector.hpp"
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

void SortKeyBoundary::Reset() {
	const annotated_lock_guard<annotated_mutex> guard {lock};
	key.clear();
	is_set = false;
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
		const auto &modifier = modifiers[col];
		const bool descending = modifier.order_type == OrderType::DESCENDING;
		comparison.ResetFromCache(comparison_cache);
		if (descending == (modifier.null_type == OrderByNullType::NULLS_FIRST)) {
			VectorOperations::DistinctComparatorFill(input, boundary_values.data[col], comparison, remaining);
		} else {
			VectorOperations::DistinctComparatorNullsFirstFill(input, boundary_values.data[col], comparison, remaining);
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
