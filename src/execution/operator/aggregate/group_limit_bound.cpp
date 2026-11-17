#include "duckdb/execution/operator/aggregate/group_limit_bound.hpp"

#include "duckdb/common/sorting/sort_key.hpp"
#include "duckdb/common/vector/vector_iterator.hpp"
#include "pdqsort.h"

#include <absl/algorithm/container.h>

#include <algorithm>

namespace duckdb {

namespace {

idx_t KeyWidth(const vector<LogicalType> &types) {
	idx_t width = 0;
	for (auto &type : types) {
		const auto physical = type.InternalType();
		if (!TypeIsConstantSize(physical)) {
			return DConstants::INVALID_INDEX;
		}
		width += 1 + GetTypeIdSize(physical);
	}
	return width;
}

template <class KEY>
bool KeepSmallest(vector<KEY> &keys, idx_t limit, bool may_repeat) {
	while (true) {
		if (keys.size() < limit) {
			if (may_repeat) {
				duckdb_pdqsort::pdqsort_branchless(keys.begin(), keys.end());
				keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
			}
			return false;
		}
		const auto end = keys.begin() + NumericCast<int64_t>(limit);
		duckdb_pdqsort::nth_element_branchless(keys.begin(), end - 1, keys.end());
		if (!may_repeat) {
			break;
		}
		duckdb_pdqsort::pdqsort_branchless(keys.begin(), end);
		const auto distinct_end = std::unique(keys.begin(), end);
		if (distinct_end == end) {
			break;
		}
		keys.erase(distinct_end, end);
	}
	keys.resize(limit);
	return true;
}

template <class SORT_KEY>
class FixedGroupBoundKeys final : public GroupBoundKeys {
public:
	FixedGroupBoundKeys(idx_t limit, idx_t width_p)
	    : GroupBoundKeys(limit), width(width_p),
	      encoded(make_unsafe_uniq_array_uninitialized<SORT_KEY>(STANDARD_VECTOR_SIZE)),
	      targets(make_unsafe_uniq_array_uninitialized<data_ptr_t>(STANDARD_VECTOR_SIZE)) {
	}

	bool Add(DataChunk &groups, const vector<OrderModifiers> &order) override {
		const auto count = groups.size();
		for (idx_t i = 0; i < count; i++) {
			encoded[i] = SORT_KEY();
			targets[i] = data_ptr_cast(&encoded[i].part0);
		}
		CreateSortKeyHelpers::CreateSortKey(groups, order, targets.get(), offsets);
		bool compacted = false;
		for (idx_t i = 0; i < count; i++) {
			encoded[i].ByteSwap();
			if (full && !(encoded[i] < limit_key)) {
				continue;
			}
			keys.push_back(encoded[i]);
			if (keys.size() >= next_compaction) {
				Compact();
				compacted = true;
			}
		}
		return compacted && full;
	}

	string_t LimitKey() override {
		Store(limit_key, limit_bytes);
		return string_t(const_char_ptr_cast(limit_bytes), UnsafeNumericCast<uint32_t>(width));
	}

	void AppendKeys(string &bytes, vector<idx_t> &ends) override {
		data_t key_bytes[SORT_KEY::INLINE_LENGTH];
		for (idx_t i = 0; i < limit; i++) {
			Store(keys[i], key_bytes);
			bytes.append(const_char_ptr_cast(key_bytes), width);
			ends.push_back(bytes.size());
		}
	}

private:
	static void Store(const SORT_KEY &key, data_ptr_t target) {
		for (idx_t i = 0; i < SORT_KEY::PARTS; i++) {
			duckdb::Store<uint64_t>(BSwapIfLE((&key.part0)[i]), target + i * sizeof(uint64_t));
		}
	}

	void Compact() {
		if (KeepSmallest(keys, limit, may_repeat)) {
			full = true;
			limit_key = keys.back();
		}
		next_compaction = 2 * MaxValue(limit, keys.size());
	}

	idx_t width;
	unsafe_unique_array<SORT_KEY> encoded;
	unsafe_unique_array<data_ptr_t> targets;
	unsafe_vector<idx_t> offsets;
	vector<SORT_KEY> keys;
	SORT_KEY limit_key;
	data_t limit_bytes[SORT_KEY::INLINE_LENGTH];
};

class StringGroupBoundKeys final : public GroupBoundKeys {
public:
	StringGroupBoundKeys(idx_t limit, Allocator &allocator_p)
	    : GroupBoundKeys(limit), allocator(allocator_p), encoded_cache(allocator, LogicalType::BLOB),
	      encoded(encoded_cache), heap(allocator) {
	}

	bool Add(DataChunk &groups, const vector<OrderModifiers> &order) override {
		encoded.ResetFromCache(encoded_cache);
		CreateSortKeyHelpers::CreateSortKey(groups, order, encoded);
		bool compacted = false;
		for (auto entry : encoded.Values<string_t>()) {
			const auto key = entry.GetValueUnsafe();
			if (full && !(key < limit_key)) {
				continue;
			}
			keys.push_back(heap.AddBlob(key));
			if (keys.size() >= next_compaction) {
				Compact();
				compacted = true;
			}
		}
		return compacted && full;
	}

	string_t LimitKey() override {
		return limit_key;
	}

	void AppendKeys(string &bytes, vector<idx_t> &ends) override {
		for (idx_t i = 0; i < limit; i++) {
			bytes.append(keys[i].GetData(), keys[i].GetSize());
			ends.push_back(bytes.size());
		}
	}

private:
	void Compact() {
		const auto kept_limit = KeepSmallest(keys, limit, may_repeat);
		StringHeap kept(allocator);
		for (auto &key : keys) {
			key = kept.AddBlob(key);
		}
		heap.Destroy();
		heap.Move(kept);
		if (kept_limit) {
			full = true;
			limit_key = keys.back();
		}
		next_compaction = 2 * MaxValue(limit, keys.size());
	}

	Allocator &allocator;
	VectorCache encoded_cache;
	Vector encoded;
	StringHeap heap;
	vector<string_t> keys;
	string_t limit_key;
};

} // namespace

static vector<column_t> BoundColumns(idx_t group_count, const GroupLimit &group_limit) {
	vector<column_t> result(group_limit.columns.begin(), group_limit.columns.end());
	for (column_t column = 0; column < group_count; column++) {
		if (absl::c_find(result, column) == result.end()) {
			result.push_back(column);
		}
	}
	return result;
}

static vector<OrderModifiers> BoundOrder(idx_t column_count, const GroupLimit &group_limit) {
	auto result = group_limit.order;
	while (result.size() < column_count) {
		result.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST);
	}
	return result;
}

GroupLimitBound::GroupLimitBound(const vector<LogicalType> &group_types, const GroupLimit &group_limit)
    : columns(BoundColumns(group_types.size(), group_limit)), order(BoundOrder(columns.size(), group_limit)),
      limit(group_limit.limit.GetIndex()), ordered(!group_limit.columns.empty()),
      boundary(group_types[columns[0]], order[0], group_limit.filter) {
	for (auto column : columns) {
		types.push_back(group_types[column]);
	}
	key_width = KeyWidth(types);
}

unique_ptr<GroupBoundKeys> GroupLimitBound::CreateKeys(Allocator &allocator) const {
	if (key_width <= 8) {
		return make_uniq<FixedGroupBoundKeys<SortKey<SortKeyType::NO_PAYLOAD_FIXED_8>>>(limit, key_width);
	}
	if (key_width <= 16) {
		return make_uniq<FixedGroupBoundKeys<SortKey<SortKeyType::NO_PAYLOAD_FIXED_16>>>(limit, key_width);
	}
	if (key_width <= 24) {
		return make_uniq<FixedGroupBoundKeys<SortKey<SortKeyType::NO_PAYLOAD_FIXED_24>>>(limit, key_width);
	}
	if (key_width <= 32) {
		return make_uniq<FixedGroupBoundKeys<SortKey<SortKeyType::NO_PAYLOAD_FIXED_32>>>(limit, key_width);
	}
	return make_uniq<StringGroupBoundKeys>(limit, allocator);
}

unique_ptr<GroupLimitBound> GroupLimitBound::TryCreate(const vector<LogicalType> &group_types,
                                                       const GroupLimit &group_limit) {
	if (!group_limit.limit.IsValid() || group_limit.limit.GetIndex() == 0 || group_types.empty() ||
	    (!group_limit.filter && group_limit.columns.empty())) {
		return nullptr;
	}
	return make_uniq<GroupLimitBound>(group_types, group_limit);
}

void GroupLimitBound::Publish(GroupBoundKeys &keys) {
	if (ordered) {
		boundary.Update(keys.LimitKey());
		return;
	}
	const annotated_lock_guard<annotated_mutex> guard {lock};
	if (frozen || !boundary.Update(keys.LimitKey())) {
		return;
	}
	key_bytes.clear();
	key_ends.clear();
	keys.AppendKeys(key_bytes, key_ends);
}

bool GroupLimitBound::Freeze(vector<string_t> &result) {
	const annotated_lock_guard<annotated_mutex> guard {lock};
	if (frozen) {
		return false;
	}
	frozen = true;
	result.clear();
	idx_t begin = 0;
	for (const auto end : key_ends) {
		result.emplace_back(key_bytes.data() + begin, UnsafeNumericCast<uint32_t>(end - begin));
		begin = end;
	}
	return true;
}

void GroupLimitBound::Reset() {
	boundary.Reset();
	const annotated_lock_guard<annotated_mutex> guard {lock};
	key_bytes.clear();
	key_ends.clear();
	frozen = false;
}

GroupLimitBoundSink::GroupLimitBoundSink(Allocator &allocator, const GroupLimitBound &bound)
    : filter(allocator, bound.Types(), bound.Order()), candidate_sel(STANDARD_VECTOR_SIZE),
      keys(bound.CreateKeys(allocator)) {
	ordered.InitializeEmpty(bound.Types());
	candidate_groups.InitializeEmpty(bound.Types());
}

idx_t GroupLimitBoundSink::Select(GroupLimitBound &bound, DataChunk &groups, SelectionVector &sel) {
	filtered = filter.Refresh(bound.Boundary());
	if (!filtered) {
		return groups.size();
	}
	ordered.ReferenceColumns(groups, bound.Columns());
	return filter.Admit(ordered, sel);
}

void GroupLimitBoundSink::AddNewGroups(GroupLimitBound &bound, GroupedAggregateHashTable &ht, idx_t new_groups,
                                       const vector<LogicalType> &group_types, bool abandoned) {
	if (created_groups >= bound.Limit()) {
		if (ht.Count() < table_groups + new_groups) {
			keys->MayRepeat();
		}
		table_groups = ht.Count();
		if (new_group_chunk.ColumnCount() == 0) {
			new_group_chunk.InitializeEmpty(group_types);
		}
		ht.SliceNewGroups(new_group_chunk, new_groups);
		Add(bound, new_group_chunk);
		return;
	}
	created_groups += new_groups;
	if (created_groups < bound.Limit()) {
		return;
	}
	if (abandoned) {
		keys->MayRepeat();
	}
	AggregateHTScanState scan_state;
	ht.InitializeScan(scan_state);
	DataChunk groups;
	groups.Initialize(Allocator::DefaultAllocator(), group_types);
	while (ht.ScanGroups(scan_state, groups)) {
		if (groups.size() != 0) {
			Add(bound, groups);
		}
	}
	table_groups = ht.Count();
}

void GroupLimitBoundSink::Add(GroupLimitBound &bound, DataChunk &groups) {
	ordered.ReferenceColumns(groups, bound.Columns());
	if (filtered) {
		idx_t before;
		filter.Select(ordered, false, candidate_sel, before);
		if (before == 0) {
			return;
		}
		candidate_groups.Slice(ordered, candidate_sel, before);
	} else {
		candidate_groups.Reference(ordered);
	}
	if (keys->Add(candidate_groups, bound.Order())) {
		bound.Publish(*keys);
	}
}

GroupLimitBoundSource::GroupLimitBoundSource(Allocator &allocator, const GroupLimitBound &bound)
    : filter(allocator, bound.Types(), bound.Order()) {
	ordered.InitializeEmpty(bound.Types());
}

optional_idx GroupLimitBoundSource::Select(const GroupLimitBound &bound, DataChunk &groups, SelectionVector &sel) {
	if (!filter.Refresh(bound.Boundary())) {
		return optional_idx();
	}
	ordered.ReferenceColumns(groups, bound.Columns());
	idx_t before;
	return filter.Select(ordered, true, sel, before);
}

} // namespace duckdb
