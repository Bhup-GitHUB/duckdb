#include "duckdb/execution/operator/aggregate/group_limit_bound.hpp"

#include "duckdb/common/vector/vector_iterator.hpp"

#include <absl/algorithm/container.h>

#include <algorithm>

namespace duckdb {

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
}

unique_ptr<GroupLimitBound> GroupLimitBound::TryCreate(const vector<LogicalType> &group_types,
                                                       const GroupLimit &group_limit) {
	if (!group_limit.limit.IsValid() || group_limit.limit.GetIndex() == 0 || group_types.empty() ||
	    (!group_limit.filter && group_limit.columns.empty())) {
		return nullptr;
	}
	return make_uniq<GroupLimitBound>(group_types, group_limit);
}

void GroupLimitBound::Publish(const vector<string_t> &candidates, string_t limit_key) {
	const annotated_lock_guard<annotated_mutex> guard {lock};
	if (frozen || !boundary.Update(limit_key)) {
		return;
	}
	key_bytes.clear();
	key_ends.clear();
	for (idx_t i = 0; i < limit; i++) {
		key_bytes.append(candidates[i].GetData(), candidates[i].GetSize());
		key_ends.push_back(key_bytes.size());
	}
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
    : allocator(allocator), filter(allocator, bound.Types(), bound.Order()), candidate_sel(STANDARD_VECTOR_SIZE),
      keys_cache(allocator, LogicalType::BLOB), keys(keys_cache), candidate_heap(allocator) {
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

void GroupLimitBoundSink::AddCandidates(GroupLimitBound &bound, DataChunk &groups) {
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
	const auto limit = bound.Limit();
	keys.ResetFromCache(keys_cache);
	CreateSortKeyHelpers::CreateSortKey(candidate_groups, bound.Order(), keys);
	bool compacted = false;
	for (auto entry : keys.Values<string_t>()) {
		const auto key = entry.GetValueUnsafe();
		if (full && !(key < limit_key)) {
			continue;
		}
		candidates.push_back(candidate_heap.AddBlob(key));
		if (candidates.size() >= 2 * limit) {
			Compact(limit);
			compacted = true;
		}
	}
	if (!full && candidates.size() >= limit) {
		Compact(limit);
		compacted = true;
	}
	if (compacted && full) {
		bound.Publish(candidates, limit_key);
	}
}

void GroupLimitBoundSink::Compact(idx_t limit) {
	absl::c_sort(candidates);
	candidates.erase(std::ranges::unique(candidates).begin(), candidates.end());
	if (candidates.size() >= limit) {
		candidates.resize(limit);
		full = true;
	}
	StringHeap kept(allocator);
	for (auto &candidate : candidates) {
		candidate = kept.AddBlob(candidate);
	}
	candidate_heap.Destroy();
	candidate_heap.Move(kept);
	if (full) {
		limit_key = candidates.back();
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
