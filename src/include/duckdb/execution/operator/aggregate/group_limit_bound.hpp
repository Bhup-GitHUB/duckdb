//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/aggregate/group_limit_bound.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/sorting/sort_key_boundary.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/types/vector_cache.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/planner/group_limit.hpp"

namespace duckdb {

class GroupBoundKeys {
public:
	virtual ~GroupBoundKeys() = default;

	virtual bool Add(DataChunk &groups, const vector<OrderModifiers> &order) = 0;
	virtual string_t LimitKey() = 0;
	virtual void AppendKeys(string &bytes, vector<idx_t> &ends) = 0;

	void MayRepeat() {
		may_repeat = true;
	}

protected:
	explicit GroupBoundKeys(idx_t limit_p) : limit(limit_p), next_compaction(limit_p) {
	}

	const idx_t limit;
	bool full = false;
	bool may_repeat = false;
	idx_t next_compaction;
};

class GroupLimitBound {
public:
	GroupLimitBound(const vector<LogicalType> &group_types, const GroupLimit &group_limit);

	static unique_ptr<GroupLimitBound> TryCreate(const vector<LogicalType> &group_types, const GroupLimit &group_limit);

	bool Ordered() const {
		return ordered;
	}
	idx_t Limit() const {
		return limit;
	}
	const vector<column_t> &Columns() const {
		return columns;
	}
	const vector<LogicalType> &Types() const {
		return types;
	}
	const vector<OrderModifiers> &Order() const {
		return order;
	}
	const SortKeyBoundary &Boundary() const {
		return boundary;
	}

	unique_ptr<GroupBoundKeys> CreateKeys(Allocator &allocator) const;
	void Publish(GroupBoundKeys &keys);
	bool Freeze(vector<string_t> &keys);
	void Reset();

private:
	vector<column_t> columns;
	vector<LogicalType> types;
	vector<OrderModifiers> order;
	idx_t limit;
	bool ordered;
	idx_t key_width;
	SortKeyBoundary boundary;
	annotated_mutex lock;
	string key_bytes DUCKDB_GUARDED_BY(lock);
	vector<idx_t> key_ends DUCKDB_GUARDED_BY(lock);
	bool frozen DUCKDB_GUARDED_BY(lock) = false;
};

class GroupLimitBoundSink {
public:
	GroupLimitBoundSink(Allocator &allocator, const GroupLimitBound &bound);

	idx_t Select(GroupLimitBound &bound, DataChunk &groups, SelectionVector &sel);
	void AddNewGroups(GroupLimitBound &bound, GroupedAggregateHashTable &ht, idx_t new_groups,
	                  const vector<LogicalType> &group_types, bool abandoned);

private:
	void Add(GroupLimitBound &bound, DataChunk &groups);

	SortKeyBoundaryFilter filter;
	DataChunk ordered;
	bool filtered = false;
	SelectionVector candidate_sel;
	DataChunk candidate_groups;
	DataChunk new_group_chunk;
	unique_ptr<GroupBoundKeys> keys;
	idx_t table_groups = 0;
	idx_t created_groups = 0;
};

class GroupLimitBoundSource {
public:
	GroupLimitBoundSource(Allocator &allocator, const GroupLimitBound &bound);

	optional_idx Select(const GroupLimitBound &bound, DataChunk &groups, SelectionVector &sel);

private:
	SortKeyBoundaryFilter filter;
	DataChunk ordered;
};

} // namespace duckdb
