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
#include "duckdb/common/types/vector_cache.hpp"
#include "duckdb/planner/group_limit.hpp"

namespace duckdb {

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

	void Publish(const vector<string> &candidates, const string &limit_key);
	bool Freeze(vector<string> &keys);
	void Reset();

private:
	vector<column_t> columns;
	vector<LogicalType> types;
	vector<OrderModifiers> order;
	idx_t limit;
	bool ordered;
	SortKeyBoundary boundary;
	annotated_mutex lock;
	vector<string> keys DUCKDB_GUARDED_BY(lock);
	bool frozen DUCKDB_GUARDED_BY(lock) = false;
};

class GroupLimitBoundSink {
public:
	GroupLimitBoundSink(Allocator &allocator, const GroupLimitBound &bound);

	idx_t Select(GroupLimitBound &bound, DataChunk &groups, SelectionVector &sel);
	void AddCandidates(GroupLimitBound &bound, DataChunk &groups);

private:
	void Compact(idx_t limit);

	SortKeyBoundaryFilter filter;
	DataChunk ordered;
	bool filtered = false;
	SelectionVector candidate_sel;
	DataChunk candidate_groups;
	VectorCache keys_cache;
	Vector keys;
	vector<string> candidates;
	string limit_key;
	bool full = false;
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
