//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/sorting/sort_key_boundary.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/vector_cache.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"

namespace duckdb {

class SortKeyBoundary {
public:
	SortKeyBoundary(const LogicalType &first_type, OrderModifiers first_modifiers,
	                shared_ptr<DynamicFilterData> dynamic_filter);

	idx_t Version() const {
		return version.load(std::memory_order_acquire);
	}
	bool Get(string &result) const;
	bool Update(string_t key);
	void Reset();

private:
	mutable annotated_mutex lock;
	string key DUCKDB_GUARDED_BY(lock);
	bool is_set DUCKDB_GUARDED_BY(lock) = false;
	atomic<idx_t> version {0};
	const shared_ptr<DynamicFilterData> dynamic_filter;
	const OrderModifiers first_modifiers;
	Vector first_value DUCKDB_GUARDED_BY(lock);
	Value dynamic_value DUCKDB_GUARDED_BY(lock);
};

class SortKeyBoundaryFilter {
public:
	SortKeyBoundaryFilter(Allocator &allocator, const vector<LogicalType> &types, vector<OrderModifiers> modifiers);

	bool Refresh(const SortKeyBoundary &boundary);
	idx_t Select(DataChunk &keys, bool inclusive, SelectionVector &sel, idx_t &before);
	const string &Key() const {
		return key;
	}

private:
	const vector<OrderModifiers> modifiers;
	DataChunk compare_chunk;
	DataChunk boundary_values;
	VectorCache comparison_cache;
	Vector comparison;
	SelectionVector ties;
	string key;
	idx_t version = 0;
	bool is_set = false;
};

} // namespace duckdb
