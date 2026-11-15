//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/group_limit.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_idx.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"

namespace duckdb {

struct GroupLimit {
	optional_idx limit;
	vector<idx_t> columns;
	vector<OrderModifiers> order;
	shared_ptr<DynamicFilterData> filter;
};

} // namespace duckdb
