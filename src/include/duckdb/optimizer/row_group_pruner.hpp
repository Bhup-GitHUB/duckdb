//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/row_group_pruner.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include "duckdb/storage/table/row_group_order_options.hpp"

namespace duckdb {
class ClientContext;
class LogicalGet;
class LogicalOperator;

class RowGroupPruner {
public:
	explicit RowGroupPruner(ClientContext &context);

	//! Reorder and try to prune row groups in queries with LIMIT or simple aggregates
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);
	//! Whether we can perform the optimization on this operator
	bool TryOptimize(LogicalOperator &op) const;
	static bool OrderScan(ClientContext &context, LogicalOperator &child, const BoundColumnRefExpression &key,
	                      OrderType order_type, OrderByNullType null_order);

private:
	ClientContext &context;

private:
	void GetLimitAndOffset(const LogicalLimit &logical_limit, optional_idx &row_limit, optional_idx &row_offset) const;
	optional_ptr<LogicalOrder> FindLogicalOrder(const LogicalLimit &logical_limit) const;
	static optional_ptr<LogicalGet> FindLogicalGet(LogicalOperator &child, const BoundColumnRefExpression &key,
	                                               ColumnIndex &column_index);
	// row_limit, row_offset, primary_order, logical_get, logical_limit
	unique_ptr<RowGroupOrderOptions>
	CreateRowGroupReordererOptions(optional_idx row_limit, optional_idx row_offset,
	                               const BoundOrderByNode &primary_order, const LogicalGet &logical_get,
	                               const StorageIndex &storage_index, LogicalLimit &logical_limit,
	                               const bool single_order_key, const bool push_limit) const;
};

} // namespace duckdb
