//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/topn_optimizer.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"

namespace duckdb {
class ClientContext;
class LogicalOperator;
class LogicalTopN;
class Optimizer;

class TopN {
public:
	explicit TopN(ClientContext &context);

	//! Optimize ORDER BY + LIMIT to TopN
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);
	//! Whether we can perform the optimization on this operator
	static bool CanOptimize(LogicalOperator &op, optional_ptr<ClientContext> context = nullptr);
	static shared_ptr<DynamicFilterData> PushDynamicFilterIntoScans(ClientContext &context, LogicalOperator &child,
	                                                                ColumnBinding binding, const LogicalType &type,
	                                                                bool nulls_first, ExpressionType comparison_type);

private:
	void PushdownDynamicFilters(LogicalTopN &op);
	void PushdownGroupLimitFilter(LogicalLimit &limit);
	bool PushdownGroupOrderBound(LogicalTopN &op);

private:
	ClientContext &context;
};

} // namespace duckdb
