//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/filter/table_filter_dynamic_function.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/planner/filter/table_filter_functions.hpp"
#include "duckdb/planner/filter/table_filter_function_helpers.hpp"

#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/filter/zonemap_checker.hpp"

namespace duckdb {

DynamicFilterData::DynamicFilterData(ExpressionType comparison_type_p, Value constant_p)
    : comparison_type(comparison_type_p), constant(std::move(constant_p)) {
}

unique_ptr<Expression> DynamicFilterData::ToExpression(const Expression &column) const {
	return BoundComparisonExpression::Create(comparison_type, column.Copy(),
	                                         make_uniq<BoundConstantExpression>(constant));
}

void DynamicFilterData::SetValue(Value val) {
	if (val.IsNull()) {
		return;
	}
	lock_guard<mutex> l(lock);
	constant = std::move(val);
	version.store(version.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

void DynamicFilterData::Reset() {
	lock_guard<mutex> l(lock);
	version.store(0, std::memory_order_release);
}

DynamicFilterFunctionData::DynamicFilterFunctionData(shared_ptr<DynamicFilterData> filter_data_p)
    : filter_data(std::move(filter_data_p)) {
}

unique_ptr<FunctionData> DynamicFilterFunctionData::Copy() const {
	return make_uniq<DynamicFilterFunctionData>(filter_data);
}

bool DynamicFilterFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<DynamicFilterFunctionData>();
	return filter_data.get() == other.filter_data.get();
}

struct DynamicFilterLocalState : public FunctionLocalState {
	idx_t version = 0;
	unique_ptr<Vector> constant;
};

static unique_ptr<FunctionLocalState>
DynamicFilterInitLocalState(ExpressionState &state, const BoundFunctionExpression &expr, FunctionData *bind_data) {
	return make_uniq<DynamicFilterLocalState>();
}

static idx_t SelectDynamicFilter(const Vector &input, ExpressionType comparison_type, const Vector &constant,
                                 SelectionVector &result_sel, idx_t count) {
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		return VectorOperations::Equals(input, constant, nullptr, count, &result_sel, nullptr);
	case ExpressionType::COMPARE_NOTEQUAL:
		return VectorOperations::NotEquals(input, constant, nullptr, count, &result_sel, nullptr);
	case ExpressionType::COMPARE_GREATERTHAN:
		return VectorOperations::GreaterThan(input, constant, nullptr, count, &result_sel, nullptr);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return VectorOperations::GreaterThanEquals(input, constant, nullptr, count, &result_sel, nullptr);
	case ExpressionType::COMPARE_LESSTHAN:
		return VectorOperations::LessThan(input, constant, nullptr, count, &result_sel, nullptr);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return VectorOperations::LessThanEquals(input, constant, nullptr, count, &result_sel, nullptr);
	default:
		throw InternalException("Unknown comparison type for DynamicFilter: " + EnumUtil::ToString(comparison_type));
	}
}

static idx_t DynamicFilterSelect(DataChunk &args, ExpressionState &state, optional_ptr<const SelectionVector> sel,
                                 optional_ptr<SelectionVector> true_sel, optional_ptr<SelectionVector> false_sel) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &func_data = func_expr.BindInfo()->Cast<DynamicFilterFunctionData>();
	auto count = args.size();
	if (!func_data.filter_data) {
		return SetAllTrueSelection(count, sel, true_sel, false_sel);
	}

	auto &filter_data = *func_data.filter_data;
	auto &input = args.data[0];
	DynamicFilterLocalState call_state;
	auto local_state_ptr = ExecuteFunctionState::GetFunctionState(state);
	auto &local = local_state_ptr ? local_state_ptr->Cast<DynamicFilterLocalState>() : call_state;
	if (filter_data.version.load(std::memory_order_acquire) != local.version) {
		Value constant;
		{
			lock_guard<mutex> lock(filter_data.lock);
			constant = filter_data.constant;
			local.version = filter_data.version.load(std::memory_order_relaxed);
		}
		local.constant.reset();
		if (local.version != 0 && constant.type() == input.GetType()) {
			local.constant = make_uniq<Vector>(constant, count_t(count));
		}
	}
	if (!local.constant) {
		return SetAllTrueSelection(count, sel, true_sel, false_sel);
	}
	FlatVector::SetSize(*local.constant, count);

	SelectionVector temp_true(count);
	auto result_true_sel = (!true_sel || (sel && true_sel.get() == sel.get())) ? &temp_true : true_sel.get();
	const auto approved_count =
	    SelectDynamicFilter(input, filter_data.comparison_type, *local.constant, *result_true_sel, count);
	return TranslateSelection(count, sel, *result_true_sel, approved_count, true_sel, false_sel);
}

ScalarFunction DynamicFilterScalarFun::GetFunction(const LogicalType &input_type) {
	ScalarFunction func(NAME, {input_type}, LogicalType::BOOLEAN, nullptr, TableFilterFunctions::Bind);
	func.GetSignature().GetParameter(0).SetName("col");
	func.SetInitStateCallback(DynamicFilterInitLocalState);
	func.SetSelectCallback(DynamicFilterSelect);
	func.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	func.SetFilterPruneCallback(DynamicFilterScalarFun::FilterPrune);
	func.SetSerializeCallback(TableFilterFunctionSerialize);
	func.SetDeserializeCallback(TableFilterFunctionDeserialize);
	func.SetToStringCallback(TableFilterFunctionToString);
	return func;
}

FilterPropagateResult DynamicFilterScalarFun::FilterPrune(const FunctionStatisticsPruneInput &input) {
	if (!input.bind_data) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	auto &data = input.bind_data->Cast<DynamicFilterFunctionData>();
	if (!data.filter_data) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	lock_guard<mutex> lock(data.filter_data->lock);
	if (data.filter_data->version.load(std::memory_order_relaxed) == 0) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	auto column_stats = input.ChildStats(0);
	if (!column_stats || column_stats->GetType() != data.filter_data->constant.type()) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	return ZonemapChecker::CompileComparison(data.filter_data->comparison_type, data.filter_data->constant)
	    ->Check(*column_stats, nullptr);
}

string DynamicFilterScalarFun::ToString(const string &column_name, optional_ptr<const DynamicFilterData> filter_data) {
	if (filter_data) {
		// The comparison operator is fixed at plan time; the bound (?) is filled in at runtime.
		return "Dynamic Filter (" + column_name + " " + ExpressionTypeToOperator(filter_data->comparison_type) + " ?)";
	}
	return "Dynamic Filter";
}

ScalarFunction TableFilterDynamicFun::GetFunction() {
	return DynamicFilterScalarFun::GetFunction(LogicalType::ANY);
}

} // namespace duckdb
