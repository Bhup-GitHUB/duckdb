#include "duckdb/optimizer/statistics_propagator.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/optimizer/expression_rewriter.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/storage/statistics/string_stats.hpp"

namespace duckdb {

static bool TryGetStringStatsMax(const BaseStatistics &stats, string &max) {
	max = string(StringStats::Max(stats));
	// A truncated max stores a prefix. Use the next prefix as its upper bound.
	return StringStats::GetMaxType(stats) == StringStatsType::EXACT_STATS || StringUtil::FindNextPrefix(max);
}

template <class T>
static bool Greater(const T *left, const T *right) {
	return left && right && GreaterThan::Operation(*left, *right);
}

template <class T>
static bool GreaterOrEqual(const T *left, const T *right) {
	return left && right && GreaterThanEquals::Operation(*left, *right);
}

template <class T>
static FilterPropagateResult PropagateRangeComparison(const T *lmin, const T *lmax, const T *rmin, const T *rmax,
                                                      ExpressionType comparison, bool has_null) {
	const auto always_true =
	    has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
	const auto always_false =
	    has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
	const auto disjoint = Greater(lmin, rmax) || Greater(rmin, lmax);
	switch (comparison) {
	case ExpressionType::COMPARE_EQUAL:
		return disjoint ? always_false : FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_NOTEQUAL:
		if (lmin && lmax && rmin && rmax && Equals::Operation(*lmin, *lmax) && Equals::Operation(*rmin, *rmax) &&
		    Equals::Operation(*lmin, *rmin)) {
			return always_false;
		}
		return disjoint ? always_true : FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_GREATERTHAN:
		if (Greater(lmin, rmax)) {
			return always_true;
		}
		return GreaterOrEqual(rmin, lmax) ? always_false : FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		if (GreaterOrEqual(lmin, rmax)) {
			return always_true;
		}
		return Greater(rmin, lmax) ? always_false : FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_LESSTHAN:
		if (Greater(rmin, lmax)) {
			return always_true;
		}
		return GreaterOrEqual(lmin, rmax) ? always_false : FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		if (GreaterOrEqual(rmin, lmax)) {
			return always_true;
		}
		return Greater(lmin, rmax) ? always_false : FilterPropagateResult::NO_PRUNING_POSSIBLE;
	default:
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
}

FilterPropagateResult StatisticsPropagator::PropagateComparison(const BaseStatistics &lstats,
                                                                const BaseStatistics &rstats,
                                                                ExpressionType comparison) {
	if (lstats.GetStatsType() == StatisticsType::STRING_STATS &&
	    rstats.GetStatsType() == StatisticsType::STRING_STATS) {
		if (lstats.GetType() != rstats.GetType()) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
		string lmax_storage;
		string rmax_storage;
		string_t lmin_value;
		string_t lmax_value;
		string_t rmin_value;
		string_t rmax_value;
		const string_t *lmin = nullptr;
		const string_t *lmax = nullptr;
		const string_t *rmin = nullptr;
		const string_t *rmax = nullptr;
		if (StringStats::HasMin(lstats)) {
			lmin_value = StringStats::Min(lstats);
			lmin = &lmin_value;
		}
		if (StringStats::HasMax(lstats) && TryGetStringStatsMax(lstats, lmax_storage)) {
			lmax_value = lmax_storage;
			lmax = &lmax_value;
		}
		if (StringStats::HasMin(rstats)) {
			rmin_value = StringStats::Min(rstats);
			rmin = &rmin_value;
		}
		if (StringStats::HasMax(rstats) && TryGetStringStatsMax(rstats, rmax_storage)) {
			rmax_value = rmax_storage;
			rmax = &rmax_value;
		}
		const bool has_null = lstats.CanHaveNull() || rstats.CanHaveNull();
		return PropagateRangeComparison(lmin, lmax, rmin, rmax, comparison, has_null);
	}
	if (lstats.GetStatsType() != StatisticsType::NUMERIC_STATS || !NumericStats::HasMinMax(lstats) ||
	    !NumericStats::HasMinMax(rstats)) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	const bool has_null = lstats.CanHaveNull() || rstats.CanHaveNull();
	return VisitPhysicalType(lstats.GetType().InternalType(), [&]<class T>() {
		const auto lmin = NumericStats::GetMinUnsafe<T>(lstats);
		const auto lmax = NumericStats::GetMaxUnsafe<T>(lstats);
		const auto rmin = NumericStats::GetMinUnsafe<T>(rstats);
		const auto rmax = NumericStats::GetMaxUnsafe<T>(rstats);
		return PropagateRangeComparison(&lmin, &lmax, &rmin, &rmax, comparison, has_null);
	});
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateComparison(BoundFunctionExpression &expr,
                                                                     unique_ptr<Expression> &expr_ptr) {
	auto &left = BoundComparisonExpression::LeftMutable(expr);
	auto &right = BoundComparisonExpression::RightMutable(expr);
	auto left_stats = PropagateExpression(left);
	auto right_stats = PropagateExpression(right);
	if (!left_stats || !right_stats) {
		return nullptr;
	}
	// propagate the statistics of the comparison operator
	auto propagate_result = PropagateComparison(*left_stats, *right_stats, expr.GetExpressionType());
	switch (propagate_result) {
	case FilterPropagateResult::FILTER_ALWAYS_TRUE:
		expr_ptr = make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
		return PropagateExpression(expr_ptr);
	case FilterPropagateResult::FILTER_ALWAYS_FALSE:
		expr_ptr = make_uniq<BoundConstantExpression>(Value::BOOLEAN(false));
		return PropagateExpression(expr_ptr);
	case FilterPropagateResult::FILTER_TRUE_OR_NULL: {
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(left));
		children.push_back(std::move(right));
		expr_ptr = ExpressionRewriter::ConstantOrNull(context, std::move(children), Value::BOOLEAN(true));
		return nullptr;
	}
	case FilterPropagateResult::FILTER_FALSE_OR_NULL: {
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(left));
		children.push_back(std::move(right));
		expr_ptr = ExpressionRewriter::ConstantOrNull(context, std::move(children), Value::BOOLEAN(false));
		return nullptr;
	}
	default:
		// FIXME: we can propagate nulls here, i.e. this expression will have nulls only if left and right has nulls
		return nullptr;
	}
}

} // namespace duckdb
