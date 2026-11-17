#include "duckdb/execution/operator/helper/physical_execute.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/planner/operator/logical_execute.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalExecute &op) {
	if (op.prepared->physical_plan) {
		D_ASSERT(op.children.empty());
		auto &plan = op.prepared->physical_plan->Root();
		return Make<PhysicalExecute>(plan, op.prepared);
	}

	D_ASSERT(op.children.size() == 1);
	if (op.replaces) {
		PhysicalPlanGenerator inner_planner(context);
		op.prepared->physical_plan = inner_planner.PlanInternal(*op.children[0]);
		op.prepared->types = op.prepared->physical_plan->Root().types;
		auto &statements = ClientData::Get(context).prepared_statements;
		const auto replaced = statements[*op.replaces];
		for (auto &entry : statements) {
			if (entry.second == replaced) {
				entry.second = op.prepared;
			}
		}
		return Make<PhysicalExecute>(op.prepared->physical_plan->Root(), op.prepared);
	}
	auto &plan = CreatePlan(*op.children[0]);
	return Make<PhysicalExecute>(plan, op.prepared);
}

} // namespace duckdb
