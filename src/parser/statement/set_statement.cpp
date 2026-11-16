#include "duckdb/parser/statement/set_statement.hpp"

#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/keyword_helper.hpp"

namespace duckdb {

SetStatement::SetStatement(Identifier name_p, SetScope scope_p, SetType type_p)
    : SQLStatement(StatementType::SET_STATEMENT), name(std::move(name_p)), scope(scope_p), set_type(type_p) {
}

// Set Variable

SetVariableStatement::SetVariableStatement(Identifier name_p, unique_ptr<ParsedExpression> value_p, SetScope scope_p)
    : SetStatement(std::move(name_p), scope_p, SetType::SET), value(std::move(value_p)) {
}

SetVariableStatement::SetVariableStatement(const SetVariableStatement &other)
    : SetStatement(other), value(other.value->Copy()) {
}

unique_ptr<SQLStatement> SetVariableStatement::Copy() const {
	return unique_ptr<SetVariableStatement>(new SetVariableStatement(*this));
}

static string ScopeToString(SetScope scope) {
	switch (scope) {
	case SetScope::AUTOMATIC:
		return "";
	case SetScope::LOCAL:
		return "LOCAL";
	case SetScope::SESSION:
		return "SESSION";
	case SetScope::GLOBAL:
		return "GLOBAL";
	case SetScope::VARIABLE:
		return "VARIABLE";
	default:
		throw InternalException("ToString not implemented for SetScope of type: %s", EnumUtil::ToString(scope));
	}
}

static string ScopePrefix(SetScope scope) {
	auto result = ScopeToString(scope);
	return result.empty() ? result : result + " ";
}

static bool IsSearchPathList(const Identifier &name, const ParsedExpression &value) {
	if (name != "search_path" || value.GetExpressionType() != ExpressionType::VALUE_CONSTANT) {
		return false;
	}
	auto &literal = value.Cast<ConstantExpression>().GetLiteral();
	return literal.kind == LiteralKind::STRING && !literal.text.empty();
}

string SetVariableStatement::ToString() const {
	auto result = "SET " + ScopePrefix(scope) + SQLIdentifier::ToString(name) + " TO ";
	if (IsSearchPathList(name, *value)) {
		result += CatalogSearchEntry::ListToString(
		    CatalogSearchEntry::ParseList(value->Cast<ConstantExpression>().GetLiteral().text));
	} else {
		result += value->ToString();
	}
	return result + ";";
}

// Reset Variable

ResetVariableStatement::ResetVariableStatement(Identifier name_p, SetScope scope_p)
    : SetStatement(std::move(name_p), scope_p, SetType::RESET) {
}

unique_ptr<SQLStatement> ResetVariableStatement::Copy() const {
	return unique_ptr<ResetVariableStatement>(new ResetVariableStatement(*this));
}

string ResetVariableStatement::ToString() const {
	return "RESET " + ScopePrefix(scope) + (name.empty() ? "ALL" : SQLIdentifier::ToString(name)) + ";";
}

} // namespace duckdb
