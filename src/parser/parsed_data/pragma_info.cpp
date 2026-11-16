#include "duckdb/parser/parsed_data/pragma_info.hpp"

#include <absl/algorithm/container.h>

#include "duckdb/parser/keyword_helper.hpp"

namespace duckdb {

unique_ptr<PragmaInfo> PragmaInfo::Copy() const {
	auto result = make_uniq<PragmaInfo>();
	result->name = name;
	for (auto &param : parameters) {
		result->parameters.push_back(param->Copy());
	}
	for (auto &entry : named_parameters) {
		result->named_parameters.insert(make_pair(entry.first, entry.second->Copy()));
	}
	return result;
}

string PragmaInfo::ToString() const {
	string result = "";
	result += "PRAGMA";
	// FIXME: Can pragma's live in different catalog/schemas ?
	result += " " + SQLIdentifier(name);
	if (!parameters.empty() || !named_parameters.empty()) {
		vector<string> stringified;
		for (auto &param : parameters) {
			stringified.push_back(param->ToString());
		}
		vector<string> named;
		for (auto &entry : named_parameters) {
			named.push_back(SQLIdentifier::ToString(entry.first) + " = " + entry.second->ToString());
		}
		absl::c_sort(named);
		stringified.insert(stringified.end(), named.begin(), named.end());
		result += "(" + StringUtil::Join(stringified, ", ") + ")";
	}
	result += ";";
	return result;
}

} // namespace duckdb
