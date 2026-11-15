#include "duckdb/parser/constraints/foreign_key_constraint.hpp"

#include "duckdb/common/limits.hpp"
#include "duckdb/parser/keyword_helper.hpp"

namespace duckdb {

ForeignKeyConstraint::ForeignKeyConstraint() : Constraint(ConstraintType::FOREIGN_KEY) {
}

ForeignKeyConstraint::ForeignKeyConstraint(vector<Identifier> pk_columns, vector<Identifier> fk_columns,
                                           ForeignKeyInfo info)
    : Constraint(ConstraintType::FOREIGN_KEY), pk_columns(std::move(pk_columns)), fk_columns(std::move(fk_columns)),
      info(std::move(info)) {
}

string ForeignKeyConstraint::ToString() const {
	if (info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE ||
	    info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
		string base = "FOREIGN KEY (";

		for (idx_t i = 0; i < fk_columns.size(); i++) {
			if (i > 0) {
				base += ", ";
			}
			base += SQLIdentifier(fk_columns[i]);
		}
		base += ") REFERENCES ";
		if (!info.schema.empty() && info.schema != DEFAULT_SCHEMA) {
			base += SQLIdentifier(info.schema);
			base += ".";
		}
		base += SQLIdentifier(info.table);
		if (!pk_columns.empty()) {
			base += "(";

			for (idx_t i = 0; i < pk_columns.size(); i++) {
				if (i > 0) {
					base += ", ";
				}
				base += SQLIdentifier(pk_columns[i]);
			}
			base += ")";
		}

		return base;
	}

	return "";
}

unique_ptr<Constraint> ForeignKeyConstraint::Copy() const {
	return make_uniq<ForeignKeyConstraint>(*this);
}

bool ForeignKeyConstraint::Equals(const Constraint &other) const {
	if (!Constraint::Equals(other)) {
		return false;
	}
	auto &foreign_key = other.Cast<ForeignKeyConstraint>();
	if (info.IsAppendConstraint() != foreign_key.info.IsAppendConstraint()) {
		return false;
	}
	if (!info.IsAppendConstraint()) {
		return true;
	}
	const auto default_schema = [](const Identifier &schema) {
		return schema.empty() || schema == DEFAULT_SCHEMA;
	};
	const bool same_schema =
	    default_schema(info.schema) ? default_schema(foreign_key.info.schema) : info.schema == foreign_key.info.schema;
	return same_schema && info.table == foreign_key.info.table && fk_columns == foreign_key.fk_columns &&
	       pk_columns == foreign_key.pk_columns;
}

} // namespace duckdb
