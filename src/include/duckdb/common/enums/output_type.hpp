//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/enums/output_type.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"

namespace duckdb {

enum class ExplainOutputType : uint8_t { ALL = 0, OPTIMIZED_ONLY = 1, PHYSICAL_ONLY = 2 };

// Column shape of EXPLAIN output: CLI = two columns {key, value};
// SERVER = single "QUERY PLAN" column with one row per plan line (PostgreSQL-compatible).
enum class ExplainFormatShape : uint8_t { CLI = 0, SERVER = 1 };

} // namespace duckdb
