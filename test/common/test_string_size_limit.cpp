#include "catch.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector/string_vector.hpp"

using namespace duckdb;

TEST_CASE("Adding a string over the maximum string size fails before its length is narrowed", "[string]") {
	Vector vector(LogicalType::VARCHAR);
	const char data[] = "x";
	const idx_t size = string_t::MAX_STRING_SIZE + 1;
	REQUIRE_THROWS_AS(StringVector::AddString(vector, data, size), ObjectSizeException);
	REQUIRE_THROWS_AS(StringVector::AddStringOrBlob(vector, data, size), ObjectSizeException);
	REQUIRE_THROWS_AS(StringVector::AddString(vector, std::string_view(data, size)), ObjectSizeException);
	REQUIRE_THROWS_AS(string_t::Safe({data, size}), ObjectSizeException);
	REQUIRE(StringVector::AddString(vector, data, 1).GetString() == "x");
	REQUIRE(string_t::Safe(data).GetString() == "x");
}
