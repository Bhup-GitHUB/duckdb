//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/sorting/sort_key_boundary.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/vector_cache.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"

namespace duckdb {

class SortKeyBoundary {
public:
	SortKeyBoundary(const LogicalType &first_type, OrderModifiers first_modifiers,
	                shared_ptr<DynamicFilterData> dynamic_filter);

	idx_t Version() const {
		return version.load(std::memory_order_acquire);
	}
	bool Get(string &result) const;
	bool Update(string_t key);
	struct Quantile {
		string key;
		idx_t rows;
	};
	void Contribute(idx_t producer, vector<Quantile> quantiles, idx_t limit);
	idx_t NewProducer() {
		return producers.fetch_add(1, std::memory_order_relaxed);
	}
	void Reset();

private:
	struct Point {
		const string *key;
		idx_t rows;
	};

	mutable annotated_mutex lock;
	string key DUCKDB_GUARDED_BY(lock);
	bool is_set DUCKDB_GUARDED_BY(lock) = false;
	vector<vector<Quantile>> reports DUCKDB_GUARDED_BY(lock);
	vector<Point> points DUCKDB_GUARDED_BY(lock);
	atomic<idx_t> producers {0};
	atomic<idx_t> version {0};
	const shared_ptr<DynamicFilterData> dynamic_filter;
	const OrderModifiers first_modifiers;
	Vector first_value DUCKDB_GUARDED_BY(lock);
	Value dynamic_value DUCKDB_GUARDED_BY(lock);
};

class SortKeyBoundaryFilter {
public:
	SortKeyBoundaryFilter(Allocator &allocator, const vector<LogicalType> &types, vector<OrderModifiers> modifiers);

	bool Refresh(const SortKeyBoundary &boundary);
	idx_t Select(DataChunk &keys, bool inclusive, SelectionVector &sel, idx_t &before);
	idx_t Admit(DataChunk &keys, SelectionVector &sel);
	const string &Key() const {
		return key;
	}

private:
	void Compare(const Vector &input, idx_t col, Vector &result, idx_t count);
	const int8_t *DictionaryOrders(const Vector &input, idx_t count);
	void GatherOrders(const int8_t *dictionary, const Vector &input, idx_t count);
	void Orders(const Vector &input, idx_t col, idx_t count);
	idx_t SelectOrdered(DataChunk &keys, bool inclusive, SelectionVector &sel, idx_t &before);

private:
	const vector<OrderModifiers> modifiers;
	DataChunk compare_chunk;
	DataChunk boundary_values;
	VectorCache comparison_cache;
	Vector comparison;
	SelectionVector ties;
	string key;
	idx_t version = 0;
	bool is_set = false;
	unique_ptr<Vector> dictionary_orders;
	idx_t dictionary_capacity = 0;
	string dictionary_id;
	idx_t dictionary_version = 0;
};

} // namespace duckdb
