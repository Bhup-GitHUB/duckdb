//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/table/row_group_reorderer.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/sorting/sort_key.hpp"
#include "duckdb/function/partition_stats.hpp"
#include "duckdb/storage/table/row_group.hpp"
#include "duckdb/storage/table/row_group_order_options.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/storage/table/segment_tree.hpp"

namespace duckdb {

using RowGroupOrderKey = SortKey<SortKeyType::NO_PAYLOAD_FIXED_24>;

struct RowGroupOrderKeys {
	static RowGroupOrderKey Get(const BaseStatistics &stats, OrderByStatistics order_by, OrderType order_type,
	                            OrderByNullType null_order);
	static RowGroupOrderKey Get(const Value &value, OrderType order_type, OrderByNullType null_order);
	static RowGroupOrderKey Null(OrderByNullType null_order);
	static bool IsNull(const RowGroupOrderKey &key, OrderByNullType null_order);
};

struct OffsetPruningResult {
	idx_t offset_remainder;
	idx_t pruned_row_group_count;
	idx_t leading_null_group_offset;
};

class RowGroupReorderer {
public:
	RowGroupReorderer(const RowGroupOrderOptions &options_p, TransactionData transaction_p);
	optional_ptr<SegmentNode<RowGroup>> GetRootSegment(RowGroupSegmentTree &row_groups);
	optional_ptr<SegmentNode<RowGroup>> GetNextRowGroup(SegmentNode<RowGroup> &row_group);
	//! The number of rows in the row groups that will be scanned (after pruning)
	idx_t ScanRowCount() const;

	static OffsetPruningResult GetOffsetAfterPruning(OrderByStatistics order_by, OrderType order_type,
	                                                 OrderByNullType null_order, const StorageIndex &column_idx,
	                                                 idx_t row_offset, bool single_order_key,
	                                                 vector<PartitionStatistics> &stats);
	static vector<idx_t> GetOffsetPrunedPartitions(const RowGroupOrderOptions &options,
	                                               const vector<PartitionStatistics> &stats);

private:
	const RowGroupOrderOptions options;
	const TransactionData transaction;

	idx_t offset;
	bool initialized;
	vector<reference<SegmentNode<RowGroup>>> ordered_row_groups;
};

} // namespace duckdb
