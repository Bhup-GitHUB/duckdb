#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/common/atomic.hpp"

#include "duckdb/common/assert.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/arena_containers/arena_vector.hpp"
#include "duckdb/common/sorting/sort_key_boundary.hpp"
#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/planner/filter/table_filter_function_helpers.hpp"
#include "pdqsort.h"

namespace duckdb {

PhysicalTopN::PhysicalTopN(PhysicalPlan &physical_plan, vector<LogicalType> types, vector<BoundOrderByNode> orders,
                           idx_t limit, idx_t offset, shared_ptr<DynamicFilterData> dynamic_filter_p,
                           idx_t estimated_cardinality, vector<idx_t> projections_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::TOP_N, std::move(types), estimated_cardinality),
      orders(std::move(orders)), limit(limit), offset(offset), dynamic_filter(std::move(dynamic_filter_p)),
      projections(std::move(projections_p)),
      sort_key_type(TupleDataLayout::GetSortKeyType(this->orders, LogicalType::BLOB, true)) {
}

PhysicalTopN::~PhysicalTopN() {
}

//===--------------------------------------------------------------------===//
// Heaps
//===--------------------------------------------------------------------===//
struct TopNScanState {
	TopNScanState() : pos(0), sel(STANDARD_VECTOR_SIZE) {
	}

	idx_t pos;
	vector<sel_t> scan_order;
	SelectionVector sel;
};

class TopNHeap {
public:
	TopNHeap(ClientContext &context, const PhysicalTopN &op);
	virtual ~TopNHeap() = default;

	Allocator &allocator;
	const vector<LogicalType> &payload_types;
	const vector<BoundOrderByNode> &orders;
	const vector<idx_t> &projections;
	vector<OrderModifiers> modifiers;
	idx_t limit;
	idx_t offset;
	idx_t heap_size;
	ExpressionExecutor executor;
	DataChunk sort_chunk;
	DataChunk heap_data;
	DataChunk sort_keys;

	SelectionVector matching_sel;

	unique_ptr<SortKeyBoundaryFilter> boundary_filter;
	SelectionVector boundary_sel;
	//! This heap's slot among the shared boundary's producers, and the size at which it next reports there
	idx_t producer = DConstants::INVALID_INDEX;
	idx_t next_contribution;

public:
	void Sink(DataChunk &input, optional_ptr<SortKeyBoundary> boundary = nullptr);
	virtual void Combine(TopNHeap &other) = 0;
	virtual void Reduce() = 0;
	virtual void Finalize() = 0;
	virtual idx_t Count() const = 0;

	virtual void InitializeScan(TopNScanState &state, bool exclude_offset) = 0;
	void Scan(TopNScanState &state, DataChunk &chunk, idx_t &pos);

	bool CheckBoundaryValues(DataChunk &sort_chunk, DataChunk &payload, const SortKeyBoundary &boundary);

	template <class TARGET>
	TARGET &Cast() {
		DynamicCastCheck<TARGET>(this);
		return reinterpret_cast<TARGET &>(*this);
	}

public:
	idx_t ReduceThreshold() const {
		return MaxValue<idx_t>(STANDARD_VECTOR_SIZE * 5ULL, 2ULL * heap_size);
	}

	idx_t InitialHeapAllocSize() const {
		return MinValue<idx_t>(STANDARD_VECTOR_SIZE * 100ULL, ReduceThreshold()) + STANDARD_VECTOR_SIZE;
	}

protected:
	virtual void AddRows(DataChunk &input) = 0;
	virtual void PublishBoundary(SortKeyBoundary &boundary) = 0;
};

template <SortKeyType SORT_KEY_TYPE>
class TopNKeyHeap final : public TopNHeap {
	using SORT_KEY = SortKey<SORT_KEY_TYPE>;
	static_assert(SORT_KEY::HAS_PAYLOAD, "Top-N keys carry their row as payload");

	struct HeapOrder {
		bool operator()(const SORT_KEY &lhs, const SORT_KEY &rhs) const {
			if constexpr (SORT_KEY::CONSTANT_SIZE) {
				if (lhs.part0 != rhs.part0) {
					return lhs.part0 < rhs.part0;
				}
			}
			return lhs < rhs;
		}
	};

public:
	TopNKeyHeap(ClientContext &context, const PhysicalTopN &op)
	    : TopNHeap(context, op), arena_allocator(allocator), heap(arena_allocator), sort_key_heap(allocator) {
		heap.reserve(InitialHeapAllocSize());
		if (SORT_KEY::CONSTANT_SIZE) {
			row_keys = make_unsafe_uniq_array_uninitialized<SORT_KEY>(STANDARD_VECTOR_SIZE);
			row_targets = make_unsafe_uniq_array_uninitialized<data_ptr_t>(STANDARD_VECTOR_SIZE);
		}
	}

	void Combine(TopNHeap &other_p) override;
	void Reduce() override;
	void Finalize() override;
	void InitializeScan(TopNScanState &state, bool exclude_offset) override;

	idx_t Count() const override {
		return heap.size();
	}

private:
	void AddRows(DataChunk &input) override;
	void PublishBoundary(SortKeyBoundary &boundary) override;

	void AddSmallHeap(DataChunk &input);
	void AddLargeHeap(DataChunk &input);

	static idx_t GetRow(const SORT_KEY &entry) {
		return entry.payload.u.pad;
	}

	static void SetRow(SORT_KEY &entry, idx_t row) {
		entry.payload.u.pad = row;
	}

	void EncodeRows() {
		if constexpr (SORT_KEY::CONSTANT_SIZE) {
			const auto count = sort_chunk.size();
			for (idx_t r = 0; r < count; r++) {
				row_keys[r] = SORT_KEY();
				row_targets[r] = data_ptr_cast(&row_keys[r].part0);
			}
			CreateSortKeyHelpers::CreateSortKey(sort_chunk, modifiers, row_targets.get(), row_offsets);
			for (idx_t r = 0; r < count; r++) {
				row_keys[r].ByteSwap();
			}
		} else {
			sort_keys.Reset();
			CreateSortKeyHelpers::CreateSortKey(sort_chunk, modifiers, sort_keys.data[0]);
		}
	}

	SORT_KEY RowKey(idx_t r) {
		if constexpr (SORT_KEY::CONSTANT_SIZE) {
			return row_keys[r];
		} else {
			SORT_KEY key;
			key.Construct(FlatVector::GetDataMutable<string_t>(sort_keys.data[0])[r]);
			return key;
		}
	}

	static void CopyKey(SORT_KEY &entry, StringHeap &target) {
		if constexpr (!SORT_KEY::CONSTANT_SIZE) {
			const auto size = entry.GetHeapSize();
			if (size == 0) {
				return;
			}
			auto copy = target.GetAllocator().Allocate(size);
			memcpy(copy, entry.GetData(), size);
			entry.SetData(copy);
		}
	}

	string_t KeyBytes(const SORT_KEY &entry) {
		if constexpr (!SORT_KEY::CONSTANT_SIZE) {
			if (entry.GetHeapSize() != 0) {
				return string_t(const_char_ptr_cast(entry.GetData()), UnsafeNumericCast<uint32_t>(entry.GetHeapSize()));
			}
		}
		for (idx_t i = 0; i < SORT_KEY::PARTS; i++) {
			Store<uint64_t>(BSwapIfLE((&entry.part0)[i]), key_bytes + i * sizeof(uint64_t));
		}
		return string_t(const_char_ptr_cast(key_bytes), UnsafeNumericCast<uint32_t>(SORT_KEY::INLINE_LENGTH));
	}

	inline bool EntryShouldBeAdded(const SORT_KEY &sort_key) {
		if (heap_size == 0) {
			// the heap has no capacity (LIMIT 0) - no entry can ever be added
			return false;
		}
		if (heap.size() < heap_size) {
			// heap is not full yet - the entry can be added
			return true;
		}
		if (sort_key < heap.front()) {
			// sort key is smaller than current max value
			return true;
		}
		// heap is full and there is no room for the entry
		return false;
	}

	inline void AddEntryToHeap(const SORT_KEY &entry) {
		D_ASSERT(heap_size > 0);
		if (heap.size() >= heap_size) {
			D_ASSERT(!heap.empty());
			std::pop_heap(heap.begin(), heap.end(), HeapOrder());
			heap.pop_back();
		}
		heap.push_back(entry);
		std::push_heap(heap.begin(), heap.end(), HeapOrder());
	}

private:
	ArenaAllocator arena_allocator;
	unsafe_arena_vector<SORT_KEY> heap;
	StringHeap sort_key_heap;
	unsafe_unique_array<SORT_KEY> row_keys;
	unsafe_unique_array<data_ptr_t> row_targets;
	unsafe_vector<idx_t> row_offsets;
	data_t key_bytes[SORT_KEY::PARTS * sizeof(uint64_t)];
};

//===--------------------------------------------------------------------===//
// TopNHeap
//===--------------------------------------------------------------------===//
TopNHeap::TopNHeap(ClientContext &context, const PhysicalTopN &op)
    : allocator(BufferAllocator::Get(context)), payload_types(op.children[0].get().types), orders(op.orders),
      projections(op.projections), limit(op.limit), offset(op.offset), heap_size(op.limit + op.offset),
      executor(context), matching_sel(STANDARD_VECTOR_SIZE), boundary_sel(STANDARD_VECTOR_SIZE),
      next_contribution(MaxValue<idx_t>(heap_size / 16, 1)) {
	// initialize the executor and the sort_chunk
	vector<LogicalType> sort_types;
	for (auto &order : orders) {
		auto &expr = order.expression;
		sort_types.push_back(expr->GetReturnType());
		executor.AddExpression(*expr);
		modifiers.emplace_back(order.type, order.null_order);
	}
	vector<LogicalType> sort_keys_type {LogicalType::BLOB};
	sort_keys.Initialize(allocator, sort_keys_type);
	heap_data.Initialize(allocator, payload_types, InitialHeapAllocSize());
	sort_chunk.Initialize(allocator, sort_types);
	boundary_filter = make_uniq<SortKeyBoundaryFilter>(allocator, sort_types, modifiers);
}

static unique_ptr<TopNHeap> CreateTopNHeap(ClientContext &context, const PhysicalTopN &op) {
	switch (op.sort_key_type) {
#define DUCKDB_SORT_KEY_CASE(SORT_KEY_TYPE)                                                                            \
	case SortKeyType::SORT_KEY_TYPE:                                                                                   \
		return make_uniq<TopNKeyHeap<SortKeyType::SORT_KEY_TYPE>>(context, op);
		DUCKDB_FOR_EACH_PAYLOAD_SORT_KEY_TYPE(DUCKDB_SORT_KEY_CASE)
#undef DUCKDB_SORT_KEY_CASE
	default:
		throw NotImplementedException("Top-N heap for %s", EnumUtil::ToString(op.sort_key_type));
	}
}

template <SortKeyType SORT_KEY_TYPE>
void TopNKeyHeap<SORT_KEY_TYPE>::AddRows(DataChunk &input) {
	static constexpr idx_t SMALL_HEAP_THRESHOLD = 100;
	EncodeRows();
	if (heap_size <= SMALL_HEAP_THRESHOLD) {
		AddSmallHeap(input);
	} else {
		AddLargeHeap(input);
	}
}

template <SortKeyType SORT_KEY_TYPE>
void TopNKeyHeap<SORT_KEY_TYPE>::PublishBoundary(SortKeyBoundary &boundary) {
	D_ASSERT(!heap.empty());
	if (heap.size() >= heap_size) {
		boundary.Update(KeyBytes(heap.front()));
	} else if (heap_size > 8 * STANDARD_VECTOR_SIZE && heap.size() >= next_contribution) {
		if (producer == DConstants::INVALID_INDEX) {
			producer = boundary.NewProducer();
		}
		static constexpr idx_t QUANTILES = 8;
		vector<SortKeyBoundary::Quantile> quantiles;
		quantiles.reserve(QUANTILES);
		D_ASSERT(heap.size() >= QUANTILES);
		auto rest = heap.begin();
		for (idx_t q = 1; q <= QUANTILES; q++) {
			const auto rows = (heap.size() * q + QUANTILES - 1) / QUANTILES;
			const auto nth = heap.begin() + NumericCast<int64_t>(rows - 1);
			duckdb_pdqsort::nth_element_branchless(rest, nth, heap.end());
			const auto bytes = KeyBytes(*nth);
			quantiles.push_back({string(bytes.GetData(), bytes.GetSize()), rows});
			rest = nth + 1;
		}
		std::make_heap(heap.begin(), heap.end(), HeapOrder());
		boundary.Contribute(producer, std::move(quantiles), heap_size);
		next_contribution = heap.size() * 2;
	}
}

template <SortKeyType SORT_KEY_TYPE>
void TopNKeyHeap<SORT_KEY_TYPE>::AddSmallHeap(DataChunk &input) {
	// insert the sort keys into the priority queue
	constexpr idx_t BASE_INDEX = NumericLimits<uint32_t>::Maximum();

	bool any_added = false;
	for (idx_t r = 0; r < input.size(); r++) {
		auto entry = RowKey(r);
		if (!EntryShouldBeAdded(entry)) {
			continue;
		}
		// replace the previous top entry with the new entry
		SetRow(entry, BASE_INDEX + r);
		AddEntryToHeap(entry);
		any_added = true;
	}
	if (!any_added) {
		// early-out: no matches
		return;
	}

	// for all matching entries we need to copy over the corresponding payload values
	idx_t match_count = 0;
	for (auto &entry : heap) {
		if (GetRow(entry) < BASE_INDEX) {
			continue;
		}
		// this entry was added in this chunk
		CopyKey(entry, sort_key_heap);
		// to finalize the addition of this entry we need to move over the payload data
		matching_sel.set_index(match_count, GetRow(entry) - BASE_INDEX);
		SetRow(entry, heap_data.size() + match_count);
		match_count++;
	}

	// copy over the input rows to the payload chunk
	heap_data.Append(input, matching_sel, match_count, VectorAppendMode::ALLOW_RESIZE);
}

template <SortKeyType SORT_KEY_TYPE>
void TopNKeyHeap<SORT_KEY_TYPE>::AddLargeHeap(DataChunk &input) {
	idx_t base_index = heap_data.size();
	idx_t match_count = 0;
	for (idx_t r = 0; r < input.size(); r++) {
		auto entry = RowKey(r);
		if (!EntryShouldBeAdded(entry)) {
			continue;
		}
		// replace the previous top entry with the new entry
		CopyKey(entry, sort_key_heap);
		SetRow(entry, base_index + match_count);
		AddEntryToHeap(entry);
		matching_sel.set_index(match_count++, r);
	}
	if (match_count == 0) {
		// early-out: no matches
		return;
	}

	// copy over the input rows to the payload chunk
	heap_data.Append(input, matching_sel, match_count, VectorAppendMode::ALLOW_RESIZE);
}

bool TopNHeap::CheckBoundaryValues(DataChunk &sort_chunk, DataChunk &payload, const SortKeyBoundary &boundary) {
	if (!boundary_filter->Refresh(boundary)) {
		return true;
	}
	idx_t before;
	const auto count = boundary_filter->Select(sort_chunk, false, boundary_sel, before);
	if (count == 0) {
		return false;
	}
	if (count < sort_chunk.size()) {
		sort_chunk.Slice(boundary_sel, count);
		payload.Slice(boundary_sel, count);
	}
	return true;
}

void TopNHeap::Sink(DataChunk &input, optional_ptr<SortKeyBoundary> global_boundary) {
	if (heap_size == 0) {
		// LIMIT 0 without OFFSET - the heap can never hold any entry, so there is nothing to do
		return;
	}

	// compute the ordering values for the new chunk
	sort_chunk.Reset();
	executor.Execute(input, sort_chunk);

	if (global_boundary) {
		// if we have a global boundary value check which rows pass before doing anything
		if (!CheckBoundaryValues(sort_chunk, input, *global_boundary)) {
			// nothing in this chunk can be in the final result
			return;
		}
	}

	AddRows(input);

	// if we modified the heap we might be able to update the global boundary
	if (global_boundary) {
		PublishBoundary(*global_boundary);
	}
}

template <SortKeyType SORT_KEY_TYPE>
void TopNKeyHeap<SORT_KEY_TYPE>::Combine(TopNHeap &other_p) {
	auto &other = other_p.Cast<TopNKeyHeap>();
	// "other" is sorted at this point
	idx_t match_count = 0;
	// merge the heap of other into this
	for (idx_t i = 0; i < other.heap.size(); i++) {
		// heap is full - check the latest entry
		auto new_entry = other.heap[i];
		if (!EntryShouldBeAdded(new_entry)) {
			break; // We can break out of the loop here since we sorted the heap
		}
		// add this entry
		CopyKey(new_entry, sort_key_heap);
		matching_sel.set_index(match_count, GetRow(new_entry));
		SetRow(new_entry, heap_data.size() + match_count);
		AddEntryToHeap(new_entry);

		match_count++;
		if (match_count >= STANDARD_VECTOR_SIZE) {
			// flush
			heap_data.Append(other.heap_data, matching_sel, match_count, VectorAppendMode::ALLOW_RESIZE);
			match_count = 0;
		}
	}
	if (match_count > 0) {
		// flush
		heap_data.Append(other.heap_data, matching_sel, match_count, VectorAppendMode::ALLOW_RESIZE);
		match_count = 0;
	}
	Reduce();
}

template <SortKeyType SORT_KEY_TYPE>
void TopNKeyHeap<SORT_KEY_TYPE>::Finalize() {
	duckdb_pdqsort::pdqsort_branchless(heap.begin(), heap.end());
}

template <SortKeyType SORT_KEY_TYPE>
void TopNKeyHeap<SORT_KEY_TYPE>::Reduce() {
	if (heap_data.size() < ReduceThreshold()) {
		// only reduce when we pass the reduce threshold
		return;
	}
	// we have too many values in the heap - reduce them
	StringHeap new_sort_heap(allocator);
	DataChunk new_heap_data;
	new_heap_data.Initialize(allocator, payload_types, heap.size());

	SelectionVector new_payload_sel(heap.size());
	for (idx_t i = 0; i < heap.size(); i++) {
		auto &entry = heap[i];
		// move the sort key to the new sort heap
		CopyKey(entry, new_sort_heap);
		// move this heap entry to position X in the payload chunk
		new_payload_sel.set_index(i, GetRow(entry));
		SetRow(entry, i);
	}

	// copy over the data from the current payload chunk to the new payload chunk
	new_heap_data.Slice(heap_data, new_payload_sel, heap.size());
	new_heap_data.Flatten();

	sort_key_heap.Destroy();
	sort_key_heap.Move(new_sort_heap);
	heap_data.Reference(new_heap_data);
}

template <SortKeyType SORT_KEY_TYPE>
void TopNKeyHeap<SORT_KEY_TYPE>::InitializeScan(TopNScanState &state, bool exclude_offset) {
	// we sorted the heap in Finalize(), so we can sequentially here
	// we do this because sorting the heap is much more efficient than popping one by one
	state.scan_order.resize(heap.size());
	for (idx_t i = 0; i < heap.size(); i++) {
		state.scan_order[i] = UnsafeNumericCast<sel_t>(GetRow(heap[i]));
	}

	state.pos = exclude_offset ? offset : 0;
}

void TopNHeap::Scan(TopNScanState &state, DataChunk &chunk, idx_t &pos) {
	if (pos >= state.scan_order.size()) {
		return;
	}
	SelectionVector sel(state.scan_order.data() + pos, state.scan_order.size() - pos);
	idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.scan_order.size() - pos);
	pos += STANDARD_VECTOR_SIZE;

	chunk.Reset();
	D_ASSERT(projections.size() == chunk.ColumnCount());
	for (idx_t i = 0; i < projections.size(); ++i) {
		D_ASSERT(projections[i] < heap_data.ColumnCount());
		chunk.data[i].Slice(heap_data.data[projections[i]], sel, count);
	}
	chunk.SetCardinality(count);
}

class TopNGlobalSinkState : public GlobalSinkState {
public:
	TopNGlobalSinkState(ClientContext &context, const PhysicalTopN &op)
	    : heap(CreateTopNHeap(context, op)),
	      boundary_value(op.orders[0].expression->GetReturnType(),
	                     OrderModifiers(op.orders[0].type, op.orders[0].null_order), op.dynamic_filter) {
	}

	mutex lock;
	unique_ptr<TopNHeap> heap;
	SortKeyBoundary boundary_value;
};

class TopNLocalSinkState : public LocalSinkState {
public:
	TopNLocalSinkState(ExecutionContext &context, const PhysicalTopN &op) : heap(CreateTopNHeap(context.client, op)) {
	}

	unique_ptr<TopNHeap> heap;
};

unique_ptr<LocalSinkState> PhysicalTopN::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<TopNLocalSinkState>(context, *this);
}

unique_ptr<GlobalSinkState> PhysicalTopN::GetGlobalSinkState(ClientContext &context) const {
	if (dynamic_filter) {
		dynamic_filter->Reset();
	}
	return make_uniq<TopNGlobalSinkState>(context, *this);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
SinkResultType PhysicalTopN::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	// append to the local sink state
	auto &gstate = input.global_state.Cast<TopNGlobalSinkState>();
	auto &sink = input.local_state.Cast<TopNLocalSinkState>();
	sink.heap->Sink(chunk, &gstate.boundary_value);
	sink.heap->Reduce();
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
SinkCombineResultType PhysicalTopN::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<TopNGlobalSinkState>();
	auto &lstate = input.local_state.Cast<TopNLocalSinkState>();

	// scan the local top N and append it to the global heap
	lstate.heap->Finalize();
	lock_guard<mutex> guard(gstate.lock);
	gstate.heap->Combine(*lstate.heap);

	return SinkCombineResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalTopN::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                        OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<TopNGlobalSinkState>();
	// global finalize: compute the final top N
	gstate.heap->Finalize();
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class TopNLocalSourceState : public LocalSourceState {
public:
	idx_t pos = DConstants::INVALID_INDEX;
	idx_t end = DConstants::INVALID_INDEX;
	idx_t batch_index = DConstants::INVALID_INDEX;
};

class TopNGlobalSourceState : public GlobalSourceState {
public:
	explicit TopNGlobalSourceState(TopNGlobalSinkState &sink_p) : sink(sink_p), batch_index(0) {
		sink.heap->InitializeScan(state, true);
		total_rows = state.scan_order.size() - MinValue<idx_t>(sink.heap->offset, state.scan_order.size());
	}

	idx_t MaxThreads() override {
		return MaxValue<idx_t>(sink.heap->Count() / TUPLES_PER_BATCH, 1);
	}

public:
	static constexpr idx_t CHUNKS_PER_BATCH = 60;
	static constexpr idx_t TUPLES_PER_BATCH = CHUNKS_PER_BATCH * STANDARD_VECTOR_SIZE;

	TopNGlobalSinkState &sink;
	TopNScanState state;
	idx_t batch_index;
	idx_t total_rows;
	atomic<idx_t> rows_scanned {0};
};

unique_ptr<GlobalSourceState> PhysicalTopN::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<TopNGlobalSourceState>(this->sink_state->Cast<TopNGlobalSinkState>());
}

ProgressData PhysicalTopN::GetProgress(ClientContext &context, GlobalSourceState &gstate) const {
	auto &state = gstate.Cast<TopNGlobalSourceState>();
	if (state.total_rows == 0) {
		return ProgressData {1.0, 1.0, false};
	}
	return ProgressData {double(state.rows_scanned.load()), double(state.total_rows), false};
}

unique_ptr<LocalSourceState> PhysicalTopN::GetLocalSourceState(ExecutionContext &context,
                                                               GlobalSourceState &gstate) const {
	return make_uniq<TopNLocalSourceState>();
}

SourceResultType PhysicalTopN::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                               OperatorSourceInput &input) const {
	if (limit == 0) {
		return SourceResultType::FINISHED;
	}
	auto &sink = sink_state->Cast<TopNGlobalSinkState>();
	auto &gstate = input.global_state.Cast<TopNGlobalSourceState>();
	auto &lstate = input.local_state.Cast<TopNLocalSourceState>();

	if (lstate.pos == lstate.end) {
		// Obtain new scan indices from the global state
		annotated_lock_guard<annotated_mutex> guard(gstate.lock);
		lstate.pos = gstate.state.pos;
		gstate.state.pos += TopNGlobalSourceState::TUPLES_PER_BATCH;
		lstate.end = gstate.state.pos;
		lstate.batch_index = gstate.batch_index++;
	}

	sink.heap->Scan(gstate.state, chunk, lstate.pos);
	gstate.rows_scanned.fetch_add(chunk.size(), std::memory_order_relaxed);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

OperatorPartitionData PhysicalTopN::GetPartitionData(ExecutionContext &context, DataChunk &chunk,
                                                     GlobalSourceState &gstate_p, LocalSourceState &lstate_p,
                                                     const OperatorPartitionInfo &partition_info) const {
	if (partition_info.RequiresPartitionColumns()) {
		throw InternalException("PhysicalOrder::GetPartitionData: partition columns not supported");
	}
	auto &lstate = lstate_p.Cast<TopNLocalSourceState>();
	return OperatorPartitionData(lstate.batch_index);
}

InsertionOrderPreservingMap<string> PhysicalTopN::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Top"] = to_string(limit);
	if (offset > 0) {
		result["Offset"] = to_string(offset);
	}

	string orders_info;
	for (idx_t i = 0; i < orders.size(); i++) {
		if (i > 0) {
			orders_info += "\n";
		}
		orders_info += orders[i].expression->ToString() + " ";
		orders_info += orders[i].type == OrderType::DESCENDING ? "DESC" : "ASC";
	}
	result["Order By"] = orders_info;
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
