#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/storage/arena_allocator.hpp"
#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/bswap.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/string_vector.hpp"
#include "duckdb/common/enums/order_type.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include <new>

namespace duckdb {

//! The min/max/arg_min/arg_max "n" aggregates require the requested n to be strictly smaller than this value
static constexpr int64_t MIN_MAX_N_MAX_VALUE = 1000000;

// For basic types
template <class T>
struct HeapEntry {
	T value;

	void Assign(ArenaAllocator &allocator, const T &val) {
		value = val;
	}
};

// For strings that require arena allocation
template <>
struct HeapEntry<string_t> {
	string_t value;
	uint32_t capacity;
	data_ptr_t allocated_data;

	HeapEntry() : value(), capacity(0), allocated_data(nullptr) {
	}

	// Not copyable
	HeapEntry(const HeapEntry &other) = delete;
	HeapEntry &operator=(const HeapEntry &other) = delete;

	// But movable
	HeapEntry(HeapEntry &&other) noexcept {
		if (other.value.IsInlined()) {
			value = other.value;
			capacity = 0;
			allocated_data = nullptr;
		} else {
			capacity = other.capacity;
			allocated_data = other.allocated_data;
			value = string_t(const_char_ptr_cast(allocated_data), UnsafeNumericCast<uint32_t>(other.value.GetSize()));
			other.allocated_data = nullptr;
		}
	}

	HeapEntry &operator=(HeapEntry &&other) noexcept {
		if (other.value.IsInlined()) {
			value = other.value;
		} else {
			capacity = other.capacity;
			allocated_data = other.allocated_data;
			value = string_t(const_char_ptr_cast(allocated_data), UnsafeNumericCast<uint32_t>(other.value.GetSize()));
			other.allocated_data = nullptr;
		}
		return *this;
	}

	void Assign(ArenaAllocator &allocator, const string_t &new_val) {
		value = Store(allocator, new_val, capacity, allocated_data);
	}

	static string_t Store(ArenaAllocator &allocator, const string_t &new_val, uint32_t &capacity,
	                      data_ptr_t &allocated_data) {
		if (new_val.IsInlined()) {
			return new_val;
		}

		// Short path for first assignment
		if (allocated_data == nullptr) {
			auto new_size = UnsafeNumericCast<uint32_t>(new_val.GetSize());
			auto new_capacity = NextPowerOfTwo(new_size);
			if (new_capacity > string_t::MAX_STRING_SIZE) {
				throw ObjectSizeException("Resulting string/blob too large!");
			}
			capacity = UnsafeNumericCast<uint32_t>(new_capacity);
			allocated_data = allocator.Allocate(capacity);
			memcpy(allocated_data, new_val.GetData(), new_size);
			return string_t(const_char_ptr_cast(allocated_data), new_size);
		}

		// double allocation until value fits
		if (capacity < new_val.GetSize()) {
			auto old_size = capacity;
			auto new_size = UnsafeNumericCast<uint32_t>(new_val.GetSize());
			auto new_capacity = NextPowerOfTwo(new_size);
			if (new_capacity > string_t::MAX_STRING_SIZE) {
				throw ObjectSizeException("Resulting string/blob too large!");
			}
			do {
				capacity *= 2;
			} while (capacity < new_val.GetSize());
			allocated_data = allocator.Reallocate(allocated_data, old_size, capacity);
		}
		auto new_size = UnsafeNumericCast<uint32_t>(new_val.GetSize());
		memcpy(allocated_data, new_val.GetData(), new_size);
		return string_t(const_char_ptr_cast(allocated_data), new_size);
	}
};

struct PrefixedString {
	uint64_t high;
	uint64_t low;
	string_t value;

	static PrefixedString Make(const string_t &value) {
		uint64_t words[2] = {0, 0};
		memcpy(words, value.GetData(), MinValue<idx_t>(value.GetSize(), sizeof(words)));
		return PrefixedString {BSwapIfLE(words[0]), BSwapIfLE(words[1]), value};
	}

	friend bool operator>(const PrefixedString &left, const PrefixedString &right) {
		if (left.high != right.high) {
			return left.high > right.high;
		}
		if (left.low != right.low) {
			return left.low > right.low;
		}
		const auto left_size = left.value.GetSize();
		const auto right_size = right.value.GetSize();
		const auto common = MinValue(left_size, right_size);
		if (common > sizeof(uint64_t) * 2) {
			const auto skip = sizeof(uint64_t) * 2;
			const auto result = memcmp(left.value.GetData() + skip, right.value.GetData() + skip, common - skip);
			if (result != 0) {
				return result > 0;
			}
		}
		return left_size > right_size;
	}

	friend bool operator==(const PrefixedString &left, const PrefixedString &right) {
		return left.high == right.high && left.low == right.low && Equals::Operation(left.value, right.value);
	}
};

template <>
struct HeapEntry<PrefixedString> {
	PrefixedString value;
	uint32_t capacity;
	data_ptr_t allocated_data;

	HeapEntry() : value(), capacity(0), allocated_data(nullptr) {
	}

	HeapEntry(const HeapEntry &other) = delete;
	HeapEntry &operator=(const HeapEntry &other) = delete;

	HeapEntry(HeapEntry &&other) noexcept : value(other.value), capacity(other.capacity) {
		allocated_data = other.allocated_data;
		other.allocated_data = nullptr;
	}

	HeapEntry &operator=(HeapEntry &&other) noexcept {
		value = other.value;
		capacity = other.capacity;
		allocated_data = other.allocated_data;
		other.allocated_data = nullptr;
		return *this;
	}

	void Assign(ArenaAllocator &allocator, const PrefixedString &new_val) {
		value = PrefixedString {new_val.high, new_val.low,
		                        HeapEntry<string_t>::Store(allocator, new_val.value, capacity, allocated_data)};
	}
};

template <class T, class T_COMPARATOR>
class UnaryAggregateHeap {
public:
	UnaryAggregateHeap() = default;

	UnaryAggregateHeap(ArenaAllocator &allocator, idx_t capacity_p) {
		Initialize(allocator, capacity_p);
	}

	void Initialize(ArenaAllocator &allocator, const idx_t capacity_p) {
		capacity = capacity_p;
		auto ptr = allocator.AllocateAligned(capacity * sizeof(HeapEntry<T>));
		memset(ptr, 0, capacity * sizeof(HeapEntry<T>));
		heap = reinterpret_cast<HeapEntry<T> *>(ptr);
		size = 0;
	}

	bool IsEmpty() const {
		return size == 0;
	}
	idx_t Size() const {
		return size;
	}
	idx_t Capacity() const {
		return capacity;
	}

	void Insert(ArenaAllocator &allocator, const T &value) {
		D_ASSERT(capacity != 0); // must be initialized

		// If the heap is not full, insert the value into a new slot
		if (size < capacity) {
			heap[size++].Assign(allocator, value);
			std::push_heap(heap, heap + size, Compare);
		}
		// If the heap is full, check if the value is greater than the smallest value in the heap
		// If it is, assign the new value to the slot and re-heapify
		else if (T_COMPARATOR::Operation(value, heap[0].value)) {
			std::pop_heap(heap, heap + size, Compare);
			heap[size - 1].Assign(allocator, value);
			std::push_heap(heap, heap + size, Compare);
		}
		D_ASSERT(std::is_heap(heap, heap + size, Compare));
	}

	void Insert(ArenaAllocator &allocator, const UnaryAggregateHeap &other) {
		for (idx_t slot = 0; slot < other.Size(); slot++) {
			Insert(allocator, other.heap[slot].value);
		}
	}

	HeapEntry<T> *SortAndGetHeap() {
		std::sort_heap(heap, heap + size, Compare);
		return heap;
	}

	static const T &GetValue(const HeapEntry<T> &slot) {
		return slot.value;
	}

private:
	static bool Compare(const HeapEntry<T> &left, const HeapEntry<T> &right) {
		return T_COMPARATOR::Operation(left.value, right.value);
	}

	idx_t capacity;
	HeapEntry<T> *heap;
	idx_t size;
};

struct List {
	uint32_t capacity;
	uint32_t size;
	List *next;
};

template <class K, class V, class K_COMPARATOR>
class BinaryAggregateHeap {
	using STORAGE_TYPE = pair<HeapEntry<K>, HeapEntry<V>>;

public:
	BinaryAggregateHeap() = default;

	BinaryAggregateHeap(ArenaAllocator &allocator, idx_t capacity_p) {
		Initialize(allocator, capacity_p);
	}

	void Initialize(ArenaAllocator &allocator, const idx_t capacity_p) {
		capacity = capacity_p;
		allocated_capacity = 0;
		heap = nullptr;
		size = 0;
	}

	bool IsEmpty() const {
		return size == 0;
	}
	idx_t Size() const {
		return size;
	}
	idx_t Capacity() const {
		return capacity;
	}

	void Insert(ArenaAllocator &allocator, const K &key, const V &value) {
		D_ASSERT(capacity != 0); // must be initialized

		// If the heap is not full, insert the value into a new slot
		if (size < capacity) {
			if (size == allocated_capacity) {
				Grow(allocator);
			}
			heap[size].first.Assign(allocator, key);
			heap[size].second.Assign(allocator, value);
			size++;
			std::push_heap(heap, heap + size, Compare);
		}
		// If the heap is full, check if the value is greater than the smallest value in the heap
		// If it is, assign the new value to the slot and re-heapify
		else if (K_COMPARATOR::Operation(key, heap[0].first.value)) {
			std::pop_heap(heap, heap + size, Compare);
			heap[size - 1].first.Assign(allocator, key);
			heap[size - 1].second.Assign(allocator, value);
			std::push_heap(heap, heap + size, Compare);
		}
		D_ASSERT(std::is_heap(heap, heap + size, Compare));
	}

	void Insert(ArenaAllocator &allocator, const BinaryAggregateHeap &other) {
		for (idx_t slot = 0; slot < other.Size(); slot++) {
			Insert(allocator, other.heap[slot].first.value, other.heap[slot].second.value);
		}
	}

	STORAGE_TYPE *SortAndGetHeap() {
		std::sort_heap(heap, heap + size, Compare);
		return heap;
	}

	static const V &GetValue(const STORAGE_TYPE &slot) {
		return slot.second.value;
	}

private:
	void Grow(ArenaAllocator &allocator) {
		D_ASSERT(allocated_capacity < capacity);
		const auto old_allocated_capacity = allocated_capacity;
		if (allocated_capacity == 0) {
			allocated_capacity = 1;
		} else if (allocated_capacity > capacity / 2) {
			allocated_capacity = capacity;
		} else {
			if (!TryMultiplyOperator::Operation(allocated_capacity, static_cast<idx_t>(2), allocated_capacity)) {
				throw OutOfRangeException("Overflow of 'allocated_capacity' in BinaryAggregateHeap::Grow");
			}
		}

		const auto old_size = old_allocated_capacity * sizeof(STORAGE_TYPE);
		idx_t new_size;
		if (!TryMultiplyOperator::Operation(allocated_capacity, static_cast<idx_t>(sizeof(STORAGE_TYPE)), new_size)) {
			throw OutOfRangeException("Overflow of 'new_size' in BinaryAggregateHeap::Grow");
		}
		auto ptr = heap ? allocator.ReallocateAligned(reinterpret_cast<data_ptr_t>(heap), old_size, new_size)
		                : allocator.AllocateAligned(new_size);
		memset(ptr + old_size, 0, new_size - old_size);
		heap = reinterpret_cast<STORAGE_TYPE *>(ptr);
	}

	static bool Compare(const STORAGE_TYPE &left, const STORAGE_TYPE &right) {
		return K_COMPARATOR::Operation(left.first.value, right.first.value);
	}

	idx_t capacity = 0;
	idx_t allocated_capacity = 0;
	STORAGE_TYPE *heap = nullptr;
	idx_t size = 0;
};

enum class ArgMinMaxNullHandling { IGNORE_ANY_NULL, HANDLE_ARG_NULL, HANDLE_ANY_NULL };

struct ArgMinMaxFunctionData : FunctionData {
	explicit ArgMinMaxFunctionData(ArgMinMaxNullHandling null_handling_p = ArgMinMaxNullHandling::IGNORE_ANY_NULL,
	                               bool nulls_last_p = true)
	    : null_handling(null_handling_p), nulls_last(nulls_last_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<ArgMinMaxFunctionData>();
		copy->null_handling = null_handling;
		copy->nulls_last = nulls_last;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ArgMinMaxFunctionData>();
		return other.null_handling == null_handling && other.nulls_last == nulls_last;
	}

	ArgMinMaxNullHandling null_handling;
	bool nulls_last;
};

//------------------------------------------------------------------------------
// Specializations for fixed size types, strings, and anything else (using sortkey)
//------------------------------------------------------------------------------
template <class T>
struct MinMaxFixedValue {
	using TYPE = T;
	using EXTRA_STATE = bool;

	static TYPE Create(const UnifiedVectorFormat &format, const idx_t idx) {
		return UnifiedVectorFormat::GetData<T>(format)[idx];
	}

	static void Assign(Vector &vector, const idx_t idx, const TYPE &value, const bool nulls_last) {
		FlatVector::GetDataMutable<T>(vector)[idx] = value;
	}

	// Nothing to do here
	static EXTRA_STATE CreateExtraState() {
		return false;
	}

	static void PrepareData(const Vector &input, EXTRA_STATE &, UnifiedVectorFormat &format, const bool nulls_last) {
		input.ToUnifiedFormat(format);
	}
};

struct MinMaxStringValue {
	using TYPE = PrefixedString;
	using EXTRA_STATE = bool;

	static TYPE Create(const UnifiedVectorFormat &format, const idx_t idx) {
		return PrefixedString::Make(UnifiedVectorFormat::GetData<string_t>(format)[idx]);
	}

	static void Assign(Vector &vector, const idx_t idx, const TYPE &value, const bool nulls_last) {
		FlatVector::GetDataMutable<string_t>(vector)[idx] = StringVector::AddStringOrBlob(vector, value.value);
	}

	// Nothing to do here
	static EXTRA_STATE CreateExtraState() {
		return false;
	}

	static void PrepareData(const Vector &input, EXTRA_STATE &, UnifiedVectorFormat &format, const bool nulls_last) {
		input.ToUnifiedFormat(format);
	}
};

// Use sort key to serialize/deserialize values
struct MinMaxFallbackValue {
	using TYPE = PrefixedString;
	using EXTRA_STATE = Vector;

	static TYPE Create(const UnifiedVectorFormat &format, const idx_t idx) {
		return PrefixedString::Make(UnifiedVectorFormat::GetData<string_t>(format)[idx]);
	}

	static void Assign(Vector &vector, const idx_t idx, const TYPE &value, const bool nulls_last) {
		auto order_by_null_type = nulls_last ? OrderByNullType::NULLS_LAST : OrderByNullType::NULLS_FIRST;
		OrderModifiers modifiers(OrderType::ASCENDING, order_by_null_type);
		CreateSortKeyHelpers::DecodeSortKey(value.value, vector, idx, modifiers);
	}

	static EXTRA_STATE CreateExtraState() {
		return Vector(LogicalTypeId::BLOB);
	}

	static void PrepareData(const Vector &input, EXTRA_STATE &extra_state, UnifiedVectorFormat &format,
	                        const bool nulls_last) {
		auto order_by_null_type = nulls_last ? OrderByNullType::NULLS_LAST : OrderByNullType::NULLS_FIRST;
		const OrderModifiers modifiers(OrderType::ASCENDING, order_by_null_type);
		CreateSortKeyHelpers::CreateSortKeyWithValidity(input, extra_state, modifiers);
		input.Flatten();
		extra_state.ToUnifiedFormat(format);
	}
};

template <class T, bool NULLS_LAST>
struct ValueOrNull {
	T value;
	bool is_valid;

	bool operator==(const ValueOrNull &other) const {
		return is_valid == other.is_valid && value == other.value;
	}

	bool operator>(const ValueOrNull &other) const {
		if (is_valid && other.is_valid) {
			return GreaterThan::Operation(value, other.value);
		}
		if (!is_valid && !other.is_valid) {
			return false;
		}

		return is_valid ^ NULLS_LAST;
	}
};

template <class T, bool NULLS_LAST>
struct MinMaxFixedValueOrNull {
	using TYPE = ValueOrNull<T, NULLS_LAST>;
	using EXTRA_STATE = bool;

	static TYPE Create(const UnifiedVectorFormat &format, const idx_t idx) {
		return TYPE {UnifiedVectorFormat::GetData<T>(format)[idx], format.validity.RowIsValid(idx)};
	}

	static void Assign(Vector &vector, const idx_t idx, const TYPE &value, const bool nulls_last) {
		FlatVector::ValidityMutable(vector).Set(idx, value.is_valid);
		FlatVector::GetDataMutable<T>(vector)[idx] = value.value;
	}

	static EXTRA_STATE CreateExtraState() {
		return false;
	}

	static void PrepareData(const Vector &input, EXTRA_STATE &extra_state, UnifiedVectorFormat &format,
	                        const bool nulls_last) {
		input.ToUnifiedFormat(format);
	}
};

struct MinMaxWidenedValue {
	using TYPE = int64_t;
	using EXTRA_STATE = Vector;

	static bool CanWiden(PhysicalType type) {
		switch (type) {
		case PhysicalType::BOOL:
		case PhysicalType::INT8:
		case PhysicalType::INT16:
		case PhysicalType::UINT8:
		case PhysicalType::UINT16:
		case PhysicalType::UINT32:
		case PhysicalType::UINT64:
			return true;
		default:
			return false;
		}
	}

	static TYPE Create(const UnifiedVectorFormat &format, const idx_t idx) {
		return UnifiedVectorFormat::GetData<int64_t>(format)[idx];
	}

	static EXTRA_STATE CreateExtraState() {
		return Vector(LogicalType::BIGINT);
	}

	static void PrepareData(const Vector &input, EXTRA_STATE &extra_state, UnifiedVectorFormat &format,
	                        const bool nulls_last) {
		switch (input.GetType().InternalType()) {
		case PhysicalType::BOOL:
			Widen<bool>(input, extra_state);
			break;
		case PhysicalType::INT8:
			Widen<int8_t>(input, extra_state);
			break;
		case PhysicalType::INT16:
			Widen<int16_t>(input, extra_state);
			break;
		case PhysicalType::UINT8:
			Widen<uint8_t>(input, extra_state);
			break;
		case PhysicalType::UINT16:
			Widen<uint16_t>(input, extra_state);
			break;
		case PhysicalType::UINT32:
			Widen<uint32_t>(input, extra_state);
			break;
		case PhysicalType::UINT64:
			Widen<uint64_t>(input, extra_state);
			break;
		default:
			throw InternalException("MinMaxWidenedValue: unsupported type %s", input.GetType().ToString());
		}
		extra_state.ToUnifiedFormat(format);
	}

private:
	template <class T>
	static void Widen(const Vector &input, Vector &result) {
		auto writer = FlatVector::Writer<int64_t>(result, input.size());
		for (auto entry : input.Values<T>()) {
			if (!entry.IsValid()) {
				writer.WriteNull();
			} else if (std::is_same_v<T, uint64_t>) {
				writer.WriteValue(static_cast<int64_t>(static_cast<uint64_t>(entry.GetValue()) ^ (uint64_t(1) << 63)));
			} else {
				writer.WriteValue(static_cast<int64_t>(entry.GetValue()));
			}
		}
	}
};

template <bool NULLS_LAST>
struct MinMaxWidenedValueOrNull : MinMaxFixedValueOrNull<int64_t, NULLS_LAST> {
	using EXTRA_STATE = MinMaxWidenedValue::EXTRA_STATE;

	static EXTRA_STATE CreateExtraState() {
		return MinMaxWidenedValue::CreateExtraState();
	}

	static void PrepareData(const Vector &input, EXTRA_STATE &extra_state, UnifiedVectorFormat &format,
	                        const bool nulls_last) {
		MinMaxWidenedValue::PrepareData(input, extra_state, format, nulls_last);
	}
};

//------------------------------------------------------------------------------
// MinMaxN Operation (common for both ArgMinMaxN and MinMaxN)
//------------------------------------------------------------------------------
struct MinMaxNOperation {
	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input) {
		if (!source.is_initialized) {
			// source is empty, nothing to do
			return;
		}

		if (!target.is_initialized) {
			target.Initialize(aggr_input.allocator, source.heap.Capacity());
		} else if (source.heap.Capacity() != target.heap.Capacity()) {
			throw InvalidInputException("Mismatched n values in min/max/arg_min/arg_max");
		}

		// Merge the heaps
		target.heap.Insert(aggr_input.allocator, source.heap);
	}

	template <class STATE>
	static void Finalize(Vector &state_vector, AggregateFinalizeInputData &input_data, Vector &result, idx_t count,
	                     idx_t offset) {
		// We only expect bind data from arg_max, otherwise nulls last is the default
		const bool nulls_last =
		    input_data.bind_data ? input_data.bind_data->Cast<ArgMinMaxFunctionData>().nulls_last : true;

		UnifiedVectorFormat state_format;
		state_vector.ToUnifiedFormat(state_format);

		const auto states = UnifiedVectorFormat::GetData<STATE *>(state_format);

		const auto old_len = ListVector::GetListSize(result);

		// Count the number of new entries
		idx_t new_entries = 0;
		for (idx_t i = 0; i < count; i++) {
			const auto state_idx = state_format.sel->get_index(i);
			auto &state = *states[state_idx];
			new_entries += state.heap.Size();
		}

		// Resize the list vector to fit the new entries
		ListVector::Reserve(result, old_len + new_entries);

		auto result_data = FlatVector::Writer<list_entry_t>(result, count, offset);
		auto &child_data = ListVector::GetChildMutable(result);

		idx_t current_offset = old_len;
		for (idx_t i = 0; i < count; i++) {
			const auto state_idx = state_format.sel->get_index(i);
			auto &state = *states[state_idx];

			if (!state.is_initialized || state.heap.IsEmpty()) {
				result_data.WriteNull();
				continue;
			}

			// Add the entries to the list vector
			list_entry_t list_entry;
			list_entry.offset = current_offset;
			list_entry.length = state.heap.Size();
			result_data.WriteValue(list_entry);

			// Turn the heap into a sorted list, invalidating the heap property
			auto heap = state.heap.SortAndGetHeap();

			for (idx_t slot = 0; slot < state.heap.Size(); slot++) {
				STATE::VAL_TYPE::Assign(child_data, current_offset++, state.heap.GetValue(heap[slot]), nulls_last);
			}
		}

		D_ASSERT(current_offset == old_len + new_entries);
		ListVector::SetListSize(result, current_offset);
		result.Verify();
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &aggr_input_data) {
		state.~STATE();
	}

	static bool IgnoreNull() {
		return true;
	}
};

} // namespace duckdb
