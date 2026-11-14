#include "duckdb/common/serializer/serializer.hpp"

#include "duckdb/common/helper.hpp"

namespace duckdb {

void Serializer::List::WriteElement(data_ptr_t ptr, idx_t size) {
	serializer.WriteDataPtr(ptr, size);
}

void Serializer::BeginDataParts(idx_t size) {
	parts = make_unsafe_uniq_array_uninitialized<data_t>(size);
	parts_size = size;
	parts_written = 0;
}

void Serializer::WriteDataPart(const_data_ptr_t ptr, idx_t count) {
	D_ASSERT(parts_written + count <= parts_size);
	if (count != 0) {
		memcpy(parts.get() + parts_written, ptr, count);
	}
	parts_written += count;
}

data_ptr_t Serializer::AllocDataPart(idx_t count) {
	D_ASSERT(parts_written + count <= parts_size);
	const auto allocated = parts.get() + parts_written;
	parts_written += count;
	return allocated;
}

void Serializer::EndDataParts() {
	D_ASSERT(parts_written == parts_size);
	WriteDataPtr(parts.get(), parts_size);
	parts.reset();
	parts_size = 0;
	parts_written = 0;
}

} // namespace duckdb
