//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/radix.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/bswap.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/limits.hpp"

#include <cfloat>
#include <cstring> // strlen() on Solaris
#include <limits.h>

namespace duckdb {

struct Radix {
public:
	template <class T>
	static inline void EncodeData(data_ptr_t dataptr, T value) {
		throw NotImplementedException("Cannot create data from this type");
	}

	template <class T>
	static inline T DecodeData(const_data_ptr_t input) {
		throw NotImplementedException("Cannot read data from this type");
	}

	static inline uint32_t EncodeFloat(float x) {
		return EncodeFloating<float, uint32_t>(x);
	}

	static inline float DecodeFloat(uint32_t input) {
		return DecodeFloating<float, uint32_t>(input);
	}

	static inline uint64_t EncodeDouble(double x) {
		return EncodeFloating<double, uint64_t>(x);
	}

	static inline double DecodeDouble(uint64_t input) {
		return DecodeFloating<double, uint64_t>(input);
	}

private:
	template <class T>
	static void EncodeSigned(data_ptr_t dataptr, T value);
	template <class T>
	static T DecodeSigned(const_data_ptr_t input);

	//! Positive numbers set the sign bit and negative numbers flip every bit, without a branch on the sign; both zeros
	//! share a key, every NaN sorts last and the infinities take the keys next to the ends.
	template <class F, class U>
	static inline U EncodeFloating(F x) {
		constexpr U SIGN = U(1) << (sizeof(U) * 8 - 1);
		constexpr U INFINITE = std::bit_cast<U>(std::numeric_limits<F>::infinity());
		const auto bits = std::bit_cast<U>(x);
		const auto magnitude = static_cast<U>(bits & ~SIGN);
		if (static_cast<U>(magnitude - 1) >= static_cast<U>(INFINITE - 1)) [[unlikely]] {
			if (magnitude == 0) {
				return SIGN;
			}
			if (magnitude > INFINITE) {
				return ~U(0);
			}
			return bits & SIGN ? U(0) : static_cast<U>(~U(0) - 1);
		}
		return static_cast<U>(bits ^
		                      (static_cast<U>(static_cast<std::make_signed_t<U>>(bits) >> (sizeof(U) * 8 - 1)) | SIGN));
	}

	template <class F, class U>
	static inline F DecodeFloating(U input) {
		constexpr U SIGN = U(1) << (sizeof(U) * 8 - 1);
		auto bits = (input & SIGN) != 0 ? static_cast<U>(input ^ SIGN) : static_cast<U>(~input);
		bits = input == 0 ? std::bit_cast<U>(-std::numeric_limits<F>::infinity()) : bits;
		bits = input == static_cast<U>(~U(0) - 1) ? std::bit_cast<U>(std::numeric_limits<F>::infinity()) : bits;
		return std::bit_cast<F>(bits);
	}
};

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, bool value) {
	Store<uint8_t>(value ? 1 : 0, dataptr);
}

template <class T>
void Radix::EncodeSigned(data_ptr_t dataptr, T value) {
	using UNSIGNED = typename MakeUnsigned<T>::type;
	constexpr auto SIGN = static_cast<UNSIGNED>(UNSIGNED(1) << (sizeof(UNSIGNED) * 8 - 1));
	Store<UNSIGNED>(BSwapIfLE(static_cast<UNSIGNED>(static_cast<UNSIGNED>(value) ^ SIGN)), dataptr);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, int8_t value) {
	EncodeSigned<int8_t>(dataptr, value);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, int16_t value) {
	EncodeSigned<int16_t>(dataptr, value);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, int32_t value) {
	EncodeSigned<int32_t>(dataptr, value);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, int64_t value) {
	EncodeSigned<int64_t>(dataptr, value);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, uint8_t value) {
	Store<uint8_t>(value, dataptr);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, uint16_t value) {
	Store<uint16_t>(BSwapIfLE(value), dataptr);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, uint32_t value) {
	Store<uint32_t>(BSwapIfLE(value), dataptr);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, uint64_t value) {
	Store<uint64_t>(BSwapIfLE(value), dataptr);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, hugeint_t value) {
	EncodeData<int64_t>(dataptr, value.upper);
	EncodeData<uint64_t>(dataptr + sizeof(value.upper), value.lower);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, uhugeint_t value) {
	EncodeData<uint64_t>(dataptr, value.upper);
	EncodeData<uint64_t>(dataptr + sizeof(value.upper), value.lower);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, float value) {
	uint32_t converted_value = EncodeFloat(value);
	Store<uint32_t>(BSwapIfLE(converted_value), dataptr);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, double value) {
	uint64_t converted_value = EncodeDouble(value);
	Store<uint64_t>(BSwapIfLE(converted_value), dataptr);
}

template <>
inline void Radix::EncodeData(data_ptr_t dataptr, interval_t value) {
	EncodeData<int32_t>(dataptr, value.months);
	dataptr += sizeof(value.months);
	EncodeData<int32_t>(dataptr, value.days);
	dataptr += sizeof(value.days);
	EncodeData<int64_t>(dataptr, value.micros);
}

template <>
inline bool Radix::DecodeData(const_data_ptr_t input) {
	return Load<uint8_t>(input) != 0;
}

template <class T>
T Radix::DecodeSigned(const_data_ptr_t input) {
	using UNSIGNED = typename MakeUnsigned<T>::type;
	constexpr auto SIGN = static_cast<UNSIGNED>(UNSIGNED(1) << (sizeof(UNSIGNED) * 8 - 1));
	return static_cast<T>(static_cast<UNSIGNED>(BSwapIfLE(Load<UNSIGNED>(input)) ^ SIGN));
}

template <>
inline int8_t Radix::DecodeData(const_data_ptr_t input) {
	return DecodeSigned<int8_t>(input);
}

template <>
inline int16_t Radix::DecodeData(const_data_ptr_t input) {
	return DecodeSigned<int16_t>(input);
}

template <>
inline int32_t Radix::DecodeData(const_data_ptr_t input) {
	return DecodeSigned<int32_t>(input);
}

template <>
inline int64_t Radix::DecodeData(const_data_ptr_t input) {
	return DecodeSigned<int64_t>(input);
}

template <>
inline uint8_t Radix::DecodeData(const_data_ptr_t input) {
	return Load<uint8_t>(input);
}

template <>
inline uint16_t Radix::DecodeData(const_data_ptr_t input) {
	return BSwapIfLE(Load<uint16_t>(input));
}

template <>
inline uint32_t Radix::DecodeData(const_data_ptr_t input) {
	return BSwapIfLE(Load<uint32_t>(input));
}

template <>
inline uint64_t Radix::DecodeData(const_data_ptr_t input) {
	return BSwapIfLE(Load<uint64_t>(input));
}

template <>
inline hugeint_t Radix::DecodeData(const_data_ptr_t input) {
	hugeint_t result;
	result.upper = DecodeData<int64_t>(input);
	result.lower = DecodeData<uint64_t>(input + sizeof(int64_t));
	return result;
}

template <>
inline uhugeint_t Radix::DecodeData(const_data_ptr_t input) {
	uhugeint_t result;
	result.upper = DecodeData<uint64_t>(input);
	result.lower = DecodeData<uint64_t>(input + sizeof(uint64_t));
	return result;
}

template <>
inline float Radix::DecodeData(const_data_ptr_t input) {
	return DecodeFloat(BSwapIfLE(Load<uint32_t>(input)));
}

template <>
inline double Radix::DecodeData(const_data_ptr_t input) {
	return DecodeDouble(BSwapIfLE(Load<uint64_t>(input)));
}

template <>
inline interval_t Radix::DecodeData(const_data_ptr_t input) {
	interval_t result;
	result.months = DecodeData<int32_t>(input);
	result.days = DecodeData<int32_t>(input + sizeof(int32_t));
	result.micros = DecodeData<int64_t>(input + sizeof(int32_t) + sizeof(int32_t));
	return result;
}

} // namespace duckdb
