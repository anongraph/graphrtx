#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "cuda_helpers.cuh"

#include "test_configuration.h"


#if LARGE_KEYS != 0
using key_type = uint64_t;
#else
using key_type = uint32_t;
#endif

using value_type = uint32_t;

HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr float bias_exponent(float f) {
    float shift = float(size_t{1} << std::abs(EXPONENT_BIAS));
#if EXPONENT_BIAS < 0
    return f / shift;
#else
    return f * shift; 
#endif
}

// DISTANCE BETWEEN FLOATS
constexpr float delta = bias_exponent(1.0f);

#if INT_TO_FLOAT_CONVERSION_MODE == 1
// this works with triangles because they have to satisfy tmin < t < tmax
// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#ray-extents
constexpr float eps = delta;
#else
constexpr float eps = delta / 2;
#endif

HOSTDEVICEQUALIFIER INLINEQUALIFIER
float uint32_as_float(uint32_t i) {
#if INT_TO_FLOAT_CONVERSION_MODE == 2
    // CONFIRMED TO BE WORKING UP TO 2^29
    // after that, we quickly reach NaN range
    //uint32_t skip = (i << 1u) + 0x2f800000u; // constant chosen by trial and error
    uint32_t skip = (i << 1u) + 0x3f000000u; // constant is 0.5 in binary
    float result = *reinterpret_cast<float*>(&skip);
#elif INT_TO_FLOAT_CONVERSION_MODE == 1
    // CONFIRMED TO BE WORKING UP TO (2^24 - 2)
    float result = static_cast<float>(i) + 1;
#else
    // CONFIRMED TO BE WORKING UP TO (2^23 - 1)
    float result = static_cast<float>(i) + 0.5;
#endif
    return bias_exponent(result);
}

constexpr uint32_t x_bits = KEY_DECOMPOSITION / 10000 % 100;
constexpr uint32_t y_bits = KEY_DECOMPOSITION / 100 % 100;
constexpr uint32_t z_bits = KEY_DECOMPOSITION % 100;
constexpr key_type x_mask = key_type((size_t{1} << x_bits) - 1);
constexpr key_type y_mask = key_type((size_t{1} << y_bits) - 1);


HOSTDEVICEQUALIFIER INLINEQUALIFIER
void key_to_coordinates(key_type key, float& x, float& y, float& z) {
#if INT_TO_FLOAT_CONVERSION_MODE == 3
    static_assert(x_bits > 0);
    static_assert(x_bits < 24);
    static_assert(y_bits < 24);
    static_assert(z_bits < 24);
    static_assert(x_bits + y_bits + z_bits == sizeof(key_type) * 8);

    x = uint32_as_float(key & x_mask);
    y = uint32_as_float((key >> x_bits) & y_mask);
    if constexpr (z_bits > 0) {
        z = uint32_as_float(key >> (x_bits + y_bits));
    } else {
        z = uint32_as_float(0);
    }

#else
    x = uint32_as_float(key);
    y = uint32_as_float(0);
    z = uint32_as_float(0);
#endif
}


#if INT_TO_FLOAT_CONVERSION_MODE == 0
constexpr size_t max_key = (size_t{1} << 23u) - 1u;
#elif INT_TO_FLOAT_CONVERSION_MODE == 1
constexpr size_t max_key = (size_t{1} << 24u) - 2u;
#elif INT_TO_FLOAT_CONVERSION_MODE == 2
constexpr size_t max_key = (size_t{1} << 29u) - 1u;
#elif INT_TO_FLOAT_CONVERSION_MODE == 3
constexpr size_t max_key = std::numeric_limits<size_t>::max();
#else
#error illegal conversion mode
#endif


constexpr value_type NOT_FOUND = 0;


HOSTDEVICEQUALIFIER INLINEQUALIFIER
float plus_eps(float f) {
#if INT_TO_FLOAT_CONVERSION_MODE == 2
    uint32_t i = *reinterpret_cast<uint32_t*>(&f);
    i += 1;
    return *reinterpret_cast<float*>(&i);
#elif INT_TO_FLOAT_CONVERSION_MODE == 1
    // treat largest key as a special case (not currently used)
    //constexpr float precision_limit = float(size_t{1} << 24u);
    //constexpr float next_after_precision_limit = float((size_t{1} << 24u) + 2);
    //return f == precision_limit ? next_after_precision_limit : f + eps;
    return f + eps;
#else
    return f + eps;
#endif
}

HOSTDEVICEQUALIFIER INLINEQUALIFIER
float minus_eps(float f) {
#if INT_TO_FLOAT_CONVERSION_MODE == 2
    uint32_t i = *reinterpret_cast<uint32_t*>(&f);
    i -= 1;
    return *reinterpret_cast<float*>(&i);
#else
    return f - eps;
#endif
}


#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                      \
      }                                                                 \
  }


template <typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type unpack(uint32_t i0, uint32_t i1) {
    static_assert(sizeof(packed_type) == 8);
    uint64_t uptr = static_cast<uint64_t>(i0) << 32u | i1;
    packed_type ptr = *reinterpret_cast<packed_type*>(&uptr);
    return ptr;
}

template <typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void pack(packed_type ptr, uint32_t& i0, uint32_t& i1) {
    static_assert(sizeof(packed_type) == 8);
    const uint64_t uptr = *reinterpret_cast<uint64_t*>(&ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffffull;
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type get_payload_64() {
    static_assert(sizeof(packed_type) == 8);
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpack<packed_type>(u0, u1);
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void set_payload_64(packed_type i) {
    static_assert(sizeof(packed_type) == 8);
    uint32_t i0, i1;
    pack(i, i0, i1);
    optixSetPayload_0(i0);
    optixSetPayload_1(i1);
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
packed_type get_payload_32() {
    static_assert(sizeof(packed_type) == 4);
    return (packed_type) optixGetPayload_0();
}

template<typename packed_type>
DEVICEQUALIFIER INLINEQUALIFIER
void set_payload_32(packed_type i) {
    static_assert(sizeof(packed_type) == 4);
    optixSetPayload_0(i);
}
