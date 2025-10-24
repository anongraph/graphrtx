#include <cuda_runtime.h>

#include "test_configuration.h"

#include "launch_parameters.cuh"
#include "optix_helpers.cuh"


extern "C" __constant__ launch_parameters params;


extern "C" __global__ void __closesthit__test() {
    // normally, do nothing here ...

    // ... unless we explicitly force handling hits in this shader
#if USE_CLOSESTHIT_INSTEAD_OF_ANYHIT != 0
    const uint32_t primitive_id = optixGetPrimitiveIndex();
    value_type value = params.build_values[primitive_id];
    set_payload_32(value);
#endif
}


extern "C" __global__ void __miss__test() {
    // do nothing
}


// this function is called for every potential ray-aabb intersection
extern "C" __global__ void __intersection__test() {
    const uint32_t ix = optixGetLaunchIndex().x;
    const uint32_t primitive_id = optixGetPrimitiveIndex();
 
    key_type hit = params.build_keys[primitive_id];
    key_type lower_bound = params.query_lower[ix];

#if RANGE_QUERY_HIT_COUNT_LOG != 0
    key_type upper_bound = params.query_upper[ix];
    if (!(lower_bound <= hit && hit <= upper_bound))
        return;
#else
    if (lower_bound != hit)
        return;
#endif

    value_type value = params.build_values[primitive_id];
#if MULTIPLE_HITS_PER_RAY != 0
    set_payload_32(get_payload_32<value_type>() + value);
    // do not report hit, since we effectively inlined the any-hit shader here!
#else
    set_payload_32(value);
    // the ray should be terminated now, but that is only possible from the any-hit shader
#endif
}


// this function is called for every reported (i.e. confirmed) ray-primitive intersection
extern "C" __global__ void __anyhit__test() {
#if USE_CLOSESTHIT_INSTEAD_OF_ANYHIT != 0
    // entering this block means that something went wrong when disabling the any-hit shader
    uint8_t kill = *(uint8_t*)nullptr;
#endif

    const uint32_t primitive_id = optixGetPrimitiveIndex();

    value_type value = params.build_values[primitive_id];

#if MULTIPLE_HITS_PER_RAY != 0
    // aggregate all hits
    set_payload_32(get_payload_32<value_type>() + value);
    // reject the hit, this prevents tmax from being reduced
    optixIgnoreIntersection();
#else
    // just override the value
    set_payload_32(value);
#endif
}


// this is the entry point
extern "C" __global__ void __raygen__test() {
    constexpr uint32_t rays_per_thread = size_t{1} << NUM_RAYS_PER_THREAD_LOG;
#if USE_CLOSESTHIT_INSTEAD_OF_ANYHIT != 0
    constexpr uint32_t ray_flags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT;
#else
    constexpr uint32_t ray_flags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
#endif

    constexpr size_t num_rays_per_thread = size_t{1} << NUM_RAYS_PER_THREAD_LOG;
    const uint32_t thread_offset = optixGetLaunchIndex().x * num_rays_per_thread;

    for (uint32_t ix = thread_offset; ix < thread_offset + rays_per_thread; ++ix) {

#if INT_TO_FLOAT_CONVERSION_MODE == 3

        // point query vs range query distinction
        // this can also be decided at runtime
#if RANGE_QUERY_HIT_COUNT_LOG != 0
        key_type lower_bound = params.query_lower[ix];
        key_type upper_bound = params.query_upper[ix];

        // decompose the key into x and yz
        key_type smallest_yz = lower_bound >> x_bits;
        key_type largest_yz = upper_bound >> x_bits;
        float smallest_x = uint32_as_float(lower_bound & x_mask);
        float largest_x = uint32_as_float(upper_bound & x_mask);
        float smallest_possible_x = uint32_as_float(0);
        float largest_possible_x = uint32_as_float(x_mask);

        value_type i0 = 0;
        // cast one ray per yz offset
        for (uint64_t yz = smallest_yz; yz <= largest_yz; ++yz) {
            float offset_y = uint32_as_float(yz & y_mask);
            float offset_z = z_bits == 0 ? uint32_as_float(0) : uint32_as_float(yz >> y_bits);

#if START_RAY_AT_ZERO != 0
            float3 origin = make_float3(0, offset_y, offset_z);
            float3 direction = make_float3(1, 0, 0);
            float tmin = minus_eps(yz == smallest_yz ? smallest_x : smallest_possible_x);
            float tmax = plus_eps(yz == largest_yz ? largest_x : largest_possible_x);
#else
            float start_x = minus_eps(yz == smallest_yz ? smallest_x : smallest_possible_x);
            float end_x = plus_eps(yz == largest_yz ? largest_x : largest_possible_x);
            float3 origin = make_float3(start_x, offset_y, offset_z);
            float3 direction = make_float3(1, 0, 0);
            float tmin = 0;
            float tmax = end_x - start_x;
#endif

            optixTrace(
                    params.traversable,
                    origin,
                    direction,
                    tmin,
                    tmax,
                    0.0f,
                    OptixVisibilityMask(255),
                    // we can use TERMINATE_ON_FIRST_HIT for a range
                    // query since all hits will be rejected anyway
                    ray_flags,
                    0,
                    0,
                    0,
                    i0);
        }
        params.result[ix] = i0;

#else // RANGE_QUERY_HIT_COUNT_LOG == 0

        key_type key = params.query_lower[ix];

        float offset_x, offset_y, offset_z;
        key_to_coordinates(key, offset_x, offset_y, offset_z);

        float start_x = minus_eps(offset_x);
        float end_x = plus_eps(offset_x);
        float start_z = minus_eps(offset_z);
        float end_z = plus_eps(offset_z);

#if PERPENDICULAR_RAYS != 0
#if START_RAY_AT_ZERO != 0
        float3 origin = make_float3(offset_x, offset_y, 0);
        float3 direction = make_float3(0, 0, 1);
        float tmin = start_z;
        float tmax = end_z;
#else
        float3 origin = make_float3(offset_x, offset_y, start_z);
        float3 direction = make_float3(0, 0, 1);
        float tmin = 0;
        float tmax = end_z - start_z;
#endif
#else // PERPENDICULAR_RAYS == 0
#if START_RAY_AT_ZERO != 0
        float3 origin = make_float3(0, offset_y, offset_z);
        float3 direction = make_float3(1, 0, 0);
        float tmin = start_x;
        float tmax = end_x;
#else
        float3 origin = make_float3(start_x, offset_y, offset_z);
        float3 direction = make_float3(1, 0, 0);
        float tmin = 0;
        float tmax = end_x - start_x;
#endif
#endif
    
        value_type i0 = NOT_FOUND;
        optixTrace(
                params.traversable,
                origin,
                direction,
                tmin,
                tmax,
                0.0f,
                OptixVisibilityMask(255),
                ray_flags,
                0,
                0,
                0,
                i0);
        params.result[ix] = i0;
#endif // RANGE_QUERY_HIT_COUNT_LOG


#else // INT_TO_FLOAT_CONVERSION_MODE != 3

        float zero = uint32_as_float(0);

        // point query vs range query distinction
        // this can also be decided at run-time
#if RANGE_QUERY_HIT_COUNT_LOG != 0
        float lower_bound = uint32_as_float(params.query_lower[ix]);
        float upper_bound = uint32_as_float(params.query_upper[ix]);

#if START_RAY_AT_ZERO != 0
        float3 origin = make_float3(0, zero, zero);
        float3 direction = make_float3(1, 0, 0);
        float tmin = minus_eps(lower_bound);
        float tmax = plus_eps(upper_bound);
#else
        float start_x = minus_eps(lower_bound);
        float end_x = plus_eps(upper_bound);
        float3 origin = make_float3(start_x, zero, zero);
        float3 direction = make_float3(1, 0, 0);
        float tmin = 0;
        float tmax = end_x - start_x;
#endif

        value_type i0 = 0;
        optixTrace(
                params.traversable,
                origin,
                direction,
                tmin,
                tmax,
                0.0f,
                OptixVisibilityMask(255),
                // we can use TERMINATE_ON_FIRST_HIT for a range
                // query since all hits will be rejected anyway
                ray_flags,
                0,
                0,
                0,
                i0);
        params.result[ix] = i0;
#else // RANGE_QUERY_HIT_COUNT_LOG == 0

        float key = uint32_as_float(params.query_lower[ix]);

#if PERPENDICULAR_RAYS != 0
        float3 origin = make_float3(key, zero, 0);
        float3 direction = make_float3(0, 0, 1);
        float tmin = 0;
        float tmax = plus_eps(zero);
#else
#if START_RAY_AT_ZERO != 0
        float3 origin = make_float3(0, zero, zero);
        float3 direction = make_float3(1, 0, 0);
        float tmin = minus_eps(key);
        float tmax = plus_eps(key);
#else
        float start_x = minus_eps(key);
        float end_x = plus_eps(key);
        float3 origin = make_float3(start_x, zero, zero);
        float3 direction = make_float3(1, 0, 0);
        float tmin = 0;
        float tmax = end_x - start_x;
#endif
#endif

        value_type i0 = NOT_FOUND;
        optixTrace(
                params.traversable,
                origin,
                direction,
                tmin,
                tmax,
                0.0f,
                OptixVisibilityMask(255),
                ray_flags,
                0,
                0,
                0,
                i0);
        params.result[ix] = i0;
#endif // RANGE_QUERY_HIT_COUNT_LOG
#endif // INT_TO_FLOAT_CONVERSION_MODE
    }
}
