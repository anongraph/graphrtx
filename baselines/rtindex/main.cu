#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <set>
#include <unordered_set>

#ifndef __CUDACC_RTC__
#include <thread>
#endif

#include <cub/cub.cuh>
#include <nvtx3/nvtx3.hpp>

#include "test_configuration.h"

#include "cuda_helpers.cuh"
#include "optix_wrapper.h"
#include "optix_pipeline.h"
#include "optix_helpers.cuh"
#include "launch_parameters.cuh"


#define OVAR(x) (#x) << "=" << (x)
#define IVAR(x) ("i_" #x) << "=" << (x)


void convert_keys_to_primitives(
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer& primitive_buffer,
    double& convert_time_ms
) {
    cudaEvent_t convert_start, convert_stop;
    cudaEventCreate(&convert_start);
    cudaEventCreate(&convert_stop);

#if PRIMITIVE == 0
    primitive_buffer.alloc(3 * key_count * sizeof(float3));
    auto buffer_pointer = primitive_buffer.ptr<float3>();

    cudaEventRecord(convert_start, 0);
    lambda_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>([=] DEVICEQUALIFIER {
        const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= key_count) return;

        float x, y, z;
        key_to_coordinates(keys_device_pointer[tid], x, y, z);

        float just_below_x = minus_eps(x);
        float just_above_x = plus_eps(x);
        float just_below_y = minus_eps(y);
        float just_above_y = plus_eps(y);
        float just_below_z = minus_eps(z);
        float just_above_z = plus_eps(z);

        // triangle (-eps, eps, -eps) -- (-eps, -eps, -eps) -- (eps, 0, eps) includes the point (0, 0, 0)
        // offset this triangle in xyz direction
        buffer_pointer[3 * tid + 0] = make_float3(just_below_x, just_above_y, just_below_z);
        buffer_pointer[3 * tid + 1] = make_float3(just_below_x, just_below_y, just_below_z);
        buffer_pointer[3 * tid + 2] = make_float3(just_above_x,            y, just_above_z);
    });
    cudaEventRecord(convert_stop, 0);

#elif PRIMITIVE == 1
    primitive_buffer.alloc(key_count * sizeof(float3));
    auto buffer_pointer = primitive_buffer.ptr<float3>();

    cudaEventRecord(convert_start, 0);
    lambda_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>([=] DEVICEQUALIFIER {
        const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= key_count) return;

        key_type key = keys_device_pointer[tid];
        float x, y, z;
        key_to_coordinates(keys_device_pointer[tid], x, y, z);

        buffer_pointer[tid] = make_float3(x, y, z);
    });
    cudaEventRecord(convert_stop, 0);

#elif PRIMITIVE == 2
    primitive_buffer.alloc(2 * key_count * sizeof(float3));
    auto buffer_pointer = primitive_buffer.ptr<float3>();

    cudaEventRecord(convert_start, 0);
    lambda_kernel<<<SDIV(key_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>([=] DEVICEQUALIFIER {
        const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= key_count) return;

        key_type key = keys_device_pointer[tid];
        float x, y, z;
        key_to_coordinates(keys_device_pointer[tid], x, y, z);

        float just_above_x = plus_eps(x);
        float just_above_y = plus_eps(y);
        float just_above_z = plus_eps(z);

        buffer_pointer[2 * tid + 0] = make_float3(x, y, z);
        buffer_pointer[2 * tid + 1] = make_float3(just_above_x, just_above_y, just_above_z);
    });
    cudaEventRecord(convert_stop, 0);

#else
#error unknown primitive type
#endif

    cudaEventSynchronize(convert_stop);
    float delta;
    cudaEventElapsedTime(&delta, convert_start, convert_stop);
    convert_time_ms = delta;
    cudaDeviceSynchronize(); CUERR
}

void setup_structure_input(OptixBuildInput& bi, void** buffer, void** secondary_buffer, size_t key_count) {

#if FORCE_SINGLE_ANYHIT == 1
    static uint32_t build_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
#else
    static uint32_t build_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
#endif

#if PRIMITIVE == 0
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.numVertices         = 3 * (unsigned) key_count;
    bi.triangleArray.vertexBuffers       = (CUdeviceptr*) buffer;
    bi.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.numIndexTriplets    = 0;
    bi.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
    bi.triangleArray.preTransform        = 0;
    bi.triangleArray.flags               = build_input_flags;
    bi.triangleArray.numSbtRecords       = 1;

#elif PRIMITIVE == 1
    bi.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    bi.sphereArray.numVertices         = (unsigned) key_count;
    bi.sphereArray.vertexBuffers       = (CUdeviceptr*) buffer;
    bi.sphereArray.radiusBuffers       = (CUdeviceptr*) secondary_buffer;
    bi.sphereArray.vertexStrideInBytes = 0;
    bi.sphereArray.radiusStrideInBytes = 0;
    bi.sphereArray.singleRadius        = true;
    bi.sphereArray.flags               = build_input_flags;
    bi.sphereArray.numSbtRecords       = 1;

#elif PRIMITIVE == 2
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.numPrimitives = (unsigned) key_count;
    bi.customPrimitiveArray.aabbBuffers   = (CUdeviceptr*) buffer;
    bi.customPrimitiveArray.strideInBytes = 0;
    bi.customPrimitiveArray.flags         = build_input_flags;
    bi.customPrimitiveArray.numSbtRecords = 1;

#else
#error unknown primitive type
#endif
}

void setup_build_input(OptixAccelBuildOptions& bi, bool update = false) {
    bi.buildFlags = OPTIX_BUILD_FLAG_NONE;
#if COMPACTION != 0
    bi.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
#endif
#if NUM_UPDATES_LOG > 0
    bi.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
#endif
    bi.motionOptions.numKeys = 1;
    bi.operation = update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
}

OptixTraversableHandle build_traversable(
    const optix_wrapper& optix,
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer& as_buffer,
    size_t& uncompacted_size,
    size_t& final_size,
    double& convert_time_ms,
    double& build_time_ms,
    double& compact_time_ms
) {
    uncompacted_size = 0;
    final_size = 0;
    convert_time_ms = 0;
    build_time_ms = 0;
    compact_time_ms = 0;

    cuda_buffer primitive_buffer;
    cuda_buffer secondary_primitive_buffer;
    convert_keys_to_primitives(keys_device_pointer, key_count, primitive_buffer, convert_time_ms);
#if PRIMITIVE == 1
    // we need an additional radius buffer for the spheres
    // in theory, we could append to the other buffer, but there might be alignment issues
    std::vector<float> default_radius{0.25};
    secondary_primitive_buffer.alloc_and_upload(default_radius);
#endif

    OptixTraversableHandle structure_handle{0};

    OptixBuildInput structure_input = {};
    setup_structure_input(structure_input, &primitive_buffer.raw_ptr, &secondary_primitive_buffer.raw_ptr, key_count);

    OptixAccelBuildOptions structure_options = {};
    setup_build_input(structure_options);

    OptixAccelBufferSizes structure_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix.optix_context,
            &structure_options,
            &structure_input,
            1,  // num_build_inputs
            &structure_buffer_sizes
    ))

    uncompacted_size = structure_buffer_sizes.outputSizeInBytes;

#if COMPACTION == 1
    // ==================================================================
    // prepare compaction
    // ==================================================================
    cuda_buffer compacted_size_buffer;
    compacted_size_buffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emit_desc;
    emit_desc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = compacted_size_buffer.cu_ptr();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    cuda_buffer uncompacted_structure_buffer;
    uncompacted_structure_buffer.alloc(structure_buffer_sizes.outputSizeInBytes);
#else
    final_size = uncompacted_size;
    as_buffer.alloc(structure_buffer_sizes.outputSizeInBytes);
#endif

    cuda_buffer temp_buffer;
    temp_buffer.alloc(structure_buffer_sizes.tempSizeInBytes);

    cudaEvent_t build_start, build_stop;
    cudaEventCreate(&build_start);
    cudaEventCreate(&build_stop);
    cudaEventRecord(build_start, optix.stream);

    OPTIX_CHECK(optixAccelBuild(
            optix.optix_context,
            optix.stream,
            &structure_options,
            &structure_input,
            1,
            temp_buffer.cu_ptr(),
            temp_buffer.size_in_bytes,
#if COMPACTION == 1
            uncompacted_structure_buffer.cu_ptr(),
            uncompacted_structure_buffer.size_in_bytes,
#else
            as_buffer.cu_ptr(),
            as_buffer.size_in_bytes,
#endif
            &structure_handle,
#if COMPACTION == 1
            &emit_desc, 1
#else
            nullptr, 0
#endif
    ))

    cudaEventRecord(build_stop, optix.stream);
    cudaEventSynchronize(build_stop);
    float build_delta;
    cudaEventElapsedTime(&build_delta, build_start, build_stop);
    build_time_ms = build_delta;
    cudaDeviceSynchronize(); CUERR

    primitive_buffer.free();
    secondary_primitive_buffer.free();

#if COMPACTION == 1
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compacted_size;
    compacted_size_buffer.download(&compacted_size, 1);
    final_size = compacted_size;

    as_buffer.alloc(compacted_size);

    cudaEvent_t compact_start, compact_stop;
    cudaEventCreate(&compact_start);
    cudaEventCreate(&compact_stop);
    cudaEventRecord(compact_start, optix.stream);
    OPTIX_CHECK(optixAccelCompact(
            optix.optix_context,
            optix.stream,
            structure_handle,
            as_buffer.cu_ptr(),
            as_buffer.size_in_bytes,
            &structure_handle));

    cudaEventRecord(compact_stop, optix.stream);
    cudaEventSynchronize(compact_stop);
    float compact_delta;
    cudaEventElapsedTime(&compact_delta, compact_start, compact_stop);
    compact_time_ms = compact_delta;
    cudaDeviceSynchronize(); CUERR

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    uncompacted_structure_buffer.free();
    compacted_size_buffer.free();
#endif
    temp_buffer.free();

    return structure_handle;
}

void update_traversable(
    const optix_wrapper& optix,
    const key_type* keys_device_pointer,
    size_t key_count,
    cuda_buffer& as_buffer,
    OptixTraversableHandle structure_handle,
    size_t& update_temp_buffer_size,
    double& update_convert_time_ms,
    double& update_time_ms
) {
    update_temp_buffer_size = 0;
    update_time_ms = 0;

    cuda_buffer primitive_buffer;
    cuda_buffer secondary_primitive_buffer;
    convert_keys_to_primitives(keys_device_pointer, key_count, primitive_buffer, update_convert_time_ms);
#if PRIMITIVE == 1
    // we need an additional radius buffer for the spheres
    // in theory, we could append to the other buffer, but there might be alignment issues
    std::vector<float> default_radius{0.25};
    secondary_primitive_buffer.alloc_and_upload(default_radius);
#endif

    OptixBuildInput structure_input = {};
    setup_structure_input(structure_input, &primitive_buffer.raw_ptr, &secondary_primitive_buffer.raw_ptr, key_count);

    OptixAccelBuildOptions structure_options = {};
    setup_build_input(structure_options, true);


    OptixAccelBufferSizes structure_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix.optix_context,
            &structure_options,
            &structure_input,
            1,  // num_build_inputs
            &structure_buffer_sizes
    ))

    update_temp_buffer_size = structure_buffer_sizes.tempUpdateSizeInBytes;
    cuda_buffer temp_buffer;
    temp_buffer.alloc(update_temp_buffer_size);

    cudaEvent_t build_start, build_stop;
    cudaEventCreate(&build_start);
    cudaEventCreate(&build_stop);
    cudaEventRecord(build_start, optix.stream);

    OPTIX_CHECK(optixAccelBuild(
            optix.optix_context,
            optix.stream,
            &structure_options,
            &structure_input,
            1,
            temp_buffer.cu_ptr(),
            temp_buffer.size_in_bytes,
            as_buffer.cu_ptr(),
            as_buffer.size_in_bytes,
            &structure_handle,
            nullptr, 0
    ))

    cudaEventRecord(build_stop, optix.stream);
    cudaEventSynchronize(build_stop);
    float build_delta;
    cudaEventElapsedTime(&build_delta, build_start, build_stop);
    update_time_ms = build_delta;
    cudaDeviceSynchronize(); CUERR
    temp_buffer.free();
}


// https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of
template <typename key_type, typename compare_type = std::less<key_type>>
std::vector<std::size_t> sort_permutation(const std::vector<key_type>& vec, compare_type compare = {}) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}
template <typename key_type>
void apply_permutation(std::vector<key_type>& vec, const std::vector<std::size_t>& permutation) {
    std::vector<key_type> sorted_vec(vec.size());
    std::transform(permutation.begin(), permutation.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });
    std::swap(vec, sorted_vec);
}


constexpr value_type value_for(key_type key) {
    return ((value_type) key) * 3 / 2;
}


template <typename random_type>
bool generate_dense_build_set(random_type gen, size_t num_keys, size_t key_stride, size_t key_offset, bool swap_minmax_to_end, std::vector<key_type>& generated_keys) {
    generated_keys.resize(num_keys);
    // generate dense keys with the specified offset and stride and replication
    for (size_t i = 0; i < generated_keys.size(); ++i) {
        size_t new_key = i * key_stride + key_offset;
        if (new_key > max_key) return false;
        generated_keys[i] = key_type(new_key);
    }
    // shuffle the keys
    if (swap_minmax_to_end) {
        // reserve the first and the last key for out-of-range misses by swapping them to the end
        std::swap(generated_keys[0], generated_keys[num_keys - 2]);
        std::shuffle(generated_keys.begin(), generated_keys.end() - 2, gen);
    } else {
        std::shuffle(generated_keys.begin(), generated_keys.end(), gen);
    }
    return true;
}


template <typename random_type>
bool generate_uniform_build_set(random_type gen, size_t num_keys, size_t key_stride, size_t key_offset, bool reserve_minmax, std::vector<key_type>& generated_keys) {
    generated_keys.resize(num_keys);

    size_t base_max_key = (max_key - key_offset) / key_stride;
    std::uniform_int_distribution<size_t> dist(0, base_max_key);

    // generate dense keys with the specified offset and stride and replication
    std::unordered_set<key_type> generated_keys_set;
    while (generated_keys_set.size() < num_keys) {
        size_t new_key = dist(gen) * key_stride + key_offset;
        // this should never happen, but check anyway just to be sure
        if (new_key > max_key) return false;
        if (new_key < key_offset) return false;
        if ((new_key - key_offset) % key_stride != 0) return false;
        generated_keys_set.insert((key_type) new_key);
    }
    std::copy(generated_keys_set.begin(), generated_keys_set.end(), generated_keys.begin());
    if (reserve_minmax) {
        // reserve the first and the last key for out-of-range misses by swapping them to the end
        auto min_it = std::min_element(generated_keys.begin(), generated_keys.end());
        auto max_it = std::max_element(generated_keys.begin(), generated_keys.end());
        std::swap(*min_it, generated_keys[num_keys - 2]);
        std::swap(*max_it, generated_keys[num_keys - 1]);
    }
    return true;
}


bool generate_input(
    // number of unique keys to generate
    size_t num_build_keys,
    // uniformly pick this many keys from the build set to form the probe set
    size_t num_probe_keys,
    // distance between keys (1 means dense)
    size_t key_stride,
    // first key to generate
    size_t key_offset,
    // some of the build keys will not be inserted so that some probes will miss
    bool reserve_keys_for_misses,
    // percentage of probe keys which will not occur in the build set
    size_t miss_percent,
    // percentage of probe keys which will be strictly smaller or larger than all keys in the build set
    size_t out_of_range_percent,
    // replicate each key so that the generated build set contains duplicates
    size_t key_replication,
    // draw keys from the entire range of values
    bool uniform_keys,
    std::vector<key_type>& build_keys,
    std::vector<value_type>& build_values,
    std::vector<key_type>& probe_keys,
    std::vector<value_type>& expected_values
) {
    static std::mt19937 gen(std::random_device{}());

    if (miss_percent + out_of_range_percent > 0 && !reserve_keys_for_misses) {
        // cannot generate misses without reserving keys
        return false;
    }

    size_t num_miss_keys_to_probe = num_probe_keys * miss_percent / 100;
    size_t num_out_of_range_keys_to_probe = num_probe_keys * out_of_range_percent / 100;
    size_t num_reserved_keys = reserve_keys_for_misses ? 40u : 0u;

    std::vector<key_type> generated_keys;
    bool success;
    if (uniform_keys) {
        success = generate_uniform_build_set(gen, num_build_keys, key_stride, key_offset, reserve_keys_for_misses, generated_keys);
    } else {
        success = generate_dense_build_set(gen, num_build_keys, key_stride, key_offset, reserve_keys_for_misses, generated_keys);
    }
    if (!success) return false;

    // copy the first part to use for building the index
    size_t num_remaining_keys = num_build_keys - num_reserved_keys - 2;
    build_keys.resize(num_remaining_keys * key_replication);
    build_values.resize(build_keys.size());
    for (size_t i = 0; i < num_remaining_keys; ++i) {
        for (size_t repl = 0; repl < key_replication; ++repl) {
            // the index is chosen such that identical keys never end up next to each other
            build_keys[num_remaining_keys * repl + i] = generated_keys[i];
            build_values[num_remaining_keys * repl + i] = value_for(generated_keys[i]);
        }
    }
    // reserve the remaining keys to simulate misses
    std::vector<key_type> reserved_keys(num_reserved_keys);
    for (size_t i = 0; i < reserved_keys.size(); ++i) {
        reserved_keys[i] = generated_keys[i + num_remaining_keys];
    }
    // these are the out-of-range misses
    key_type smallest_value = generated_keys[num_build_keys - 2];
    key_type largest_value = generated_keys[num_build_keys - 1];

    probe_keys.resize(num_probe_keys);
    expected_values.resize(num_probe_keys);
    // fill first part with missed keys
    for (size_t i = 0; i < num_miss_keys_to_probe; ++i) {
        std::uniform_int_distribution<size_t> dist(0, reserved_keys.size() - 1);
        size_t random_index = dist(gen);
        probe_keys[i] = reserved_keys[random_index];
        expected_values[i] = NOT_FOUND;
    }
    // fill second part with out-of-range keys
    for (size_t i = 0; i < num_out_of_range_keys_to_probe; ++i) {
        size_t offset = num_miss_keys_to_probe;
        probe_keys[offset + i] = i & 1 ? smallest_value : largest_value;
        expected_values[offset + i] = NOT_FOUND;
    }
    // fill last part with existing keys
    for (size_t i = 0; i < num_probe_keys - num_miss_keys_to_probe - num_out_of_range_keys_to_probe; ++i) {
        std::uniform_int_distribution<size_t> dist(0, build_keys.size() - 1);
        size_t random_index = dist(gen);
        size_t offset = num_miss_keys_to_probe + num_out_of_range_keys_to_probe;
        probe_keys[offset + i] = build_keys[random_index];
        // each value will occur multiple times when key replication is active
        expected_values[offset + i] = key_replication * value_for(build_keys[random_index]);
    }
    // shuffle the entire probe set
    for (size_t i = 0; i < num_probe_keys; ++i) {
        std::uniform_int_distribution<size_t> dis(0, i);
        size_t j = dis(gen);
        std::swap(probe_keys[i], probe_keys[j]);
        std::swap(expected_values[i], expected_values[j]);
    }

    return true;
}


bool generate_range_query_input(
    size_t num_build_keys,
    size_t num_probe_ranges,
    size_t key_stride,
    size_t key_offset,
    size_t range_query_hit_count,
    std::vector<key_type>& build_keys,
    std::vector<value_type>& build_values,
    std::vector<key_type>& probe_lower,
    std::vector<key_type>& probe_upper,
    std::vector<value_type>& expected_values
) {
    static std::mt19937 gen(std::random_device{}());

    build_keys.resize(num_build_keys);
    build_values.resize(build_keys.size());
    // generate dense keys
    for (size_t i = 0; i < num_build_keys; ++i) {
        size_t new_key = i * key_stride + key_offset;
        if (new_key > max_key) return false;
        build_keys[i] = key_type(new_key);
    }
    // shuffle keys
    std::shuffle(build_keys.begin(), build_keys.end(), gen);
    // generate values
    std::map<key_type, value_type> simulated_tree;
    for (size_t i = 0; i < build_keys.size(); ++i) {
        build_values[i] = value_type(build_keys[i]) << 1u;
        simulated_tree.emplace(build_keys[i], build_values[i]);
    }

    // not enough keys to meet range size requirement
    if (num_build_keys < range_query_hit_count) return false;
    size_t largest_possible_range_start = num_build_keys - range_query_hit_count;

    probe_lower.resize(num_probe_ranges);
    probe_upper.resize(num_probe_ranges);
    expected_values.resize(num_probe_ranges);
    // draw ranges uniformly
    for (size_t i = 0; i < num_probe_ranges; ++i) {
        std::uniform_int_distribution<size_t> dist(0, largest_possible_range_start - 1);

        probe_lower[i] = key_offset + dist(gen);
        probe_upper[i] = probe_lower[i] + key_stride * range_query_hit_count - 1;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < num_probe_ranges; ++i) {
        auto lower = simulated_tree.lower_bound(probe_lower[i]);
        auto upper = simulated_tree.upper_bound(probe_upper[i]);

        // pre-compute checksum
        value_type expected = 0;
        for (auto it = lower; it != upper; ++it) {
            expected += it->second;
        }
        expected_values[i] = expected;
    }

    // shuffle probes
    for (size_t i = 0; i < num_probe_ranges; ++i) {
        std::uniform_int_distribution<size_t> dis(0, i);
        size_t j = dis(gen);
        std::swap(probe_lower[i], probe_lower[j]);
        std::swap(probe_upper[i], probe_upper[j]);
        std::swap(expected_values[i], expected_values[j]);
    }

    return true;
}


void generate_updates(
    size_t num_updates_log,
    size_t local_update_chunk_size_log,
    std::vector<key_type>& build_keys,
    std::vector<value_type>& build_values
) {
    static std::mt19937 gen(std::random_device{}());
    size_t num_updates = size_t{1} << num_updates_log;
    size_t local_update_chunk_size = size_t{1} << local_update_chunk_size_log;

#if UPDATE_TYPE == 0
    // global updates: shuffle a subset

    std::vector<size_t> indices(build_keys.size());
    // generate an index permutation so that the fisher-yates shuffle operates on a random subset
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    // perform fisher-yates shuffle on a random subset of the keys
    for (size_t i = 0; i < num_updates - 1; ++i) {
        // prevent fixed points by excluding i=j
        std::uniform_int_distribution<size_t> dist(i + 1, num_updates - 1);
        size_t j = dist(gen);
        std::swap(build_keys[indices[i]], build_keys[indices[j]]);
        std::swap(build_values[indices[i]], build_values[indices[j]]);
    }

#elif UPDATE_TYPE == 1
    // position-local updates: randomly pick and reverse sub-ranges of the build key array

    // pick update locations without replacement
    size_t num_possible_offsets = build_keys.size() / local_update_chunk_size;
    std::vector<size_t> offsets(num_possible_offsets);
    std::iota(offsets.begin(), offsets.end(), 0);
    std::shuffle(offsets.begin(), offsets.end(), gen);

    size_t num_local_updates_to_perform = num_updates / local_update_chunk_size;
    // locally reverse a sub-range of the keys (by index)
    for (size_t up = 0; up < num_local_updates_to_perform; ++up) {
        size_t index_offset = offsets[up] * local_update_chunk_size;
        size_t half_chunk_size = local_update_chunk_size >> 1u;
        for (size_t i = 0; i < half_chunk_size; ++i) {
            size_t swap_i = i + index_offset;
            size_t swap_j = local_update_chunk_size - 1 - i + index_offset;
            std::swap(build_keys[swap_i], build_keys[swap_j]);
            std::swap(build_values[swap_i], build_values[swap_j]);
        }
    }

#elif UPDATE_TYPE == 2
    // value-local updates: randomly pick and reverse sub-ranges of the SORTED build key array

    // pick update locations without replacement
    size_t num_possible_offsets = build_keys.size() / local_update_chunk_size;
    std::vector<size_t> offsets(num_possible_offsets);
    std::iota(offsets.begin(), offsets.end(), 0);
    std::shuffle(offsets.begin(), offsets.end(), gen);

    // generate a sort permutation, i.e., the smallest element in build_keys is at build_keys[perm[0]]
    auto perm = sort_permutation(build_keys);

    size_t num_local_updates_to_perform = num_updates / local_update_chunk_size;
    // locally reverse a sub-range of the keys (by rank)
    for (size_t up = 0; up < num_local_updates_to_perform; ++up) {
        size_t rank_offset = offsets[up] * local_update_chunk_size;
        size_t half_chunk_size = local_update_chunk_size >> 1u;
        for (size_t i = 0; i < half_chunk_size; ++i) {
            size_t swap_i = perm[i + rank_offset];
            size_t swap_j = perm[local_update_chunk_size - 1 - i + rank_offset];
            std::swap(build_keys[swap_i], build_keys[swap_j]);
            std::swap(build_values[swap_i], build_values[swap_j]);
        }
    }

#else
#error illegal update type
#endif
}


#define STR(x) #x
#define STRING(s) STR(s)


void benchmark() {
    std::cout << std::setprecision(20);

    constexpr bool debug = false;
    optix_wrapper optix(debug);
    optix_pipeline pipeline(&optix);

    constexpr size_t key_offset = 0;
    constexpr size_t key_stride_log = KEY_STRIDE_LOG;
    constexpr size_t key_stride = size_t{1} << key_stride_log;
    constexpr size_t miss_percentage = MISS_PERCENTAGE;
    constexpr size_t out_of_range_percentage = OUT_OF_RANGE_PERCENTAGE;
    constexpr size_t range_query_hit_count_log = RANGE_QUERY_HIT_COUNT_LOG;
    constexpr size_t num_updates_log = NUM_UPDATES_LOG;
    constexpr size_t num_build_keys_log = NUM_BUILD_KEYS_LOG;
    constexpr size_t num_probe_keys_log = NUM_PROBE_KEYS_LOG;
    constexpr size_t num_rays_per_thread_log = NUM_RAYS_PER_THREAD_LOG;
    constexpr size_t num_rays_per_thread = size_t{1} << num_rays_per_thread_log;
    constexpr size_t key_replication_log = KEY_REPLICATION_LOG;
    constexpr size_t key_replication = size_t{1} << key_replication_log;
    constexpr size_t local_update_chunk_size_log = LOCAL_UPDATE_CHUNK_SIZE_LOG;
    constexpr bool reserve_keys_for_misses = LEAVE_GAPS_FOR_MISSES;
    constexpr bool start_ray_at_zero = START_RAY_AT_ZERO;
    constexpr bool large_keys = LARGE_KEYS;
    constexpr bool closest_hit = USE_CLOSESTHIT_INSTEAD_OF_ANYHIT;
    constexpr bool skip_probing = SKIP_PROBING;
    constexpr bool force_uniform_keys = FORCE_UNIFORM_KEYS;

    do {
        std::cerr << "starting input generation" << std::endl;

        size_t num_build_keys = (size_t{1} << num_build_keys_log) - 1u;
        size_t num_probe_keys = size_t{1} << num_probe_keys_log;

        // ==================================================================
        // generate input and expected output
        // ==================================================================

        std::vector<key_type> build_keys;
        std::vector<value_type> build_values;
        std::vector<key_type> probe_lower;
        std::vector<key_type> probe_upper;
        std::vector<value_type> expected_values;
#if RANGE_QUERY_HIT_COUNT_LOG == 0
        bool possible = generate_input(
            num_build_keys,
            num_probe_keys,
            key_stride,
            key_offset,
            reserve_keys_for_misses,
            miss_percentage,
            out_of_range_percentage,
            key_replication,
            force_uniform_keys,
            build_keys,
            build_values,
            probe_lower,
            expected_values
            );
#else
        bool possible = generate_range_query_input(
            num_build_keys,
            num_probe_keys,
            key_stride,
            key_offset,
            size_t{1} << range_query_hit_count_log,
            build_keys,
            build_values,
            probe_lower,
            probe_upper,
            expected_values
            );
#endif
        if (!possible) continue;

        std::cerr << "generated input" << std::endl;

        {
#if INSERT_SORTED == 1
            auto permutation = sort_permutation(build_keys, std::less<key_type>());
#elif INSERT_SORTED == -1
            auto permutation = sort_permutation(build_keys, std::greater<key_type>());
#endif
#if INSERT_SORTED == 1 || INSERT_SORTED == -1
            apply_permutation(build_keys, permutation);
            apply_permutation(build_values, permutation);
#endif
        }
        {
#if PROBE_SORTED == 1 || PROBE_SORTED == 2
            auto permutation = sort_permutation(probe_lower, std::less<key_type>());
#elif PROBE_SORTED == -1
            auto permutation = sort_permutation(probe_lower, std::greater<key_type>());
#endif
#if PROBE_SORTED == 1 || PROBE_SORTED == -1
            apply_permutation(probe_lower, permutation);
#if RANGE_QUERY_HIT_COUNT_LOG != 0
            apply_permutation(probe_upper, permutation);
#endif
            apply_permutation(expected_values, permutation);
#endif
#if PROBE_SORTED == 2
            apply_permutation(expected_values, permutation);
#endif
        }

        std::cerr << "ordered input" << std::endl;

        cuda_buffer build_keys_buffer_d, build_values_buffer_d;
        cuda_buffer probe_lower_buffer_d, probe_upper_buffer_d, result_buffer_d;
        cuda_buffer data_structure_d;
        cuda_buffer launch_params_d;

        // ==================================================================
        // set launch parameters
        // ==================================================================

        build_keys_buffer_d.alloc_and_upload(build_keys);
        build_values_buffer_d.alloc_and_upload(build_values);
        probe_lower_buffer_d.alloc_and_upload(probe_lower);
#if RANGE_QUERY_HIT_COUNT_LOG != 0
        probe_upper_buffer_d.alloc_and_upload(probe_upper);
#endif
        result_buffer_d.alloc(num_probe_keys * sizeof(value_type));

        cudaDeviceSynchronize(); CUERR

        launch_parameters launch_params;

        size_t uncompacted_size = 0;
        size_t final_size = 0;
        size_t update_temp_buffer_size = 0;
        double convert_time_ms = 0;
        double build_time_ms = 0;
        double compact_time_ms = 0;
        double update_time_ms = 0;
        double update_convert_time_ms = 0;

        launch_params.traversable = build_traversable(
            optix, build_keys_buffer_d.ptr<key_type>(), build_keys.size(), data_structure_d,
            uncompacted_size, final_size, convert_time_ms, build_time_ms, compact_time_ms);

        std::cerr << "built structure" << std::endl;

        launch_params.build_keys = build_keys_buffer_d.ptr<key_type>();
        launch_params.build_values = build_values_buffer_d.ptr<value_type>();
        launch_params.query_lower = probe_lower_buffer_d.ptr<key_type>();
#if RANGE_QUERY_HIT_COUNT_LOG != 0
        launch_params.query_upper = probe_upper_buffer_d.ptr<key_type>();
#else
        launch_params.query_upper = nullptr;
#endif
        launch_params.result = result_buffer_d.ptr<value_type>();
        launch_params_d.alloc(sizeof(launch_params));
        launch_params_d.upload(&launch_params, 1);

        cudaDeviceSynchronize(); CUERR

        std::cerr << "uploaded launch parameters" << std::endl;

        // ==================================================================
        // update structure
        // ==================================================================

        if (num_updates_log > 0) {
            generate_updates(num_updates_log, local_update_chunk_size_log, build_keys, build_values);
            build_keys_buffer_d.upload(build_keys.data(), build_keys.size());
            build_values_buffer_d.upload(build_values.data(), build_values.size());
            std::cerr << "generated 2^" << num_updates_log << " updates" << std::endl;
            update_traversable(
                optix, build_keys_buffer_d.ptr<key_type>(), build_keys.size(), data_structure_d,
                launch_params.traversable, update_temp_buffer_size, update_convert_time_ms, update_time_ms);
            std::cerr << "updated data structure" << std::endl;
        }

        // ==================================================================
        // sort probes
        // ==================================================================

        double sort_time_ms = 0;
#if RANGE_QUERY_HIT_COUNT_LOG == 0 && PROBE_SORTED == 2
        {
            cuda_buffer temp_d, dest_d;
            size_t temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes,
                probe_lower_buffer_d.ptr<key_type>(), dest_d.ptr<key_type>(),
                probe_lower.size(), 0, sizeof(key_type)*8, optix.stream);
            temp_d.alloc(temp_storage_bytes);
            dest_d.alloc(sizeof(key_type) * probe_lower.size());

            cudaEvent_t sort_start, sort_stop;
            float sort_delta;
            cudaEventCreate(&sort_start);
            cudaEventCreate(&sort_stop);
            cudaEventRecord(sort_start, optix.stream);
            cub::DeviceRadixSort::SortKeys(temp_d.raw_ptr, temp_storage_bytes,
                probe_lower_buffer_d.ptr<key_type>(), dest_d.ptr<key_type>(),
                probe_lower.size(), 0, sizeof(key_type)*8, optix.stream);
            cudaMemcpyAsync(probe_lower_buffer_d.raw_ptr, dest_d.raw_ptr, sizeof(key_type) * probe_lower.size(), D2D, optix.stream);
            cudaEventRecord(sort_stop, optix.stream);
            cudaEventSynchronize(sort_stop);
            cudaEventElapsedTime(&sort_delta, sort_start, sort_stop);

            sort_time_ms = sort_delta;
            std::cerr << "sort: " << sort_time_ms << "ms" << std::endl;
        }
#endif
#if RANGE_QUERY_HIT_COUNT_LOG != 0 && PROBE_SORTED == 2
        {
            cuda_buffer temp_d, lower_dest_d, upper_dest_d;
            size_t temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                probe_lower_buffer_d.ptr<key_type>(), lower_dest_d.ptr<key_type>(),
                probe_upper_buffer_d.ptr<key_type>(), upper_dest_d.ptr<key_type>(),
                probe_lower.size(), 0, sizeof(key_type)*8, optix.stream);
            temp_d.alloc(temp_storage_bytes);
            lower_dest_d.alloc(sizeof(key_type) * probe_lower.size());
            upper_dest_d.alloc(sizeof(key_type) * probe_lower.size());

            cudaEvent_t sort_start, sort_stop;
            float sort_delta;
            cudaEventCreate(&sort_start);
            cudaEventCreate(&sort_stop);
            cudaEventRecord(sort_start, optix.stream);
            cub::DeviceRadixSort::SortPairs(temp_d.raw_ptr, temp_storage_bytes,
                probe_lower_buffer_d.ptr<key_type>(), lower_dest_d.ptr<key_type>(),
                probe_upper_buffer_d.ptr<key_type>(), upper_dest_d.ptr<key_type>(),
                probe_lower.size(), 0, sizeof(key_type)*8, optix.stream);
            cudaMemcpyAsync(probe_lower_buffer_d.raw_ptr, lower_dest_d.raw_ptr, sizeof(key_type) * probe_lower.size(), D2D, optix.stream);
            cudaMemcpyAsync(probe_upper_buffer_d.raw_ptr, upper_dest_d.raw_ptr, sizeof(key_type) * probe_lower.size(), D2D, optix.stream);
            cudaEventRecord(sort_stop, optix.stream);
            cudaEventSynchronize(sort_stop);
            cudaEventElapsedTime(&sort_delta, sort_start, sort_stop);

            sort_time_ms = sort_delta;
            std::cerr << "sort with values: " << sort_time_ms << "ms" << std::endl;
        }
#endif

#if SKIP_PROBING == 0
        // ==================================================================
        // launch
        // ==================================================================

        const uint32_t num_rays = probe_lower.size();
        const uint32_t num_threads = num_rays / num_rays_per_thread;
        if (num_threads * num_rays_per_thread != num_rays) {
            std::cerr << "inexact division when assigning rays to threads" << std::endl;
            std::exit(1);
        }

        {
            nvtx3::scoped_range nvtx_range{"warmup-run"};

            OPTIX_CHECK(optixLaunch(
                    pipeline.pipeline,
                    optix.stream,
                    launch_params_d.cu_ptr(),
                    launch_params_d.size_in_bytes,
                    &pipeline.sbt,
                    num_threads,
                    1,
                    1
            ))

            cudaDeviceSynchronize(); CUERR
        }

        std::cerr << "warmup successful" << std::endl;

        // ==================================================================
        // output
        // ==================================================================

        std::vector<value_type> output(expected_values.size());
        result_buffer_d.download(output.data(), expected_values.size());
        for (size_t i = 0; i < expected_values.size(); ++i) {
            if (output[i] != expected_values[i]) {
#if RANGE_QUERY_HIT_COUNT_LOG == 0
                std::cerr << i << ": for key " << probe_lower[i] << " expected " << expected_values[i] << " != actual " << output[i] << std::endl;
#else
                std::cerr << i << ": for range " << probe_lower[i] << "-" << probe_upper[i] << " expected " << expected_values[i] << " != actual " << output[i] << std::endl;
#endif
                std::exit(1);
            }
            //std::cout << i << " " << output[i] << std::endl;
        }
        std::cerr << "no errors detected" << std::endl;

        // ==================================================================
        // timing
        // ==================================================================

        size_t runs = 10;
        double accumulated_runtime_ms = 0;

        for (size_t i = 0; i < runs; ++i) {
            nvtx3::scoped_range nvtx_range{"run-" + std::to_string(i)};

            cudaEvent_t timerstart, timerstop;
            float timerdelta;
            cudaEventCreate(&timerstart);
            cudaEventCreate(&timerstop);
            cudaEventRecord(timerstart, optix.stream);

            OPTIX_CHECK(optixLaunch(
                    pipeline.pipeline,
                    optix.stream,
                    launch_params_d.cu_ptr(),
                    launch_params_d.size_in_bytes,
                    &pipeline.sbt,
                    num_threads,
                    1,
                    1
            ))

            cudaEventRecord(timerstop, optix.stream);
            cudaEventSynchronize(timerstop);
            cudaEventElapsedTime(&timerdelta, timerstart, timerstop);
            accumulated_runtime_ms += timerdelta;
        }

        cudaDeviceSynchronize(); CUERR
        double total_probe_time_ms = accumulated_runtime_ms / runs + sort_time_ms;
#else
        double total_probe_time_ms = 0;
#endif
        double convert_build_time_ms = build_time_ms + convert_time_ms;
        double convert_build_compact_time_ms = convert_build_time_ms + compact_time_ms;
        double convert_and_update_time_ms = update_convert_time_ms + update_time_ms;

        bool perpendicular_rays = PERPENDICULAR_RAYS;
        bool force_single_anyhit = FORCE_SINGLE_ANYHIT;
        bool compaction = COMPACTION;
        int64_t exponent_bias = EXPONENT_BIAS;

#if PRIMITIVE == 0
        std::cout << "i_primitive=triangle,";
#elif PRIMITIVE == 1
        std::cout << "i_primitive=sphere,";
#elif PRIMITIVE == 2
        std::cout << "i_primitive=aabb,";
#endif
#if INT_TO_FLOAT_CONVERSION_MODE == 3
        std::cout << "i_key_mode=3d,";
#elif INT_TO_FLOAT_CONVERSION_MODE == 2
        std::cout << "i_key_mode=ext,";
#elif INT_TO_FLOAT_CONVERSION_MODE == 1
        std::cout << "i_key_mode=excl,";
#else
        std::cout << "i_key_mode=safe,";
#endif
#if INSERT_SORTED == 1
        std::cout << "i_build_mode=b_asc,";
#elif INSERT_SORTED == -1
        std::cout << "i_build_mode=b_dsc,";
#else
        std::cout << "i_build_mode=b_sfl,";
#endif
#if PROBE_SORTED == 2
        std::cout << "i_probe_mode=p_cubsort,";
#elif PROBE_SORTED == 1
        std::cout << "i_probe_mode=p_asc,";
#elif PROBE_SORTED == -1
        std::cout << "i_probe_mode=p_dsc,";
#else
        std::cout << "i_probe_mode=p_sfl,";
#endif
#if UPDATE_TYPE == 2
        std::cout << "i_update_type=rank_local,";
#elif UPDATE_TYPE == 1
        std::cout << "i_update_type=pos_local,";
#else
        std::cout << "i_update_type=global,";
#endif
        std::cout << IVAR(num_build_keys_log) << ",";
        std::cout << IVAR(key_offset) << ",";
        std::cout << IVAR(key_stride_log) << ",";
        std::cout << IVAR(key_replication_log) << ",";
        std::cout << IVAR(reserve_keys_for_misses) << ",";
        std::cout << IVAR(num_updates_log) << ",";
        std::cout << IVAR(local_update_chunk_size_log) << ",";
        std::cout << IVAR(num_probe_keys_log) << ",";
        std::cout << IVAR(miss_percentage) << ",";
        std::cout << IVAR(out_of_range_percentage) << ",";
        std::cout << IVAR(range_query_hit_count_log) << ",";
        std::cout << IVAR(exponent_bias) << ",";
        std::cout << IVAR(force_single_anyhit) << ",";
        std::cout << IVAR(perpendicular_rays) << ",";
        std::cout << IVAR(start_ray_at_zero) << ",";
        std::cout << IVAR(compaction) << ",";
        std::cout << IVAR(large_keys) << ",";
        std::cout << IVAR(closest_hit) << ",";
        std::cout << IVAR(skip_probing) << ",";
        std::cout << IVAR(num_rays_per_thread_log) << ",";
        std::cout << IVAR(force_uniform_keys) << ",";
        std::cout << IVAR(x_bits) << ",";
        std::cout << IVAR(y_bits) << ",";
        std::cout << IVAR(z_bits) << ",";
        std::cout << IVAR(debug) << ",";

        std::cout << OVAR(uncompacted_size) << ",";
        std::cout << OVAR(final_size) << ",";
        std::cout << OVAR(update_temp_buffer_size) << ",";
        std::cout << OVAR(convert_time_ms) << ",";
        std::cout << OVAR(build_time_ms) << ",";
        std::cout << OVAR(convert_build_time_ms) << ",";
        std::cout << OVAR(convert_build_compact_time_ms) << ",";
        std::cout << OVAR(compact_time_ms) << ",";
        std::cout << OVAR(update_time_ms) << ",";
        std::cout << OVAR(convert_and_update_time_ms) << ",";
        std::cout << OVAR(sort_time_ms) << ",";
        std::cout << OVAR(total_probe_time_ms) << std::endl;

    } while (0);
}


int load_mtx_graph_parallel(
  const std::string& filename,
  std::vector<uint32_t>& row_ptr,
  std::vector<uint32_t>& nbrs,
  std::vector<float>& wts,
  int num_threads = std::thread::hardware_concurrency())
{
  std::cout << "Loading Matrix Market graph from " << filename
            << " using " << num_threads << " threads...\n";

  auto t_start = std::chrono::high_resolution_clock::now();

  std::ifstream infile(filename);
  if (!infile.is_open()) {
      std::cerr << "Error: Could not open graph file " << filename << "\n";
      exit(1);
  }

  std::string line;
  bool is_symmetric = false;

  // --- Step 1: Read banner and header ---
  while (std::getline(infile, line)) {
      if (line.empty() || line[0] != '%') break;
      if (line.find("symmetric") != std::string::npos) {
          is_symmetric = true;
          std::cout << "  [load] Detected symmetric format.\n";
      }
  }

  uint32_t header_rows = 0, header_cols = 0, header_edges = 0;
  {
      std::stringstream ss(line);
      ss >> header_rows >> header_cols >> header_edges;
  }

  std::cout << "  [load] Header: " << header_rows << " x " << header_cols
            << ", " << header_edges << " edges.\n";

  // --- Step 2: Read remaining lines into memory ---
  std::vector<std::string> lines;
  lines.reserve(header_edges * 1.1);
  std::string buf;
  while (std::getline(infile, buf)) {
      if (!buf.empty() && buf[0] != '%') lines.push_back(std::move(buf));
  }
  infile.close();

  const size_t M = lines.size();
  std::cout << "  [load] Read " << M << " edges.\n";

  // --- Step 3: Parse lines in parallel into (u,v,w) ---
  std::vector<std::vector<std::tuple<uint32_t, uint32_t, float>>> local_edges(num_threads);

  auto parse_chunk = [&](int tid, size_t start, size_t end) {
      auto& edges = local_edges[tid];
      edges.reserve(end - start);
      for (size_t i = start; i < end; ++i) {
          std::stringstream ss(lines[i]);
          uint32_t u, v; float w = 1.0f;
          ss >> u >> v >> w;
          edges.emplace_back(u, v, w);
      }
  };

  std::vector<std::thread> threads;
  size_t chunk = (M + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
      size_t start = t * chunk;
      size_t end   = std::min(M, start + chunk);
      if (start >= M) break;
      threads.emplace_back(parse_chunk, t, start, end);
  }
  for (auto& th : threads) th.join();

  // --- Step 4: Build global ID map from all edges ---
  std::unordered_map<uint32_t, uint32_t> id_map;
  id_map.reserve(header_rows * 2);
  uint32_t next_id = 0;
  for (const auto& edges : local_edges) {
      for (const auto& [u, v, _] : edges) {
          if (id_map.find(u) == id_map.end()) id_map[u] = next_id++;
          if (id_map.find(v) == id_map.end()) id_map[v] = next_id++;
      }
  }

  const uint32_t n = next_id;
  std::cout << "  [load] Found " << n << " unique nodes.\n";

  // --- Step 5: Remap edges to dense IDs and build adjacency lists in parallel ---
  std::vector<std::vector<std::pair<uint32_t, float>>> adj(n);
  std::mutex adj_mutex; // protect shared adjacency list writes

  auto remap_chunk = [&](int tid) {
      const auto& edges = local_edges[tid];
      std::vector<std::pair<uint32_t, std::pair<uint32_t, float>>> local;
      local.reserve(edges.size() * (is_symmetric ? 2 : 1));

      for (auto& [u, v, w] : edges) {
          uint32_t u_new = id_map.at(u);
          uint32_t v_new = id_map.at(v);
          local.emplace_back(u_new, std::make_pair(v_new, w));
          if (is_symmetric && u_new != v_new)
              local.emplace_back(v_new, std::make_pair(u_new, w));
      }

      // Append to global adjacency with minimal locking
      {
          std::lock_guard<std::mutex> lock(adj_mutex);
          for (auto& e : local)
              adj[e.first].push_back(e.second);
      }
  };

  threads.clear();
  for (int t = 0; t < num_threads; ++t)
      threads.emplace_back(remap_chunk, t);
  for (auto& th : threads) th.join();

  // --- Step 6: Sort and deduplicate adjacency lists ---
  std::cout << "  [load] Sorting & deduplicating adjacency lists...\n";

  auto sort_chunk = [&](int tid, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
          auto& vec = adj[i];
          std::sort(vec.begin(), vec.end(),
                    [](auto& a, auto& b){ return a.first < b.first; });
          vec.erase(std::unique(vec.begin(), vec.end(),
                                [](auto& a, auto& b){ return a.first == b.first; }),
                    vec.end());
      }
  };

  threads.clear();
  chunk = (n + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
      size_t start = t * chunk;
      size_t end   = std::min<size_t>(n, start + chunk);
      if (start >= n) break;
      threads.emplace_back(sort_chunk, t, start, end);
  }
  for (auto& th : threads) th.join();

  // --- Step 7: Convert to CSR ---
  std::cout << "  [load] Converting to CSR format...\n";
  row_ptr.clear(); nbrs.clear(); wts.clear();
  row_ptr.reserve(n + 1);
  row_ptr.push_back(0);

  size_t total_edges = 0;
  for (uint32_t i = 0; i < n; ++i) {
      for (auto& e : adj[i]) {
          nbrs.push_back(e.first);
          wts.push_back(e.second);
      }
      total_edges += adj[i].size();
      row_ptr.push_back(static_cast<uint32_t>(nbrs.size()));
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  std::cout << "\n--- Graph Loading Complete ---\n";
  std::cout << "Nodes: " << n << "\n";
  std::cout << "Edges: " << total_edges << "\n";
  std::cout << "Threads: " << num_threads << "\n";
  std::cout << "Elapsed: " << elapsed << " s\n";
  std::cout << "-------------------------------\n\n";

  return n;
}

#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <iomanip>

// assumes you already included everything from your OptiX/BVH file
// and pasted load_mtx_graph_parallel() above this main()

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " graph.mtx [num_threads]" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int num_threads = (argc >= 3) ? std::stoi(argv[2]) : std::thread::hardware_concurrency();

    std::vector<uint32_t> row_ptr, nbrs;
    std::vector<float> wts;

    std::cout << "=============================================\n";
    std::cout << " BVH Builder for Graph (.mtx)\n";
    std::cout << "=============================================\n";

    auto t_start_total = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------
    // Step 1: Load graph (parallel)
    // ------------------------------------------------------------
    auto t_start_load = std::chrono::high_resolution_clock::now();
    int n_nodes = load_mtx_graph_parallel(filename, row_ptr, nbrs, wts, num_threads);
    auto t_end_load = std::chrono::high_resolution_clock::now();
    double load_time_s = std::chrono::duration<double>(t_end_load - t_start_load).count();

    // ------------------------------------------------------------
    // Step 2: Convert CSR edges to AABBs (for BVH)
    // Each edge becomes a small AABB in 3D space
    // ------------------------------------------------------------
    auto t_start_convert = std::chrono::high_resolution_clock::now();

    const float eps = 1e-3f;
    size_t num_edges = nbrs.size();
    std::vector<float3> aabb_pairs(2 * num_edges);

    for (size_t src = 0; src < n_nodes; ++src) {
        for (size_t idx = row_ptr[src]; idx < row_ptr[src + 1]; ++idx) {
            uint32_t dst = nbrs[idx];
            float w = wts[idx];

            float x = static_cast<float>(src) / static_cast<float>(n_nodes);
            float y = static_cast<float>(dst) / static_cast<float>(n_nodes);
            float z = w;

            aabb_pairs[2 * idx + 0] = make_float3(x - eps, y - eps, z - eps);
            aabb_pairs[2 * idx + 1] = make_float3(x + eps, y + eps, z + eps);
        }
    }

    auto t_end_convert = std::chrono::high_resolution_clock::now();
    double convert_time_ms =
        std::chrono::duration<double, std::milli>(t_end_convert - t_start_convert).count();

    std::cout << "Converted " << num_edges << " edges to AABBs in "
              << convert_time_ms << " ms.\n";

    // ------------------------------------------------------------
    // Step 3: Upload and build BVH
    // ------------------------------------------------------------
    optix_wrapper optix(false);
    cuda_buffer aabb_buffer;
    cuda_buffer as_buffer;
    aabb_buffer.alloc_and_upload(aabb_pairs);

    size_t uncompacted_size = 0;
    size_t final_size = 0;
    double build_time_ms = 0;
    double compact_time_ms = 0;
    double dummy_convert_time = 0;

    auto t_start_bvh = std::chrono::high_resolution_clock::now();

    OptixTraversableHandle handle = build_traversable(
        optix,
        reinterpret_cast<const key_type*>(aabb_buffer.ptr<float3>()),
        num_edges,
        as_buffer,
        uncompacted_size,
        final_size,
        dummy_convert_time,
        build_time_ms,
        compact_time_ms
    );

    cudaDeviceSynchronize(); CUERR;
    auto t_end_bvh = std::chrono::high_resolution_clock::now();
    double total_bvh_time_ms =
        std::chrono::duration<double, std::milli>(t_end_bvh - t_start_bvh).count();

// ------------------------------------------------------------
// Step 4: Print summary (with dynamic primitive info)
// ------------------------------------------------------------
auto t_end_total = std::chrono::high_resolution_clock::now();
double total_time_s = std::chrono::duration<double>(t_end_total - t_start_total).count();

// Compute primitive statistics dynamically
std::string primitive_type;
size_t num_primitives = 0;
size_t primitive_bytes = 0;

#if PRIMITIVE == 0
    primitive_type = "Triangles";
    num_primitives = num_edges; // each edge/record corresponds to one triangle
    primitive_bytes = 3 * num_primitives * sizeof(float3);
#elif PRIMITIVE == 1
    primitive_type = "Spheres";
    num_primitives = num_edges; // or key_count equivalent
    primitive_bytes = num_primitives * sizeof(float3) + num_primitives * sizeof(float); // center + radius
#elif PRIMITIVE == 2
    primitive_type = "AABBs";
    num_primitives = num_edges; // each edge -> one AABB
    primitive_bytes = 2 * num_primitives * sizeof(float3); // min + max
#else
    primitive_type = "Unknown";
#endif

double primitive_size_mb = primitive_bytes / (1024.0 * 1024.0);

std::cout << "\n=========== BVH BUILD SUMMARY ===========\n";
std::cout << "File:              " << filename << "\n";
std::cout << "Nodes:             " << n_nodes << "\n";
std::cout << "Edges:             " << num_edges << "\n";
std::cout << "Threads:           " << num_threads << "\n";
std::cout << "----------------------------------------\n";
std::cout << std::fixed << std::setprecision(3);
std::cout << "Graph Load:        " << load_time_s * 1000.0 << " ms\n";
std::cout << "Primitive Convert: " << convert_time_ms << " ms\n";
std::cout << "BVH Build:         " << build_time_ms << " ms\n";
std::cout << "Compaction:        " << compact_time_ms << " ms\n";
std::cout << "BVH Total Time:    " << total_bvh_time_ms << " ms\n";
std::cout << "----------------------------------------\n";
std::cout << "Primitive Type:    " << primitive_type << "\n";
std::cout << "Primitives:        " << num_primitives << "\n";
std::cout << "Primitive Memory:  " << primitive_bytes << " bytes ("
          << primitive_size_mb << " MB)\n";
std::cout << "----------------------------------------\n";
std::cout << "Final BVH Size:    " << final_size / (1024.0 * 1024.0) << " MB\n";
std::cout << "Total Pipeline:    " << total_time_s << " s\n";
std::cout << "========================================\n";


    return 0;
}


/*
int main() {
    benchmark();
}*/
