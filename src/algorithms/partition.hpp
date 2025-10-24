#ifndef PARTITION_ALGO_HPP
#define PARTITION_ALGO_HPP

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_set>

#include "../memory/gpu_manager.hpp"

void prepare_tlas_for_frontier(GPUMemoryManager& mm,
    const std::vector<uint32_t>& frontier,
    const std::vector<uint32_t>& uasp_first,
    OptixTraversableHandle* outTLAS,
    CUdeviceptr* outTLASMem,
    CUdeviceptr* d_instance_bases,
    uint32_t* num_instances,
    CUstream stream);

#endif