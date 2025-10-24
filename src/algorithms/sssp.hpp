#ifndef SSSP_ALGO_HPP
#define SSSP_ALGO_HPP
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>

#include "memory/gpu_manager.hpp"

std::vector<float> run_sssp(OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int source,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream streamOptix);

void run_sssp_hybrid(OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int source,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream stream,
    uint32_t deg = 0);

#endif