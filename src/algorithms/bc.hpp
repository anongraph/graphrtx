#ifndef BC_ALGORITHM
#define BC_ALGORITHM
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>

#include "memory/gpu_manager.hpp"

std::vector<float> run_betweenness_optix(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream streamOptix);

void run_betweenness_hybrid(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream streamOptix, uint32_t thres = 0);

#endif