#ifndef PR_ALGO_HPP
#define PR_ALGO_HPP
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>

#include "memory/gpu_manager.hpp"

std::vector<float> run_pagerank_optix(OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    int iters,
    float damping,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream streamOptix);

void run_pagerank_bench(OptixPipeline pipe,
                               const OptixShaderBindingTable& sbt,
                               CUdeviceptr d_params,
                               const Params& baseParams,
                               int num_vertices,
                               int iters,
                               float damping,
                               GPUMemoryManager& mm,
                               const std::vector<uint32_t>& uasp_first,
                               CUstream streamOptix);

void run_pagerank_hybrid(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    int iters,
    float damping,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream stream,
    uint32_t deg = 0);

#endif