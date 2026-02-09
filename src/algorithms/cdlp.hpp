#pragma once
#include <vector>
#include <cstdint>

#include <optix.h>
#include <cuda.h>

#include "shared.h"

struct GPUMemoryManager;

std::vector<uint32_t> run_cdlp_optix(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream stream,
    int max_iters = 20);

std::vector<uint32_t> run_cdlp_hybrid(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream stream,
    int max_iters = 20,
    uint32_t thres = 100000u);
