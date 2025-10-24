#ifndef BFS_HPP
#define BFS_HPP

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>

#include "memory/gpu_manager.hpp"

std::vector<uint32_t> run_bfs(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    GPUMemoryManager& mm,
    CUdeviceptr d_params,
    const Params& baseParams,
    int source,
    int num_vertices,
    const std::vector<uint32_t>& uasp_first,
    const std::vector<uint32_t>& uasp_count);

void run_bfs_hybrid(OptixPipeline pipe,
  const OptixShaderBindingTable& sbt,
  GPUMemoryManager& mm,
  CUdeviceptr d_params,
  const Params& baseParams,
  int source,
  int num_vertices,
  const std::vector<uint32_t>& /*uasp_first*/,
  const std::vector<uint32_t>& /*uasp_count*/,
  uint32_t thres = 0);

void run_bfs_bench(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    GPUMemoryManager& mm,
    CUdeviceptr d_params,
    const Params& baseParams,
    int source,
    int num_vertices,
    const std::vector<uint32_t>& uasp_first,
    const std::vector<uint32_t>& uasp_count);

#endif