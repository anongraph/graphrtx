#ifndef TC_ALGORITHM
#define TC_ALGORITHM
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>

#include "memory/gpu_manager.hpp"

uint64_t run_triangle_counting_optix(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    const std::vector<uint32_t>& row_ptr,
    const std::vector<uint32_t>& nbrs,
    const std::vector<uint32_t>& uasp_first,
    const std::vector<uint32_t>& uasp_count,
    GPUMemoryManager& mm,
    CUstream stream);

void run_triangle_counting_bench(
  OptixPipeline pipe,
  const OptixShaderBindingTable& sbt,
  CUdeviceptr d_params,
  const Params& baseParams,
  int num_vertices,
  const std::vector<uint32_t>& row_ptr,
  const std::vector<uint32_t>& nbrs,
  const std::vector<uint32_t>& uasp_first,
  const std::vector<uint32_t>& uasp_count,
  GPUMemoryManager& mm,
  CUstream stream);

void run_triangle_counting_hybrid_safe(
  OptixPipeline pipe,
  const OptixShaderBindingTable& sbt,
  CUdeviceptr d_params,
  const Params& baseParams,
  int num_vertices,
  const std::vector<uint32_t>& row_ptr,
  const std::vector<uint32_t>& nbrs,
  const std::vector<uint32_t>& uasp_first,
  const std::vector<uint32_t>& uasp_count,
  GPUMemoryManager& mm,
  CUstream stream, float percent = 0.0f);

#endif