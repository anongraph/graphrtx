#include "partition.hpp"

void prepare_tlas_for_frontier(GPUMemoryManager& mm,
                                      const std::vector<uint32_t>& frontier,
                                      const std::vector<uint32_t>& uasp_first,
                                      OptixTraversableHandle* outTLAS,
                                      CUdeviceptr* outTLASMem,
                                      CUdeviceptr* d_instance_bases,
                                      uint32_t* num_instances,
                                      CUstream stream)
{
  cudaEvent_t start_evt, stop_evt;
  CUDA_CHECK(cudaEventCreate(&start_evt));
  CUDA_CHECK(cudaEventCreate(&stop_evt));
  CUDA_CHECK(cudaEventRecord(start_evt, stream));

  std::unordered_set<uint32_t> need;
  need.reserve(frontier.size()*2+1);
  for (uint32_t v : frontier) {
    uint32_t pid = mm.vertexToPartition(v, uasp_first[v]);
    need.insert(pid);
  }
  std::vector<uint32_t> need_vec(need.begin(), need.end());
  std::sort(need_vec.begin(), need_vec.end());

  mm.ensureResident(need_vec, stream);
  mm.buildTLASDynamic(outTLAS, outTLASMem, d_instance_bases, num_instances, stream);

  CUDA_CHECK(cudaEventRecord(stop_evt, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_evt));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
  mm.bvh_build_ms += ms;
  CUDA_CHECK(cudaEventDestroy(start_evt));
  CUDA_CHECK(cudaEventDestroy(stop_evt));
}