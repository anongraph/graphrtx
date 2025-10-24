#include "tc.hpp"
#include <iostream>
#include <chrono>
#include <numeric>
#include <algorithm>

#include "shared.h"
#include "common.hpp"
#include "partition.hpp"
#include "kernels/cuda/algorithms.cuh"


uint64_t run_cuda(const Job* d_jobs, uint32_t num_jobs,
  const uint32_t* d_row, const uint32_t* d_nbr,
  unsigned long long* d_tri, float* ms_out,
  CUstream stream)
{
  uint64_t zero=0; CUDA_CHECK(cudaMemcpyAsync(d_tri,&zero,sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream));
  cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0, stream));
  launch_tc_join_warp_coop(d_jobs, num_jobs, d_row, d_nbr, d_tri);
  CUDA_CHECK(cudaEventRecord(e1, stream));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = elapsedMs(e0,e1);
  if (ms_out) *ms_out = ms;
  CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
  uint64_t tri=0;
  CUDA_CHECK(cudaMemcpyAsync(&tri,d_tri,sizeof(uint64_t),
        cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return tri;
}

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
  CUstream stream)
{
  const int N = num_vertices;

  constexpr uint64_t NODE_STREAM_THRESHOLD   = 1'000'000ULL;
  constexpr double   GPU_FREE_FRACTION_LIMIT = 0.60;

  uint64_t estimated_jobs = 0;
  for (int u = 0; u < N; ++u) {
    const uint32_t deg_u = row_ptr[u+1] - row_ptr[u];
    for (uint32_t k = row_ptr[u]; k < row_ptr[u+1]; ++k) {
      uint32_t v = nbrs[k];
      if (u < (int)v) {
        const uint32_t deg_v = row_ptr[v+1] - row_ptr[v];
        const uint32_t ray_source = (deg_u <= deg_v) ? u : v;
        estimated_jobs += (uint64_t)uasp_count[ray_source];
      }
    }
  }
  
  if (estimated_jobs == 0) {
    mm.bvh_build_ms = 0.0f;
    return 0;
  }

  size_t freeB = 0, totalB = 0;
  cudaMemGetInfo(&freeB, &totalB);

  const uint64_t needed_bytes_monolithic =
      estimated_jobs * (uint64_t)(sizeof(Job) + sizeof(uint32_t));

  const bool too_many_nodes     = (uint64_t)N > NODE_STREAM_THRESHOLD;
  const bool exceeds_free_vram  = (double)needed_bytes_monolithic > GPU_FREE_FRACTION_LIMIT * (double)freeB;
  const bool use_streaming      = too_many_nodes || exceeds_free_vram;

  OptixTraversableHandle tlas = 0;
  CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
  uint32_t num_instances = 0;
  CUdeviceptr d_tlas_mem_streaming = 0, d_instance_bases_streaming = 0;
  bool built_streaming_tlas = false;

  if (mm.hasSingleTLAS()) {
    mm.getSingleTLAS(&tlas, &d_tlas_mem, &d_instance_bases, &num_instances);
  } else {
    std::vector<uint32_t> all_nodes(N);
    std::iota(all_nodes.begin(), all_nodes.end(), 0u);
    prepare_tlas_for_frontier(mm, all_nodes, uasp_first, &tlas, &d_tlas_mem_streaming,
                              &d_instance_bases_streaming, &num_instances, stream);
    d_tlas_mem = d_tlas_mem_streaming;
    d_instance_bases = d_instance_bases_streaming;
    built_streaming_tlas = true;
  }

  Params baseP = baseParams;
  baseP.tlas                = tlas;
  baseP.instance_prim_bases = (const uint32_t*)d_instance_bases;
  baseP.num_instances       = num_instances;

  uint64_t total_wedges = 0;
  float total_launch_ms = 0.0f;
  uint64_t total_jobs = 0;

  if (use_streaming)
  {

    auto pick_batch_size = [&](double frac = 0.5) -> size_t {
      size_t freeB_now = 0, totalB_now = 0;
      cudaMemGetInfo(&freeB_now, &totalB_now);
      const uint64_t per_job = (uint64_t)(sizeof(Job) + sizeof(uint32_t));
      const uint64_t target  = (uint64_t)(freeB_now * frac);
      uint64_t B = target / per_job;
      if (B == 0) B = 1;
      const uint64_t CAP = 25'000'000ULL;
      if (B > CAP) B = CAP;
      return (size_t)B;
    };

    const size_t B = pick_batch_size(0.5);
    Job* d_jobs = nullptr;
    uint32_t* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs,   B * sizeof(Job)));
    CUDA_CHECK(cudaMalloc(&d_counts, B * sizeof(uint32_t)));

    Job* h_jobs_pinned = nullptr;
    uint32_t* h_counts_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_jobs_pinned,   B * sizeof(Job), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_counts_pinned, B * sizeof(uint32_t), cudaHostAllocDefault));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    auto flush_batch = [&](size_t count) {
      if (count == 0) return;
      total_jobs += count; 

      CUDA_CHECK(cudaMemcpyAsync(d_jobs, h_jobs_pinned, count * sizeof(Job), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(d_counts, 0, count * sizeof(uint32_t), stream));

      Params p = baseP;
      p.jobs          = d_jobs;
      p.num_rays      = static_cast<uint32_t>(count);
      p.next_frontier = nullptr; 
      p.job_counts = d_counts; 

      CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &p, sizeof(Params), cudaMemcpyHostToDevice, stream));

      CUDA_CHECK(cudaEventRecord(e0, stream));
      OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, (unsigned)count, 1, 1));
      CUDA_CHECK(cudaEventRecord(e1, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      float ms = elapsedMs(e0, e1);
      total_launch_ms += ms;

      CUDA_CHECK(cudaMemcpy(h_counts_pinned, d_counts, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      total_wedges += std::accumulate(h_counts_pinned, h_counts_pinned + count, uint64_t{0});
    };

    size_t cur = 0;
    for (int u = 0; u < N; ++u) {
      const uint32_t deg_u = row_ptr[u+1] - row_ptr[u];
      for (uint32_t k = row_ptr[u]; k < row_ptr[u+1]; ++k) {
        uint32_t v = nbrs[k];
        if (u >= (int)v) continue;

        const uint32_t deg_v = row_ptr[v+1] - row_ptr[v];
        const uint32_t ray_source = (deg_u <= deg_v) ? u : v;
        const uint32_t partner    = (deg_u <= deg_v) ? v : u;

        const uint32_t first = uasp_first[ray_source];
        const uint32_t count = uasp_count[ray_source];
        for (uint32_t s = 0; s < count; ++s) {
          h_jobs_pinned[cur++] = { JOIN, ray_source, partner, first + s, 0, 0, 0.0f };
          if (cur == B) { flush_batch(cur); cur = 0; }
        }
      }
    }
    flush_batch(cur);

    CUDA_CHECK(cudaFree(d_jobs));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFreeHost(h_jobs_pinned));
    CUDA_CHECK(cudaFreeHost(h_counts_pinned));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
  }
  
  else
  {
    std::vector<Job> h_jobs_optix;
    h_jobs_optix.reserve((size_t)std::min<uint64_t>(estimated_jobs, (uint64_t)std::numeric_limits<size_t>::max()));

    for (int u = 0; u < N; ++u) {
      const uint32_t deg_u = row_ptr[u+1] - row_ptr[u];
      for (uint32_t k = row_ptr[u]; k < row_ptr[u+1]; ++k) {
        uint32_t v = nbrs[k];
        if (u < (int)v) {
          const uint32_t deg_v = row_ptr[v+1] - row_ptr[v];
          const uint32_t ray_source_node = (deg_u <= deg_v) ? u : v;
          const uint32_t partner_node    = (deg_u <= deg_v) ? v : u;

          const uint32_t first = uasp_first[ray_source_node];
          const uint32_t count = uasp_count[ray_source_node];
          for (uint32_t s = 0; s < count; ++s) {
            h_jobs_optix.push_back({ JOIN, ray_source_node, partner_node, first + s, 0, 0, 0.0f });
          }
        }
      }
    }

    if (h_jobs_optix.empty()) {
      mm.bvh_build_ms = 0.0f;
      return 0;
    }

    const uint32_t num_jobs = static_cast<uint32_t>(h_jobs_optix.size());
    total_jobs += num_jobs;

    Job* d_jobs_optix = nullptr;
    uint32_t* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs_optix, num_jobs * sizeof(Job)));
    CUDA_CHECK(cudaMalloc(&d_counts,    num_jobs * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_jobs_optix, h_jobs_optix.data(), num_jobs * sizeof(Job), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_counts, 0, num_jobs * sizeof(uint32_t), stream));

    Params p = baseP;
    p.jobs          = d_jobs_optix;
    p.num_rays      = num_jobs;
    p.next_frontier = nullptr; 
    p.job_counts    = d_counts;

    CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &p, sizeof(Params), cudaMemcpyHostToDevice, stream));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, stream));
    OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, num_jobs, 1, 1));
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float launch_ms = elapsedMs(e0, e1);
    total_launch_ms += launch_ms;

    std::vector<uint32_t> h_counts(num_jobs);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, num_jobs * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    total_wedges = std::accumulate(h_counts.begin(), h_counts.end(), (uint64_t)0);

    CUDA_CHECK(cudaFree(d_jobs_optix));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
  }

  mm.bvh_build_ms = total_launch_ms;

  uint64_t num_triangles = total_wedges / 3ull;
  std::cout << "  Total jobs" << total_jobs << "\n";

  if (built_streaming_tlas) {
    if (d_tlas_mem_streaming)      CUDA_CHECK(cudaFree((void*)d_tlas_mem_streaming));
    if (d_instance_bases_streaming)CUDA_CHECK(cudaFree((void*)d_instance_bases_streaming));
  }
  return num_triangles;
}

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
  CUstream stream)
{
  const int N = num_vertices;

  constexpr uint64_t NODE_STREAM_THRESHOLD   = 1'000'000ULL;
  constexpr double   GPU_FREE_FRACTION_LIMIT = 0.60;

  auto t0_est = std::chrono::high_resolution_clock::now();
  uint64_t estimated_jobs = 0;
  for (int u = 0; u < N; ++u) {
    const uint32_t deg_u = row_ptr[u+1] - row_ptr[u];
    for (uint32_t k = row_ptr[u]; k < row_ptr[u+1]; ++k) {
      uint32_t v = nbrs[k];
      if (u < (int)v) {
        const uint32_t deg_v = row_ptr[v+1] - row_ptr[v];
        const uint32_t ray_source = (deg_u <= deg_v) ? u : v;
        estimated_jobs += (uint64_t)uasp_count[ray_source];
      }
    }
  }
  auto t1_est = std::chrono::high_resolution_clock::now();
  (void)t0_est; (void)t1_est;

  if (estimated_jobs == 0) {
    mm.bvh_build_ms = 0.0f;
    return;
  }

  size_t freeB = 0, totalB = 0;
  cudaMemGetInfo(&freeB, &totalB);

  const uint64_t needed_bytes_monolithic =
      estimated_jobs * (uint64_t)(sizeof(Job) + sizeof(uint32_t));

  const bool too_many_nodes     = (uint64_t)N > NODE_STREAM_THRESHOLD;
  const bool exceeds_free_vram  = (double)needed_bytes_monolithic > GPU_FREE_FRACTION_LIMIT * (double)freeB;
  const bool use_streaming      = too_many_nodes || exceeds_free_vram;

  OptixTraversableHandle tlas = 0;
  CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
  uint32_t num_instances = 0;
  CUdeviceptr d_tlas_mem_streaming = 0, d_instance_bases_streaming = 0;
  bool built_streaming_tlas = false;

  if (mm.hasSingleTLAS()) {
    mm.getSingleTLAS(&tlas, &d_tlas_mem, &d_instance_bases, &num_instances);
  } else {
    std::vector<uint32_t> all_nodes(N);
    std::iota(all_nodes.begin(), all_nodes.end(), 0u);
    prepare_tlas_for_frontier(mm, all_nodes, uasp_first, &tlas, &d_tlas_mem_streaming,
                              &d_instance_bases_streaming, &num_instances, stream);
    d_tlas_mem = d_tlas_mem_streaming;
    d_instance_bases = d_instance_bases_streaming;
    built_streaming_tlas = true;
  }

  Params baseP = baseParams;
  baseP.tlas                = tlas;
  baseP.instance_prim_bases = (const uint32_t*)d_instance_bases;
  baseP.num_instances       = num_instances;

  uint64_t total_wedges = 0;
  float total_launch_ms = 0.0f;
  uint64_t total_jobs = 0;

  if (use_streaming)
  {
    auto pick_batch_size = [&](double frac = 0.5) -> size_t {
      size_t freeB_now = 0, totalB_now = 0;
      cudaMemGetInfo(&freeB_now, &totalB_now);
      const uint64_t per_job = (uint64_t)(sizeof(Job) + sizeof(uint32_t));
      const uint64_t target  = (uint64_t)(freeB_now * frac);
      uint64_t B = target / per_job;
      if (B == 0) B = 1;
      const uint64_t CAP = 25'000'000ULL;
      if (B > CAP) B = CAP;
      return (size_t)B;
    };

    const size_t B = pick_batch_size(0.5);
    Job* d_jobs = nullptr;
    uint32_t* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs,   B * sizeof(Job)));
    CUDA_CHECK(cudaMalloc(&d_counts, B * sizeof(uint32_t)));

    Job* h_jobs_pinned = nullptr;
    uint32_t* h_counts_pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_jobs_pinned,   B * sizeof(Job), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_counts_pinned, B * sizeof(uint32_t), cudaHostAllocDefault));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    auto flush_batch = [&](size_t count) {
      if (count == 0) return;
      total_jobs += count;

      CUDA_CHECK(cudaMemcpyAsync(d_jobs, h_jobs_pinned, count * sizeof(Job), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(d_counts, 0, count * sizeof(uint32_t), stream));

      Params p = baseP;
      p.jobs          = d_jobs;
      p.num_rays      = static_cast<uint32_t>(count);
      p.next_frontier = d_counts;

      CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &p, sizeof(Params), cudaMemcpyHostToDevice, stream));

      CUDA_CHECK(cudaEventRecord(e0, stream));
      OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, (unsigned)count, 1, 1));
      CUDA_CHECK(cudaEventRecord(e1, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      float ms = elapsedMs(e0, e1);
      total_launch_ms += ms;

      double ns_per_ray = (count > 0) ? (ms * 1e6 / (double)count) : 0.0;
      std::cout << "TriangleCounting batch"
                << " | rays=" << count
                << " | " << ms << " ms"
                << " | " << ns_per_ray << " ns/ray\n";

      CUDA_CHECK(cudaMemcpy(h_counts_pinned, d_counts, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      total_wedges += std::accumulate(h_counts_pinned, h_counts_pinned + count, uint64_t{0});
    };

    size_t cur = 0;
    for (int u = 0; u < N; ++u) {
      const uint32_t deg_u = row_ptr[u+1] - row_ptr[u];
      for (uint32_t k = row_ptr[u]; k < row_ptr[u+1]; ++k) {
        uint32_t v = nbrs[k];
        if (u >= (int)v) continue;

        const uint32_t deg_v = row_ptr[v+1] - row_ptr[v];
        const uint32_t ray_source = (deg_u <= deg_v) ? u : v;
        const uint32_t partner    = (deg_u <= deg_v) ? v : u;

        const uint32_t first = uasp_first[ray_source];
        const uint32_t count = uasp_count[ray_source];
        for (uint32_t s = 0; s < count; ++s) {
          h_jobs_pinned[cur++] = { JOIN, ray_source, partner, first + s, 0, 0, 0.0f };
          if (cur == B) { flush_batch(cur); cur = 0; }
        }
      }
    }
    flush_batch(cur);

    CUDA_CHECK(cudaFree(d_jobs));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFreeHost(h_jobs_pinned));
    CUDA_CHECK(cudaFreeHost(h_counts_pinned));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
  }
  
  else
  {
    std::vector<Job> h_jobs_optix;
    h_jobs_optix.reserve((size_t)std::min<uint64_t>(estimated_jobs, (uint64_t)std::numeric_limits<size_t>::max()));

    for (int u = 0; u < N; ++u) {
      const uint32_t deg_u = row_ptr[u+1] - row_ptr[u];
      for (uint32_t k = row_ptr[u]; k < row_ptr[u+1]; ++k) {
        uint32_t v = nbrs[k];
        if (u < (int)v) {
          const uint32_t deg_v = row_ptr[v+1] - row_ptr[v];
          const uint32_t ray_source_node = (deg_u <= deg_v) ? u : v;
          const uint32_t partner_node    = (deg_u <= deg_v) ? v : u;

          const uint32_t first = uasp_first[ray_source_node];
          const uint32_t count = uasp_count[ray_source_node];
          for (uint32_t s = 0; s < count; ++s) {
            h_jobs_optix.push_back({ JOIN, ray_source_node, partner_node, first + s, 0, 0, 0.0f });
          }
        }
      }
    }

    if (h_jobs_optix.empty()) {
      mm.bvh_build_ms = 0.0f;
      return;
    }

    const uint32_t num_jobs = static_cast<uint32_t>(h_jobs_optix.size());
    total_jobs += num_jobs;

    Job* d_jobs_optix = nullptr;
    uint32_t* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs_optix, num_jobs * sizeof(Job)));
    CUDA_CHECK(cudaMalloc(&d_counts,    num_jobs * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_jobs_optix, h_jobs_optix.data(), num_jobs * sizeof(Job), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_counts, 0, num_jobs * sizeof(uint32_t), stream));

    Params p = baseP;
    p.jobs          = d_jobs_optix;
    p.num_rays      = num_jobs;
    p.next_frontier = d_counts;

    CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &p, sizeof(Params), cudaMemcpyHostToDevice, stream));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, stream));
    OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, num_jobs, 1, 1));
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float launch_ms = elapsedMs(e0, e1);
    total_launch_ms += launch_ms;

    double ns_per_ray = (num_jobs > 0) ? (launch_ms * 1e6 / (double)num_jobs) : 0.0;
    std::cout << "TriangleCounting monolithic"
              << " | rays=" << num_jobs
              << " | " << launch_ms << " ms"
              << " | " << ns_per_ray << " ns/ray\n";

    std::vector<uint32_t> h_counts(num_jobs);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, num_jobs * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    total_wedges = std::accumulate(h_counts.begin(), h_counts.end(), (uint64_t)0);

    CUDA_CHECK(cudaFree(d_jobs_optix));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
  }

  mm.bvh_build_ms = total_launch_ms;

  std::cout << "  Total Jobs: " << total_jobs << "\n";
  std::cout << "  Total GPU time: " << total_launch_ms << " ms\n";
  std::cout << "  Average time per ray: "
            << (total_jobs > 0 ? (total_launch_ms * 1e6 / (double)total_jobs) : 0.0)
            << " ns/ray\n";

  if (built_streaming_tlas) {
    if (d_tlas_mem_streaming)       CUDA_CHECK(cudaFree((void*)d_tlas_mem_streaming));
    if (d_instance_bases_streaming) CUDA_CHECK(cudaFree((void*)d_instance_bases_streaming));
  }
}

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
  CUstream stream, float percent)
{
    float CUDA_JOB_PERCENTAGE = 0.0f;
    if(percent == 0.0f)
      float CUDA_JOB_PERCENTAGE = 0.50f;
    else {
      CUDA_JOB_PERCENTAGE = percent;
    }

    std::vector<uint32_t> degree(num_vertices);
    for (int u = 0; u < num_vertices; ++u)
        degree[u] = row_ptr[u + 1] - row_ptr[u];

    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);

    auto pick_batch_size = [&](double frac, size_t per_item) -> size_t {
        double target = freeB * frac;
        size_t B = (size_t)std::max(1.0, std::floor(target / (double)per_item));
        const size_t MIN_B = 1u << 16;
        const size_t MAX_B = 25'000'000u;
        if (B < MIN_B) B = MIN_B;
        if (B > MAX_B) B = MAX_B;
        B = (B + 255) & ~size_t(255);
        return B;
    };

    const size_t PER_OPTIX_ITEM = sizeof(Job) + sizeof(uint32_t);
    const size_t OPTIX_BATCH    = pick_batch_size(0.50, PER_OPTIX_ITEM);
    const size_t PER_CUDA_ITEM  = sizeof(Job);
    const size_t CUDA_BATCH     = pick_batch_size(0.10, PER_CUDA_ITEM);

    Job*       d_optix_jobs   = nullptr;
    uint32_t*  d_optix_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_optix_jobs,   OPTIX_BATCH * sizeof(Job)));
    CUDA_CHECK(cudaMalloc(&d_optix_counts, OPTIX_BATCH * sizeof(uint32_t)));

    Job*       h_optix_jobs   = nullptr;
    uint32_t*  h_optix_counts = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_optix_jobs,   OPTIX_BATCH * sizeof(Job), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_optix_counts, OPTIX_BATCH * sizeof(uint32_t), cudaHostAllocDefault));

    Job* d_cuda_jobs = nullptr;
    unsigned long long* d_cuda_triangles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cuda_jobs,      CUDA_BATCH * sizeof(Job)));
    CUDA_CHECK(cudaMalloc(&d_cuda_triangles, sizeof(unsigned long long)));

    Job* h_cuda_jobs = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_cuda_jobs, CUDA_BATCH * sizeof(Job), cudaHostAllocDefault));

    OptixTraversableHandle tlas = 0;
    CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
    uint32_t num_instances = 0;
    CUdeviceptr d_tlas_mem_streaming = 0, d_instance_bases_streaming = 0;
    bool built_streaming_tlas = false;

    if (mm.hasSingleTLAS()) {
        mm.getSingleTLAS(&tlas, &d_tlas_mem, &d_instance_bases, &num_instances);
    } else {
        std::vector<uint32_t> all_nodes(num_vertices);
        std::iota(all_nodes.begin(), all_nodes.end(), 0u);
        prepare_tlas_for_frontier(mm, all_nodes, uasp_first, &tlas,
                                  &d_tlas_mem_streaming, &d_instance_bases_streaming,
                                  &num_instances, stream);
        d_tlas_mem = d_tlas_mem_streaming;
        d_instance_bases = d_instance_bases_streaming;
        built_streaming_tlas = true;
    }

    Params baseP = baseParams;
    baseP.tlas                = tlas;
    baseP.instance_prim_bases = (const uint32_t*)d_instance_bases;
    baseP.num_instances       = num_instances;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    const size_t S = std::min<size_t>(100, (size_t)nbrs.size() / 2 + 1);
    std::vector<uint32_t> sample;
    sample.reserve(S);

    size_t seen_edges = 0;
    size_t stride = std::max<size_t>(1, (nbrs.size() / 2) / std::max<size_t>(1, S));
    for (int u = 0; u < num_vertices; ++u) {
        const uint32_t deg_u = degree[u];
        for (uint32_t k = row_ptr[u]; k < row_ptr[u + 1]; ++k) {
            const uint32_t v = nbrs[k];
            if (u >= (int)v) continue;
            if ((seen_edges++ % stride) == 0 && sample.size() < S) {
                const uint32_t deg_v = degree[v];
                sample.push_back(deg_u + deg_v);
            }
        }
        if (sample.size() >= S) break;
    }

    uint32_t degree_sum_threshold = 0;
    if (!sample.empty()) {
        double q = 1.0 - double(CUDA_JOB_PERCENTAGE);
        size_t nth = std::min(sample.size() - 1, (size_t)std::floor(q * sample.size()));
        std::nth_element(sample.begin(), sample.begin() + nth, sample.end());
        degree_sum_threshold = sample[nth];
    } else {
        uint64_t deg_sum_total = 0;
        for (int u = 0; u < num_vertices; ++u) deg_sum_total += degree[u];
        const double avg_deg = num_vertices ? (double)deg_sum_total / num_vertices : 0.0;
        degree_sum_threshold = (uint32_t)std::max(2.0 * avg_deg, 1.0);
    }

    uint64_t partial_tri_cuda  = 0;
    uint64_t partial_tri_optix = 0;
    float    cuda_ms_accum  = 0.0f;
    float    optix_ms_accum = 0.0f;

    size_t   optix_cur = 0;
    size_t   cuda_cur  = 0;

    uint64_t total_optix_jobs = 0;
    uint64_t total_cuda_jobs  = 0;

    auto flush_cuda = [&](size_t count) {
        if (count == 0) return;
        total_cuda_jobs += count;

        CUDA_CHECK(cudaMemcpyAsync(d_cuda_jobs, h_cuda_jobs, count * sizeof(Job), cudaMemcpyHostToDevice, stream));
        float cuda_ms = 0.0f;
        uint64_t res = run_cuda(d_cuda_jobs, (uint32_t)count,
                                baseParams.row_ptr, baseParams.nbrs,
                                d_cuda_triangles, &cuda_ms, stream);
        partial_tri_cuda += res;
        cuda_ms_accum    += cuda_ms;
    };

        auto flush_optix = [&](size_t count) {
          if (count == 0) return;
          total_optix_jobs += count;
  
          CUDA_CHECK(cudaMemcpyAsync(d_optix_jobs, h_optix_jobs, count * sizeof(Job), cudaMemcpyHostToDevice, stream));
          CUDA_CHECK(cudaMemsetAsync(d_optix_counts, 0, count * sizeof(uint32_t), stream));
  
          Params p = baseP;
          p.jobs          = d_optix_jobs;
          p.num_rays      = (uint32_t)count;
          p.next_frontier = d_optix_counts;
          CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &p, sizeof(Params), cudaMemcpyHostToDevice, stream));
  
          CUDA_CHECK(cudaEventRecord(e0, stream));
          OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, (unsigned)count, 1, 1));
          CUDA_CHECK(cudaEventRecord(e1, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          optix_ms_accum += elapsedMs(e0, e1);
  
          CUDA_CHECK(cudaMemcpy(h_optix_counts, d_optix_counts, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
          uint64_t wedges = std::accumulate(h_optix_counts, h_optix_counts + count, uint64_t{0});
          partial_tri_optix += wedges / 3;
      };

    for (int u = 0; u < num_vertices; ++u) {
        const uint32_t deg_u = degree[u];
        for (uint32_t k = row_ptr[u]; k < row_ptr[u + 1]; ++k) {
            const uint32_t v = nbrs[k];
            if (u >= (int)v) continue;

            const uint32_t deg_v = degree[v];
            const uint32_t deg_sum = deg_u + deg_v;
            const bool to_cuda = (deg_sum >= degree_sum_threshold);

            if (!to_cuda) {
                h_cuda_jobs[cuda_cur++] = { JOIN, (uint32_t)u, (uint32_t)v, 0, 0, 0, 0.0f };
                if (cuda_cur == CUDA_BATCH) {
                    flush_cuda(cuda_cur);
                    cuda_cur = 0;
                }
            } else {
                const uint32_t ray_src = (deg_u <= deg_v) ? (uint32_t)u : (uint32_t)v;
                const uint32_t partner = (deg_u <= deg_v) ? (uint32_t)v : (uint32_t)u;
                const uint32_t first = uasp_first[ray_src];
                const uint32_t cnt   = uasp_count[ray_src];

                for (uint32_t s = 0; s < cnt; ++s) {
                    h_optix_jobs[optix_cur++] = { JOIN, ray_src, partner, first + s, 0, 0, 0.0f };
                    if (optix_cur == OPTIX_BATCH) {
                        flush_optix(optix_cur);
                        optix_cur = 0;
                    }
                }
            }
        }
    }

    if (cuda_cur  > 0) flush_cuda(cuda_cur);
    if (optix_cur > 0) flush_optix(optix_cur);

    // ---------------- 9) Cleanup ----------------
    CUDA_CHECK(cudaFree(d_optix_jobs));
    CUDA_CHECK(cudaFree(d_optix_counts));
    CUDA_CHECK(cudaFreeHost(h_optix_jobs));
    CUDA_CHECK(cudaFreeHost(h_optix_counts));
    CUDA_CHECK(cudaFree(d_cuda_jobs));
    CUDA_CHECK(cudaFree(d_cuda_triangles));
    CUDA_CHECK(cudaFreeHost(h_cuda_jobs));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    if (built_streaming_tlas) {
        if (d_tlas_mem_streaming)       CUDA_CHECK(cudaFree((void*)d_tlas_mem_streaming));
        if (d_instance_bases_streaming) CUDA_CHECK(cudaFree((void*)d_instance_bases_streaming));
    }

    const uint64_t total_triangles = partial_tri_cuda + partial_tri_optix;
    mm.bvh_build_ms = cuda_ms_accum + optix_ms_accum;

    std::cout << "  Total CUDA jobs " << total_cuda_jobs << "\n";
    std::cout << "  Total OptiX jobs " << total_optix_jobs << "\n";
}
