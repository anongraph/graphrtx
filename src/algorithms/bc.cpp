#include "bc.hpp"
#include <iostream>
#include <chrono>
#include <numeric>
#include <iomanip>

#include "shared.h"
#include "common.hpp"
#include "partition.hpp"
#include "kernels/cuda/algorithms.cuh"

#ifndef DEBUG_BC
#define DEBUG_BC 0
#endif

std::vector<float> run_betweenness_optix(
  OptixPipeline pipe,
  const OptixShaderBindingTable& sbt,
  CUdeviceptr d_params,
  const Params& baseParams,
  int num_vertices,
  GPUMemoryManager& mm,
  const std::vector<uint32_t>& uasp_first,
  CUstream streamOptix)
{
  using clk = std::chrono::high_resolution_clock;
  const auto wall_t0 = clk::now();

  const int N = num_vertices;
  if (N <= 0) { mm.bvh_build_ms = 0.0; return {}; }

  auto SAFE_SYNC = [&](const char* where){
      CUDA_CHECK(cudaPeekAtLastError());
      CUDA_CHECK(cudaStreamSynchronize(streamOptix));
#if DEBUG_BC
      std::cout << "[SYNC] " << where << "\n";
#endif
  };

  cudaEvent_t startEvt, stopEvt;
  CUDA_CHECK(cudaEventCreate(&startEvt));
  CUDA_CHECK(cudaEventCreate(&stopEvt));

  float* d_bc    = nullptr; CUDA_CHECK(cudaMalloc(&d_bc,    (size_t)N * sizeof(float)));
  uint32_t* d_sigma = nullptr; CUDA_CHECK(cudaMalloc(&d_sigma, (size_t)N * sizeof(uint32_t)));
  float* d_delta = nullptr; CUDA_CHECK(cudaMalloc(&d_delta, (size_t)N * sizeof(float)));
  uint32_t* d_dist  = nullptr; CUDA_CHECK(cudaMalloc(&d_dist,  (size_t)N * sizeof(uint32_t)));

  // --- Frontier capacity from free GPU memory ---
  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
  const double HEADROOM = 0.35;
  size_t budgetB = static_cast<size_t>((1.0 - HEADROOM) * (double)freeB);
  if (budgetB < (32ull << 20)) budgetB = (32ull << 20);

  const size_t jobBytes = sizeof(Job);
  const size_t perItemBytes = sizeof(uint32_t) * 2 + jobBytes;
  const size_t fixedOverhead = 1ull << 20;

  uint32_t frontier_cap = 0;
  if (budgetB > fixedOverhead && perItemBytes > 0)
      frontier_cap = static_cast<uint32_t>(std::min<size_t>(
          std::max<size_t>((budgetB - fixedOverhead) / perItemBytes, 64ull * 1024ull),
          (size_t)N));
  else
      frontier_cap = std::min<uint32_t>(N, 64u * 1024u);

#if DEBUG_BC
  std::cout << "[DBG] N=" << N << " frontier_cap=" << frontier_cap
            << " free=" << (double)freeB/1e9 << "GB total=" << (double)totalB/1e9 << "GB\n";
#endif

  // --- Allocate frontiers & jobs ---
  uint32_t* d_frontier      = nullptr; CUDA_CHECK(cudaMalloc(&d_frontier,      (size_t)frontier_cap * sizeof(uint32_t)));
  uint32_t* d_next_frontier = nullptr; CUDA_CHECK(cudaMalloc(&d_next_frontier, (size_t)frontier_cap * sizeof(uint32_t)));
  uint32_t* d_next_count    = nullptr; CUDA_CHECK(cudaMalloc(&d_next_count,    sizeof(uint32_t)));
  Job* d_jobs          = nullptr; CUDA_CHECK(cudaMalloc(&d_jobs,          (size_t)frontier_cap * sizeof(Job)));

  uint32_t* h_level_store = nullptr;
  size_t h_level_store_cap = 0;
  auto ensure_level_host_cap = [&](size_t need) {
      if (need > h_level_store_cap) {
          if (h_level_store) { cudaFreeHost(h_level_store); h_level_store = nullptr; }
          h_level_store_cap = std::max(need, h_level_store_cap ? 2 * h_level_store_cap : (size_t)1 << 20);
          CUDA_CHECK(cudaHostAlloc(&h_level_store, h_level_store_cap * sizeof(uint32_t), cudaHostAllocPortable));
      }
  };

  std::vector<uint32_t> level_sizes;
  uint32_t level_store_off = 0;

  Params hparams = baseParams;
  hparams.num_vertices   = (uint32_t)N;
  hparams.next_capacity  = frontier_cap;
  hparams.bc_values      = d_bc;
  hparams.sigma          = d_sigma;
  hparams.delta          = d_delta;
  hparams.distances      = d_dist;

  // --- Initialize Brandes state ---
  const int source = 0;
  launch_memset_u32(d_sigma, 0u, (uint32_t)N, streamOptix);
  launch_memset_f32(d_delta, 0.0f, (uint32_t)N, streamOptix);
  CUDA_CHECK(cudaMemsetAsync(d_dist, 0xFF, (size_t)N * sizeof(uint32_t), streamOptix));
  CUDA_CHECK(cudaMemsetAsync(d_bc,   0,    (size_t)N * sizeof(float),    streamOptix));

  {
      uint32_t one = 1, zero = 0;
      uint32_t src_u = (uint32_t)source;
      CUDA_CHECK(cudaMemcpyAsync(&d_sigma[source], &one,  sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
      CUDA_CHECK(cudaMemcpyAsync(&d_dist[source],  &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
      CUDA_CHECK(cudaMemcpyAsync(d_frontier, &src_u, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
      SAFE_SYNC("init state");
  }

  // --- Reuse single TLAS if available ---
  CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
  OptixTraversableHandle tlas_single = 0;
  uint32_t num_instances_single = 0;
  const bool has_single_tlas = mm.hasSingleTLAS();
  if (has_single_tlas)
      mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);

  CUDA_CHECK(cudaEventRecord(startEvt, streamOptix));

  uint64_t total_jobs = 0;
  uint32_t total_optix_launches_fwd = 0;
  uint32_t total_optix_launches_bwd = 0;

  hparams.bc_phase = 0u;
  uint32_t frontier_size = 1u, cur_depth = 0u;

#if DEBUG_BC
  std::cout << "[FWD] begin\n";
#endif
  while (frontier_size > 0u) {
      total_jobs += frontier_size;

      ensure_level_host_cap((size_t)level_store_off + frontier_size);
      const uint32_t CHUNK = 1u << 20;
      for (uint32_t off = 0; off < frontier_size; ) {
          uint32_t c = std::min<uint32_t>(CHUNK, frontier_size - off);
          CUDA_CHECK(cudaMemcpyAsync(h_level_store + level_store_off + off,
                                     d_frontier + off,
                                     (size_t)c * sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost, streamOptix));
          off += c;
      }
      SAFE_SYNC("copy frontier to host level_store");

      level_sizes.push_back(frontier_size);
      level_store_off += frontier_size;

#if DEBUG_BC
      std::cout << "  [FWD] depth=" << cur_depth
                << " frontier_size=" << frontier_size << "\n";
#endif

      uint32_t zero = 0;
      CUDA_CHECK(cudaMemcpyAsync(d_next_count, &zero, sizeof(uint32_t),
                                 cudaMemcpyHostToDevice, streamOptix));
      SAFE_SYNC("zero next_count");

      CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
      uint32_t num_instances = 0;
      OptixTraversableHandle tlas = 0;

      if (has_single_tlas) {
          tlas = tlas_single;
          d_instance_bases = d_instance_bases_single;
          d_tlas_mem = d_tlas_mem_single;
          num_instances = num_instances_single;
      } else {
          // Dynamic TLAS build logic remains, relies on external prepare_tlas_for_frontier
          std::vector<uint32_t> curFront(frontier_size);
          CUDA_CHECK(cudaMemcpy(curFront.data(), d_frontier, (size_t)frontier_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
          SAFE_SYNC("copy current frontier to host (tlas prep)");
          prepare_tlas_for_frontier(mm, curFront, uasp_first, &tlas, &d_tlas_mem,
                                    &d_instance_bases, &num_instances, streamOptix);
      }

      launch_build_jobs_from_nodes(d_frontier, frontier_size, d_jobs, BETWEENNESS, streamOptix);
      SAFE_SYNC("build jobs fwd");

      Params iter = hparams;
      iter.tlas                = tlas;
      iter.instance_prim_bases = (const uint32_t*)d_instance_bases;
      iter.num_instances       = num_instances;
      iter.jobs                = d_jobs;
      iter.num_rays            = frontier_size;
      iter.next_frontier       = d_next_frontier;
      iter.next_count          = d_next_count;
      //iter.depth               = cur_depth; // Pass depth to the kernel
      CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iter, sizeof(Params),
                                 cudaMemcpyHostToDevice, streamOptix));

      OPTIX_CHECK(optixLaunch(pipe, streamOptix, d_params, sizeof(Params), &sbt,
                              frontier_size, 1, 1));
      ++total_optix_launches_fwd;
      SAFE_SYNC("optix fwd launch");

      uint32_t next_size = 0;
      CUDA_CHECK(cudaMemcpyAsync(&next_size, d_next_count, sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost, streamOptix));
      SAFE_SYNC("read next_size");

      std::swap(d_frontier, d_next_frontier);
      frontier_size = next_size;
      ++cur_depth;

      if (!has_single_tlas) {
          if (d_tlas_mem)       cudaFree((void*)d_tlas_mem);
          if (d_instance_bases) cudaFree((void*)d_instance_bases);
      }
  }

#if DEBUG_BC
  {
      std::vector<uint32_t> h_dist(N), h_sigma(N);
      CUDA_CHECK(cudaMemcpy(h_dist.data(),  d_dist,  N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_sigma.data(), d_sigma, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      SAFE_SYNC("dump dist/sigma after forward");

      size_t max_print = std::min<int>(N, 32);
      size_t discovered = 0, sigma_pos = 0;
      for (int i = 0; i < N; ++i) {
          if (h_dist[i] != 0xFFFFFFFFu) ++discovered;
          if (h_sigma[i] > 0) ++sigma_pos;
      }
      std::cout << "[DBG] Levels=" << level_sizes.size()
                << " optix_fwd_launches=" << total_optix_launches_fwd << "\n";
      std::cout << "[DBG] Discovered vertices: " << discovered << " / " << N << "\n";
      std::cout << "[DBG] sigma>0 count: " << sigma_pos << "\n";

      std::cout << "[DBG] dist[0:" << max_print << "): ";
      for (size_t i = 0; i < max_print; ++i)
          std::cout << (h_dist[i]==0xFFFFFFFFu? -1 : (int)h_dist[i]) << " ";
      std::cout << "\n";

      std::cout << "[DBG] sigma[0:" << max_print << "): ";
      for (size_t i = 0; i < max_print; ++i)
          std::cout << h_sigma[i] << " ";
      std::cout << "\n";

      std::cout << "[DBG] level_sizes: ";
      for (auto s: level_sizes) std::cout << s << " ";
      std::cout << "\n";
  }
#endif

  hparams.bc_phase = 1u;

  uint32_t off = level_store_off;
  std::vector<uint32_t> offsets(level_sizes.size());
  for (int i = (int)level_sizes.size() - 1; i >= 0; --i) {
      off -= level_sizes[i];
      offsets[i] = off;
  }

  launch_memset_f32(d_delta, 0.0f, (uint32_t)N, streamOptix);
  SAFE_SYNC("zero delta");

  uint32_t* d_level_scratch = nullptr;
  CUDA_CHECK(cudaMalloc(&d_level_scratch, (size_t)frontier_cap * sizeof(uint32_t)));

#if DEBUG_BC
  std::cout << "[BWD] begin (levels " << level_sizes.size() << ")\n";
#endif
  for (int li = (int)level_sizes.size() - 1; li >= 1; --li) {
      const uint32_t count = level_sizes[li];
      if (count == 0) continue;

      CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
      uint32_t num_instances = 0;
      OptixTraversableHandle tlas = 0;

      if (has_single_tlas) {
          tlas = tlas_single;
          d_instance_bases = d_instance_bases_single;
          d_tlas_mem = d_tlas_mem_single;
          num_instances = num_instances_single;
      } else {
          std::vector<uint32_t> levelNodes(count);
          std::memcpy(levelNodes.data(), h_level_store + offsets[li], (size_t)count * sizeof(uint32_t));
          prepare_tlas_for_frontier(mm, levelNodes, uasp_first, &tlas, &d_tlas_mem,
                                    &d_instance_bases, &num_instances, streamOptix);
      }

#if DEBUG_BC
      std::cout << "  [BWD] level idx=" << li << " count=" << count
                << " num_instances=" << num_instances << " tlas=" << (uint64_t)tlas << "\n";
#endif

      auto dump_delta_stats = [&](const char* tag){
#if DEBUG_BC
          std::vector<float> h_delta(N);
          CUDA_CHECK(cudaMemcpy(h_delta.data(), d_delta, N * sizeof(float), cudaMemcpyDeviceToHost));
          SAFE_SYNC(tag);
          double sum = 0.0; size_t nz = 0;
          for (int i = 0; i < N; ++i) { sum += h_delta[i]; if (h_delta[i] != 0.0f) ++nz; }
          std::cout << "    [DELTA] " << tag << " sum=" << sum << " nz=" << nz << "\n";
#endif
      };

      dump_delta_stats("before level");

      for (uint32_t pos = 0; pos < count; ) {
        const uint32_t c = std::min<uint32_t>(count - pos, frontier_cap);
        if (c == 0) { break; }

        CUDA_CHECK(cudaMemcpyAsync(d_level_scratch,
                                   h_level_store + offsets[li] + pos,
                                   (size_t)c * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, streamOptix));
        SAFE_SYNC("copy level chunk");

        launch_build_jobs_from_nodes(d_level_scratch, c, d_jobs, BETWEENNESS, streamOptix);
        SAFE_SYNC("build jobs bwd");

        if (num_instances == 0 || tlas == 0) {
#if DEBUG_BC
            std::cout << "    [BWD] skip launch (no instances)\n";
#endif
            pos += c;
            continue;
        }

        Params iter = hparams;
        iter.bc_phase            = 1u;
        iter.tlas                = tlas;
        iter.instance_prim_bases = (const uint32_t*)d_instance_bases;
        iter.num_instances       = num_instances;
        iter.jobs                = d_jobs;
        iter.num_rays            = c;

        CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iter, sizeof(Params),
                                   cudaMemcpyHostToDevice, streamOptix));
        SAFE_SYNC("copy params bwd");

        total_jobs += c;

        OPTIX_CHECK(optixLaunch(pipe, streamOptix, d_params, sizeof(Params), &sbt, c, 1, 1));
        ++total_optix_launches_bwd;
        CUDA_CHECK(cudaPeekAtLastError());
        SAFE_SYNC("optix bwd launch");

        launch_add_delta_to_bc(d_level_scratch, c, /*source=*/0, d_delta, d_bc, (uint32_t)N, streamOptix);
        SAFE_SYNC("add_delta_to_bc");

        pos += c;
      }

      dump_delta_stats("after level");

      if (!has_single_tlas) {
          if (d_tlas_mem)       cudaFree((void*)d_tlas_mem);
          if (d_instance_bases) cudaFree((void*)d_instance_bases);
      }
  }

  CUDA_CHECK(cudaEventRecord(stopEvt, streamOptix));
  CUDA_CHECK(cudaEventSynchronize(stopEvt));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, startEvt, stopEvt));

#if DEBUG_BC
  {
      std::vector<float> h_bc_dbg(N);
      CUDA_CHECK(cudaMemcpy(h_bc_dbg.data(), d_bc, N * sizeof(float), cudaMemcpyDeviceToHost));
      SAFE_SYNC("final bc snapshot");

      double sum_bc = 0.0; size_t nz_bc = 0;
      for (int i = 0; i < N; ++i) { sum_bc += h_bc_dbg[i]; if (h_bc_dbg[i] != 0.0f) ++nz_bc; }
      size_t max_print = std::min<int>(N, 32);
      std::cout << "[DBG] bc[0:" << max_print << "): ";
      for (size_t i = 0; i < max_print; ++i) std::cout << h_bc_dbg[i] << " ";
      std::cout << "\n[DBG] bc sum=" << sum_bc << " nz=" << nz_bc
                << " fwd_launches=" << total_optix_launches_fwd
                << " bwd_launches=" << total_optix_launches_bwd << "\n";
  }
#endif

  const auto wall_t1 = clk::now();
  const double e2e_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

  std::cout << "  Total jobs = " << total_jobs << "\n";

  std::vector<float> h_bc(N);
  CUDA_CHECK(cudaMemcpy(h_bc.data(), d_bc, N * sizeof(float), cudaMemcpyDeviceToHost));

  mm.bvh_build_ms = ms;
  {
      std::cout << "Betweenness Centrality results (source vertex: 0)\n";
      int printed = 0;
      for (int i = 0; i < N && printed < 10; ++i) {
          std::cout << "[" << i << ":" << std::fixed << std::setprecision(6) << h_bc[i] << "] ";
          ++printed;
      }
      if (printed < N) std::cout << "...";
      std::cout << "\n";
      std::cout << "Total Jobs: " << total_jobs << std::endl;
      std::cout << "Traversal Time (ms): " << ms << std::endl;
      std::cout << "End-to-End Time (ms): " << e2e_ms << std::endl;
  }

  // --- Cleanup ---
  CUDA_CHECK(cudaEventDestroy(startEvt));
  CUDA_CHECK(cudaEventDestroy(stopEvt));
  if (h_level_store) cudaFreeHost(h_level_store);
  if (d_level_scratch) cudaFree(d_level_scratch);

  cudaFree(d_bc);
  cudaFree(d_sigma);
  cudaFree(d_delta);
  cudaFree(d_dist);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_count);
  cudaFree(d_jobs);

  return h_bc;
}

void run_betweenness_hybrid(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream streamOptix, uint32_t thres)
{
    using clk = std::chrono::high_resolution_clock;
    const auto wall_t0 = clk::now();

    const int N = num_vertices;
    if (N <= 0) { mm.bvh_build_ms = 0.0; return; }

    uint32_t frontier_threshold = 0;
    if(thres == 0) 
      uint32_t frontier_threshold = 100000u; // split CUDA vs OptiX
    else 
      frontier_threshold = thres;

    cudaEvent_t devStart{}, devStop{};
    CUDA_CHECK(cudaEventCreate(&devStart));
    CUDA_CHECK(cudaEventCreate(&devStop));

    float*     d_bc    = nullptr; CUDA_CHECK(cudaMalloc(&d_bc,    (size_t)N * sizeof(float)));
    uint32_t*  d_sigma = nullptr; CUDA_CHECK(cudaMalloc(&d_sigma, (size_t)N * sizeof(uint32_t)));
    float*     d_delta = nullptr; CUDA_CHECK(cudaMalloc(&d_delta, (size_t)N * sizeof(float)));
    uint32_t*  d_dist  = nullptr; CUDA_CHECK(cudaMalloc(&d_dist,  (size_t)N * sizeof(uint32_t)));

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const double HEADROOM = 0.35;
    size_t budgetB = (size_t)((1.0 - HEADROOM) * (double)freeB);
    if (budgetB < (32ull << 20)) budgetB = (32ull << 20);

    const size_t jobBytes = sizeof(Job);
    const size_t perItemBytes = sizeof(uint32_t) * 2 + jobBytes;
    const size_t fixedOverhead = 1ull << 20;

    uint32_t frontier_cap = 0;
    if (budgetB > fixedOverhead && perItemBytes > 0) {
        size_t avail = (budgetB - fixedOverhead) / perItemBytes;
        frontier_cap = (uint32_t)std::min<size_t>(std::max<size_t>(avail, 64ull * 1024ull), (size_t)N);
    } else frontier_cap = std::min<uint32_t>(N, 64u * 1024u);

    uint32_t* d_frontier      = nullptr; CUDA_CHECK(cudaMalloc(&d_frontier,      (size_t)frontier_cap * sizeof(uint32_t)));
    uint32_t* d_next_frontier = nullptr; CUDA_CHECK(cudaMalloc(&d_next_frontier, (size_t)frontier_cap * sizeof(uint32_t)));
    uint32_t* d_next_count    = nullptr; CUDA_CHECK(cudaMalloc(&d_next_count,    sizeof(uint32_t)));
    Job*      d_jobs          = nullptr; CUDA_CHECK(cudaMalloc(&d_jobs,          (size_t)frontier_cap * sizeof(Job)));

    uint32_t* h_level_store = nullptr; size_t h_level_store_cap = 0;
    auto ensure_level_host_cap = [&](size_t need){
        if (need > h_level_store_cap) {
            if (h_level_store) CUDA_CHECK(cudaFreeHost(h_level_store));
            h_level_store_cap = std::max(need, h_level_store_cap ? 2 * h_level_store_cap : (size_t)1 << 20);
            CUDA_CHECK(cudaHostAlloc(&h_level_store, h_level_store_cap * sizeof(uint32_t), cudaHostAllocPortable));
        }
    };
    std::vector<uint32_t> level_sizes; level_sizes.reserve(256);
    uint32_t level_store_off = 0;

    Params hparams = baseParams;
    hparams.num_vertices   = (uint32_t)N;
    hparams.next_capacity  = frontier_cap;
    hparams.bc_values      = d_bc;
    hparams.sigma          = d_sigma;
    hparams.delta          = d_delta;
    hparams.distances      = d_dist;
    hparams.next_frontier  = d_next_frontier;
    hparams.next_count     = d_next_count;

    const int source = 0;
    launch_memset_u32(d_sigma, 0u, (uint32_t)N, streamOptix);
    launch_memset_f32(d_delta, 0.0f, (uint32_t)N, streamOptix);
    CUDA_CHECK(cudaMemsetAsync(d_dist, 0xFF, (size_t)N * sizeof(uint32_t), streamOptix));
    CUDA_CHECK(cudaMemsetAsync(d_bc,   0,    (size_t)N * sizeof(float),    streamOptix));
    {
        uint32_t one = 1, zero = 0;
        uint32_t src_u = (uint32_t)source;
        CUDA_CHECK(cudaMemcpyAsync(&d_sigma[source], &one,  sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
        CUDA_CHECK(cudaMemcpyAsync(&d_dist[source],  &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
        CUDA_CHECK(cudaMemcpyAsync(d_frontier, &src_u, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
        CUDA_CHECK(cudaStreamSynchronize(streamOptix));
    }

    // --- Single TLAS reuse ---
    CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
    OptixTraversableHandle tlas_single = 0;
    uint32_t num_instances_single = 0;
    const bool has_single_tlas = mm.hasSingleTLAS();
    if (has_single_tlas)
        mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);

    // --- Job counters ---
    uint64_t total_cuda_jobs = 0;
    uint64_t total_optix_jobs = 0;

    auto process_forward_in_batches = [&](uint32_t* d_src_frontier, uint32_t fsize, uint32_t cur_depth) -> uint32_t {
        uint32_t z = 0u;
        CUDA_CHECK(cudaMemcpyAsync(d_next_count, &z, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
        CUDA_CHECK(cudaStreamSynchronize(streamOptix));

        const uint32_t B = frontier_cap;
        for (uint32_t off = 0; off < fsize; off += B) {
            const uint32_t batch = std::min<uint32_t>(B, fsize - off);

            if (batch >= frontier_threshold) {
                total_cuda_jobs += batch; // CUDA path
                launch_bc_forward_expand_nodes(
                    d_src_frontier + off, batch,
                    baseParams.row_ptr, baseParams.nbrs,
                    d_next_frontier, d_next_count,
                    d_dist, d_sigma, cur_depth, streamOptix);
                CUDA_CHECK(cudaStreamSynchronize(streamOptix));
            } else {
                total_optix_jobs += batch; // OptiX path
                CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
                OptixTraversableHandle tlas = 0;
                uint32_t num_instances = 0;

                if (has_single_tlas) {
                    tlas = tlas_single;
                    d_instance_bases = d_instance_bases_single;
                    d_tlas_mem = d_tlas_mem_single;
                    num_instances = num_instances_single;
                } else {
                    CUDA_CHECK(cudaStreamSynchronize(streamOptix));
                    std::vector<uint32_t> curFront(batch);
                    CUDA_CHECK(cudaMemcpy(curFront.data(), d_src_frontier + off,
                                          (size_t)batch * sizeof(uint32_t),
                                          cudaMemcpyDeviceToHost));
                    prepare_tlas_for_frontier(mm, curFront, uasp_first,
                                              &tlas, &d_tlas_mem, &d_instance_bases, &num_instances, streamOptix);
                }

                launch_build_jobs_from_nodes(d_src_frontier + off, batch, d_jobs, BETWEENNESS, streamOptix);

                Params iterP = hparams;
                iterP.tlas                = tlas;
                iterP.instance_prim_bases = (const uint32_t*)d_instance_bases;
                iterP.num_instances       = num_instances;
                iterP.jobs                = d_jobs;
                iterP.num_rays            = batch;
                iterP.bc_phase            = 0u;
                CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iterP, sizeof(Params),
                                           cudaMemcpyHostToDevice, streamOptix));

                OPTIX_CHECK(optixLaunch(pipe, streamOptix, d_params, sizeof(Params), &sbt,
                                        batch, 1, 1));
                CUDA_CHECK(cudaStreamSynchronize(streamOptix));

                if (!has_single_tlas) {
                    if (d_tlas_mem)       CUDA_CHECK(cudaFree((void*)d_tlas_mem));
                    if (d_instance_bases) CUDA_CHECK(cudaFree((void*)d_instance_bases));
                }
            }
        }

        uint32_t produced_total = 0;
        CUDA_CHECK(cudaMemcpyAsync(&produced_total, d_next_count, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, streamOptix));
        CUDA_CHECK(cudaStreamSynchronize(streamOptix));
        return produced_total;
    };

    CUDA_CHECK(cudaEventRecord(devStart, streamOptix));

    hparams.bc_phase = 0u;
    uint32_t frontier_size = 1u, cur_depth = 0u;

    while (frontier_size > 0u) {
        ensure_level_host_cap(level_store_off + frontier_size);
        const uint32_t CHUNK = 1u << 20;
        for (uint32_t off = 0; off < frontier_size; ) {
            uint32_t c = std::min<uint32_t>(CHUNK, frontier_size - off);
            CUDA_CHECK(cudaMemcpyAsync(h_level_store + level_store_off + off,
                                       d_frontier + off,
                                       (size_t)c * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, streamOptix));
            off += c;
        }
        CUDA_CHECK(cudaStreamSynchronize(streamOptix));
        level_sizes.push_back(frontier_size);
        level_store_off += frontier_size;

        uint32_t next_size = process_forward_in_batches(d_frontier, frontier_size, cur_depth);
        std::swap(d_frontier, d_next_frontier);
        frontier_size = next_size;
        ++cur_depth;
    }

    hparams.bc_phase = 1u;
    uint32_t off_levels = level_store_off;
    std::vector<uint32_t> offsets(level_sizes.size());
    for (int i = (int)level_sizes.size() - 1; i >= 0; --i) {
        off_levels -= level_sizes[i];
        offsets[i] = off_levels;
    }

    launch_memset_f32(d_delta, 0.0f, (uint32_t)N, streamOptix);
    uint32_t* d_level_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_level_scratch, (size_t)frontier_cap * sizeof(uint32_t)));

    for (int li = (int)level_sizes.size() - 1; li >= 1; --li) {
        const uint32_t count = level_sizes[li];
        if (count == 0) continue;
        total_cuda_jobs += count; // backward CUDA work

        const uint32_t host_off = offsets[li];
        for (uint32_t pos = 0; pos < count; ) {
            const uint32_t c = std::min<uint32_t>(count - pos, frontier_cap);

            CUDA_CHECK(cudaMemcpyAsync(d_level_scratch,
                                       h_level_store + host_off + pos,
                                       (size_t)c * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, streamOptix));
            CUDA_CHECK(cudaStreamSynchronize(streamOptix));

            launch_bc_backward_accumulate_nodes(
                d_level_scratch, c,
                baseParams.row_ptr, baseParams.nbrs,
                d_dist, d_sigma,
                d_delta, d_delta,
                streamOptix);
            CUDA_CHECK(cudaStreamSynchronize(streamOptix));

            pos += c;
        }
    }

    if (level_store_off > 0u) {
        uint32_t remaining = level_store_off;
        uint32_t processed = 0;
        while (remaining) {
            const uint32_t c = std::min<uint32_t>(remaining, frontier_cap);
            total_cuda_jobs += c; // add_delta kernel
            CUDA_CHECK(cudaMemcpyAsync(d_level_scratch,
                                       h_level_store + processed,
                                       (size_t)c * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, streamOptix));
            CUDA_CHECK(cudaStreamSynchronize(streamOptix));
            //launch_add_delta_to_bc(d_level_scratch, c, source, d_delta, d_bc, streamOptix);
            CUDA_CHECK(cudaStreamSynchronize(streamOptix));
            processed += c;
            remaining -= c;
        }
    }

    // --- Stop timing ---
    CUDA_CHECK(cudaEventRecord(devStop, streamOptix));
    CUDA_CHECK(cudaEventSynchronize(devStop));
    float device_active_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&device_active_ms, devStart, devStop));

    const auto wall_t1 = clk::now();
    const double e2e_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

    // --- Summary print ---
    std::cout << "  Total CUDA jobs " << total_cuda_jobs << "\n";
    std::cout << "  Total OptiX jobs " << total_optix_jobs << "\n";

    mm.bvh_build_ms = device_active_ms;

    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(devStart));
    CUDA_CHECK(cudaEventDestroy(devStop));
    if (h_level_store) CUDA_CHECK(cudaFreeHost(h_level_store));
    if (d_level_scratch) cudaFree(d_level_scratch);
    cudaFree(d_bc);
    cudaFree(d_sigma);
    cudaFree(d_delta);
    cudaFree(d_dist);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_count);
    cudaFree(d_jobs);
}