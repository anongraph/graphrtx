#include "bfs.hpp"
#include <iostream>
#include <chrono>

#include "shared.h"
#include "common.hpp"
#include "partition.hpp"

#include "kernels/cuda/algorithms.cuh"

std::vector<uint32_t> run_bfs(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    GPUMemoryManager& mm,
    CUdeviceptr d_params,
    const Params& baseParams,
    int source,
    int num_vertices,
    const std::vector<uint32_t>& uasp_first,
    const std::vector<uint32_t>& uasp_count)
  {
    cudaStream_t streamD2H = nullptr, streamOptix = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamD2H, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamOptix, cudaStreamNonBlocking));
  
    cudaEvent_t evOptixDone{};
    CUDA_CHECK(cudaEventCreateWithFlags(&evOptixDone, cudaEventDisableTiming));
  
    const int N = num_vertices;
    if (N <= 0) { mm.bvh_build_ms = 0.0f; return {}; }
  
    cudaDeviceProp props{};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
    const int WAVE_MULT = 8;
    const int TARGET_THREADS = props.multiProcessorCount * 512 * WAVE_MULT;
  
    uint32_t* d_visited = nullptr;
    uint32_t* d_dist = nullptr;
    const size_t visited_words = (N + 31) / 32;
    CUDA_CHECK(cudaMalloc(&d_visited, visited_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * sizeof(uint32_t)));
  
    CUDA_CHECK(cudaMemsetAsync(d_visited, 0, visited_words * sizeof(uint32_t), streamOptix));
    CUDA_CHECK(cudaMemsetAsync(d_dist, 0xFF, N * sizeof(uint32_t), streamOptix));
  
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const double HEADROOM = 0.50;
    size_t budgetB = static_cast<size_t>((1.0 - HEADROOM) * double(freeB));
    if (budgetB < (32ull << 20)) budgetB = (32ull << 20);
  
    const size_t jobBytes = sizeof(Job);
    const size_t perItemBytes = sizeof(uint32_t) * 2 + jobBytes;
    const size_t fixedOverhead = 1ull << 20;
  
    uint32_t frontier_cap = static_cast<uint32_t>(
        std::min<size_t>(
            std::max<size_t>((budgetB - fixedOverhead) / perItemBytes, 64ull * 1024ull),
            (size_t)N));
  
    Job* d_jobs[2] = { nullptr, nullptr };
    uint32_t* d_next_frontier[2] = { nullptr, nullptr };
    uint32_t* d_next_count[2] = { nullptr, nullptr };
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&d_jobs[i], frontier_cap * sizeof(Job)));
        CUDA_CHECK(cudaMalloc(&d_next_frontier[i], frontier_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_next_count[i], sizeof(uint32_t)));
    }
  
    std::vector<uint32_t> frontier{ (uint32_t)source };
    const uint32_t w = uint32_t(source) >> 5;
    const uint32_t m = 1u << (source & 31);
    uint32_t zero = 0;
  
    CUDA_CHECK(cudaMemcpyAsync(&d_visited[w], &m, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
    CUDA_CHECK(cudaMemcpyAsync(&d_dist[source], &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
  
    Params hparams = baseParams;
    hparams.distances       = d_dist;
    hparams.visited_bitmap  = d_visited;
    hparams.num_vertices    = N;
    hparams.visited_words   = (uint32_t)visited_words;
    hparams.next_capacity   = frontier_cap;
  
    // --- TLAS: single or per-level ---
    bool has_single_tlas = mm.hasSingleTLAS();
    CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
    OptixTraversableHandle tlas_single = 0;
    uint32_t num_instances_single = 0;
  
    if (has_single_tlas) {
        mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);
        hparams.tlas                  = tlas_single;
        hparams.instance_prim_bases   = (const uint32_t*)d_instance_bases_single;
        hparams.num_instances         = num_instances_single;
    }
  
    CUDA_CHECK(cudaStreamSynchronize(streamOptix));
  
    Job*   h_jobs_pinned = nullptr;
    size_t h_jobs_bytes  = 0;
    auto ensure_host_jobs_capacity = [&](size_t needBytes) {
        if (needBytes <= h_jobs_bytes) return;
        if (h_jobs_pinned) { cudaFreeHost(h_jobs_pinned); h_jobs_pinned = nullptr; h_jobs_bytes = 0; }
        // grow with slack
        h_jobs_bytes = std::max(needBytes, (size_t)(256 * 1024));
        CUDA_CHECK(cudaHostAlloc((void**)&h_jobs_pinned, h_jobs_bytes, cudaHostAllocDefault));
    };
  
    // --- BFS main loop ---
    int buf = 0, level = 0;
    float total_ms = 0.f;
    cudaEvent_t e0{}, e1{};
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
  
    int total_job = 0;
  
    while (!frontier.empty()) {
        const int cur = buf;
        const int nxt = 1 - buf;
  
        const uint32_t numJobs = (uint32_t)frontier.size();
        if (numJobs == 0) break;
  
        ensure_host_jobs_capacity(numJobs * sizeof(Job));
        for (uint32_t i = 0; i < numJobs; ++i) {
            const uint32_t u = frontier[i];
            h_jobs_pinned[i] = { EXPAND, u, 0, 0u, (uint32_t)level, 0, 0.0f };
        }
  
        // Upload jobs & reset counters on the OptiX stream
        CUDA_CHECK(cudaMemcpyAsync(d_jobs[cur], h_jobs_pinned, numJobs * sizeof(Job),
                                   cudaMemcpyHostToDevice, streamOptix));
        CUDA_CHECK(cudaMemcpyAsync(d_next_count[cur], &zero, sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, streamOptix));
  
        // Select/build TLAS for this level
        CUdeviceptr d_tlas_mem = 0, d_instance_bases = 0;
        OptixTraversableHandle tlas = 0;
        uint32_t num_instances = 0;
  
        if (has_single_tlas) {
            tlas = tlas_single;
            d_tlas_mem = d_tlas_mem_single;
            d_instance_bases = d_instance_bases_single;
            num_instances = num_instances_single;
        } else {
            std::vector<uint32_t> curFront(frontier.begin(), frontier.end());
            // prepare_tlas_for_frontier should honor streamOptix for async builds
            prepare_tlas_for_frontier(mm, curFront, uasp_first,
                                      &tlas, &d_tlas_mem, &d_instance_bases, &num_instances, streamOptix);
        }
  
        // Update params for this level
        hparams.jobs                 = d_jobs[cur];
        hparams.num_rays             = numJobs;                 // real work size
        hparams.next_frontier        = d_next_frontier[cur];
        hparams.next_count           = d_next_count[cur];
        hparams.tlas                 = tlas;
        hparams.instance_prim_bases  = (const uint32_t*)d_instance_bases;
        hparams.num_instances        = num_instances;
  
        CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &hparams, sizeof(Params),
                                   cudaMemcpyHostToDevice, streamOptix));
  
        // Launch width padding for occupancy (guarded in raygen by i < num_rays)
        const uint32_t launch_width = std::max<uint32_t>(numJobs, (uint32_t)TARGET_THREADS);
  
        // Launch
        total_job += numJobs;
        CUDA_CHECK(cudaStreamSynchronize(streamOptix));
        CUDA_CHECK(cudaEventRecord(e0, streamOptix));
        OPTIX_CHECK(optixLaunch(pipe, streamOptix, d_params, sizeof(Params), &sbt, launch_width, 1, 1));
        CUDA_CHECK(cudaEventRecord(e1, streamOptix));
  
        // Copy next frontier size
        CUDA_CHECK(cudaEventRecord(evOptixDone, streamOptix));
        CUDA_CHECK(cudaStreamWaitEvent(streamD2H, evOptixDone, 0));
  
        uint32_t next_size = 0;
        CUDA_CHECK(cudaMemcpyAsync(&next_size, d_next_count[cur], sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, streamD2H));
        CUDA_CHECK(cudaStreamSynchronize(streamD2H));
  
        float ms_level = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_level, e0, e1));
        total_ms += ms_level;
  
        if (next_size == 0) {
            
            if (!has_single_tlas) {
                if (d_tlas_mem) cudaFree((void*)d_tlas_mem);
                if (d_instance_bases) cudaFree((void*)d_instance_bases);
            }
            break;
        }
  
        // Retrieve next frontier
        std::vector<uint32_t> next(next_size);
        CUDA_CHECK(cudaMemcpyAsync(next.data(), d_next_frontier[cur],
                                   next_size * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, streamD2H));
        CUDA_CHECK(cudaStreamSynchronize(streamD2H));
  
        if (!has_single_tlas) {
            if (d_tlas_mem) cudaFree((void*)d_tlas_mem);
            if (d_instance_bases) cudaFree((void*)d_instance_bases);
        }
  
        frontier.swap(next);
        buf = nxt;
        ++level;
    }
  
    std::cout << "  Total Jobs: " << total_job << std::endl;
  
    std::vector<uint32_t> hdist(N);
    CUDA_CHECK(cudaMemcpy(hdist.data(), d_dist, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  
    mm.bvh_build_ms = total_ms;
  
    {
        std::cout << "Source vertex: " << source << std::endl;
        const uint32_t INF = 0xFFFFFFFFu;
        int printed = 0;
    
        for (int i = 0; i < N && printed < 10; ++i) {
            if (hdist[i] == INF) continue;
            std::cout << "[" << i << ":" << hdist[i] << "] ";
            ++printed;
        }
        if (printed < N) std::cout << "...";
        std::cout << std::endl;
    
        std::cout << "Total Jobs: " << total_job << std::endl;
        std::cout << "Traversal time (ms): " << total_ms << std::endl;
    }

    for (int i = 0; i < 2; ++i) {
        if (d_jobs[i])          cudaFree(d_jobs[i]);
        if (d_next_frontier[i]) cudaFree(d_next_frontier[i]);
        if (d_next_count[i])    cudaFree(d_next_count[i]);
    }
    cudaFree(d_visited);
    cudaFree(d_dist);
  
    if (h_jobs_pinned) cudaFreeHost(h_jobs_pinned);
  
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(evOptixDone));
  
    CUDA_CHECK(cudaStreamDestroy(streamD2H));
    CUDA_CHECK(cudaStreamDestroy(streamOptix));

    return hdist;
  }
  
void run_bfs_bench(
  OptixPipeline pipe,
  const OptixShaderBindingTable& sbt,
  GPUMemoryManager& mm,
  CUdeviceptr d_params,
  const Params& baseParams,
  int source,
  int num_vertices,
  const std::vector<uint32_t>& uasp_first,
  const std::vector<uint32_t>& uasp_count)
{
  cudaStream_t streamD2H = nullptr, streamOptix = nullptr; 
  CUDA_CHECK(cudaStreamCreateWithFlags(&streamD2H, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&streamOptix, cudaStreamNonBlocking));

  cudaEvent_t evOptixDone{}; 
  CUDA_CHECK(cudaEventCreateWithFlags(&evOptixDone, cudaEventDisableTiming));

  const int N = num_vertices;
  if (N <= 0) { mm.bvh_build_ms = 0.0; return; }

  // --- Allocate visited + dist bitmaps ---
  uint32_t *d_visited = nullptr, *d_dist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_visited, ((N + 31) / 32) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_dist, N * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemsetAsync(d_visited, 0, ((N + 31) / 32) * sizeof(uint32_t), streamOptix));
  CUDA_CHECK(cudaMemsetAsync(d_dist, 0xFF, N * sizeof(uint32_t), streamOptix));

  // --- Determine frontier capacity ---
  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
  const double HEADROOM = 0.50; 
  size_t budgetB = static_cast<size_t>((1.0 - HEADROOM) * (double)freeB);
  if (budgetB < (32ull << 20)) budgetB = (32ull << 20);
  const size_t jobBytes = sizeof(Job);
  const size_t perItemBytes = sizeof(uint32_t) * 2 + jobBytes;
  const size_t fixedOverhead = 1ull << 20;
  uint32_t frontier_cap = static_cast<uint32_t>(
      std::min<size_t>(std::max<size_t>((budgetB - fixedOverhead) / perItemBytes, 64ull * 1024ull), (size_t)N));

  // --- Allocate buffers ---
  Job* d_jobs[2] = { nullptr, nullptr };
  uint32_t* d_next_frontier[2] = { nullptr, nullptr };
  uint32_t* d_next_count[2] = { nullptr, nullptr };
  for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaMalloc(&d_jobs[i], frontier_cap * sizeof(Job)));
      CUDA_CHECK(cudaMalloc(&d_next_frontier[i], frontier_cap * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&d_next_count[i], sizeof(uint32_t)));
  }

  // --- Source initialization ---
  std::vector<uint32_t> frontier{ (uint32_t)source };
  const uint32_t w = source >> 5;
  const uint32_t m = 1u << (source & 31);
  uint32_t zero = 0;
  CUDA_CHECK(cudaMemcpyAsync(&d_visited[w], &m, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
  CUDA_CHECK(cudaMemcpyAsync(&d_dist[source], &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));

  // --- Setup params ---
  Params hparams = baseParams;
  hparams.distances      = d_dist;
  hparams.visited_bitmap = d_visited;
  hparams.num_vertices   = N;
  hparams.visited_words  = (N + 31) / 32;
  hparams.next_capacity  = frontier_cap;

  // --- TLAS setup ---
  bool has_single_tlas = mm.hasSingleTLAS();
  CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
  OptixTraversableHandle tlas_single = 0;
  uint32_t num_instances_single = 0;

  if (has_single_tlas) {
      mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);
      hparams.tlas = tlas_single;
      hparams.instance_prim_bases = (const uint32_t*)d_instance_bases_single;
      hparams.num_instances = num_instances_single;
  }

  CUDA_CHECK(cudaStreamSynchronize(streamOptix));

  // --- BFS main loop ---
  int buf = 0, level = 0;
  float total_ms = 0.f;
  cudaEvent_t e0{}, e1{};
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  int total_job = 0;
  while (!frontier.empty()) {
      const int cur = buf;
      const int nxt = 1 - buf;

      // --- Prepare job list ---
      std::vector<Job> jobs;
      jobs.reserve(frontier.size());
      for (uint32_t u : frontier)
          jobs.push_back({ EXPAND, u, 0, 0u, (uint32_t)level, 0, 0.0f });
      const uint32_t numJobs = (uint32_t)jobs.size();
      if (numJobs == 0) break;

      // --- Upload jobs ---
      CUDA_CHECK(cudaMemcpyAsync(d_jobs[cur], jobs.data(), numJobs * sizeof(Job),
                                 cudaMemcpyHostToDevice, streamOptix));
      CUDA_CHECK(cudaMemcpyAsync(d_next_count[cur], &zero, sizeof(uint32_t),
                                 cudaMemcpyHostToDevice, streamOptix));

      // --- Select or build TLAS ---
      CUdeviceptr d_tlas_mem = 0, d_instance_bases = 0;
      OptixTraversableHandle tlas = 0;
      uint32_t num_instances = 0;

      if (has_single_tlas) {
          tlas = tlas_single;
          d_tlas_mem = d_tlas_mem_single;
          d_instance_bases = d_instance_bases_single;
          num_instances = num_instances_single;
      } else {
          std::vector<uint32_t> curFront(frontier.begin(), frontier.end());
          prepare_tlas_for_frontier(mm, curFront, uasp_first,
                                    &tlas, &d_tlas_mem, &d_instance_bases, &num_instances, streamOptix);
      }

      // --- Update params ---
      hparams.jobs          = d_jobs[cur];
      hparams.num_rays      = numJobs;
      hparams.next_frontier = d_next_frontier[cur];
      hparams.next_count    = d_next_count[cur];
      hparams.tlas          = tlas;
      hparams.instance_prim_bases = (const uint32_t*)d_instance_bases;
      hparams.num_instances = num_instances;

      CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &hparams, sizeof(Params),
                                 cudaMemcpyHostToDevice, streamOptix));

      // --- Launch OptiX ---
      total_job += numJobs;
      CUDA_CHECK(cudaEventRecord(e0, streamOptix));
      OPTIX_CHECK(optixLaunch(pipe, streamOptix, d_params, sizeof(Params), &sbt, numJobs, 1, 1));
      CUDA_CHECK(cudaEventRecord(e1, streamOptix));

      CUDA_CHECK(cudaEventRecord(evOptixDone, streamOptix));
      CUDA_CHECK(cudaStreamWaitEvent(streamD2H, evOptixDone, 0));

      uint32_t next_size = 0;
      CUDA_CHECK(cudaMemcpyAsync(&next_size, d_next_count[cur], sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost, streamD2H));
      CUDA_CHECK(cudaStreamSynchronize(streamD2H));

      float ms_level = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms_level, e0, e1));
      total_ms += ms_level;

      double per_ray_ns = (numJobs > 0) ? (ms_level * 1e6 / numJobs) : 0.0;
      std::cout << "BFS level " << level
                << " | frontier=" << frontier.size()
                << " | next=" << next_size
                << " | " << ms_level << " ms"
                << " | " << per_ray_ns << " ns/ray\n";

      if (next_size == 0) break;

      std::vector<uint32_t> next(next_size);
      CUDA_CHECK(cudaMemcpyAsync(next.data(), d_next_frontier[cur],
                                 next_size * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost, streamD2H));
      CUDA_CHECK(cudaStreamSynchronize(streamD2H));

      if (!has_single_tlas) {
          if (d_tlas_mem) cudaFree((void*)d_tlas_mem);
          if (d_instance_bases) cudaFree((void*)d_instance_bases);
      }

      frontier.swap(next);
      buf = nxt;
      ++level;
  }

  std::cout << "  Total Jobs: " << total_job << std::endl;
  std::cout << "  Total GPU time: " << total_ms << " ms\n";
  std::cout << "  Average time per ray: " 
            << (total_job > 0 ? (total_ms * 1e6 / total_job) : 0.0)
            << " ns/ray\n";

  mm.bvh_build_ms = total_ms;

  
  for (int i = 0; i < 2; ++i) {
      if (d_jobs[i])          cudaFree(d_jobs[i]);
      if (d_next_frontier[i]) cudaFree(d_next_frontier[i]);
      if (d_next_count[i])    cudaFree(d_next_count[i]);
  }
  cudaFree(d_visited);
  cudaFree(d_dist);

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaEventDestroy(evOptixDone));
  CUDA_CHECK(cudaStreamDestroy(streamD2H));
  CUDA_CHECK(cudaStreamDestroy(streamOptix));
}

void run_bfs_hybrid(OptixPipeline pipe,
  const OptixShaderBindingTable& sbt,
  GPUMemoryManager& mm,
  CUdeviceptr d_params,
  const Params& baseParams,
  int source,
  int num_vertices,
  const std::vector<uint32_t>& /*uasp_first*/,
  const std::vector<uint32_t>& /*uasp_count*/,
  uint32_t thres)
{
    auto checkDevPtr = [](const void* p, const char* name)->bool {
        if (!p) {
            std::cerr << "ERROR: " << name << " is null\n";
            return false;
        }
        cudaPointerAttributes a{};
        cudaError_t st = cudaPointerGetAttributes(&a, p);
        
        bool is_valid = (st == cudaSuccess) && 
                        (a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeHost);

        if (!is_valid) {
            std::cerr << "ERROR: " << name << " is not a valid device pointer. "
                      << "Status=" << st << ", Type=" << (int)a.type 
                      << " (Expected 1=Device or 2=Host for Mapped)\n";
            return false;
        }
        return true;
    };
    CUstream stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    const int N = num_vertices;
    if (source < 0 || source >= N) {
        std::cerr << "ERROR: run_bfs_hybrid: source out of range\n";
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    uint32_t frontier_threshold = 0;
    if(thres == 0)
      frontier_threshold = 100000; // switch point
    else 
      frontier_threshold = thres;

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

    // Allocate 30% of remaining free memory for these large temporary buffers.
    const double HEADROOM = 0.70;
    size_t budgetB = static_cast<size_t>((1.0 - HEADROOM) * (double)freeB);

    const size_t perItemBytes = sizeof(Job) + sizeof(uint32_t); // Job + next_frontier slot
    const size_t fixedOverhead = 1ull << 20; // 1MB buffer

    uint32_t safe_cap = static_cast<uint32_t>(
        std::min<size_t>(std::max<size_t>((budgetB - fixedOverhead) / perItemBytes, 64ull * 1024ull), (size_t)N));

    const uint32_t max_frontier_cap = safe_cap;

    // --- Device buffers ---
    uint32_t *d_visited = nullptr, *d_dist = nullptr;
    const uint32_t visited_words = (N + 31) / 32;
    CUDA_CHECK(cudaMalloc(&d_visited, visited_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_visited, 0, visited_words * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dist, 0xFF, N * sizeof(uint32_t), stream));

    // Using bounded capacity for d_jobs and d_next_frontier
    Job* d_jobs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs, max_frontier_cap * sizeof(Job)));

    uint32_t *d_next_frontier = nullptr, *d_next_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next_frontier, max_frontier_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_count, sizeof(uint32_t)));

    // --- Validate critical base pointers are device pointers ---
    const uint32_t* d_row_ptr = baseParams.row_ptr;
    const uint32_t* d_nbrs    = baseParams.nbrs;

    // The check is now correct for mapped memory.
    if (!checkDevPtr(d_row_ptr, "row_ptr") ||
        !checkDevPtr(d_nbrs,    "nbrs")     ||
        !checkDevPtr((void*)d_params, "d_params"))
    {
        // Cleanup and return
        CUDA_CHECK(cudaFree(d_visited));
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_jobs));
        CUDA_CHECK(cudaFree(d_next_frontier));
        CUDA_CHECK(cudaFree(d_next_count));
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    // --- Source vertex setup ---
    uint32_t src_mask = 1u << (source & 31);
    CUDA_CHECK(cudaMemcpyAsync(&d_visited[source >> 5], &src_mask, sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(&d_dist[source], &zero, sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));

    // --- Initialize BFS params (host copy) ---
    Params hparams = baseParams;            
    hparams.distances      = d_dist;
    hparams.visited_bitmap = d_visited;
    hparams.num_vertices   = N;
    hparams.visited_words  = visited_words;
    hparams.next_capacity  = max_frontier_cap;

    // TLAS setup
    OptixTraversableHandle tlas = 0;
    CUdeviceptr d_tlas_mem = 0, d_instance_bases = 0;
    uint32_t num_instances = 0;
    mm.getSingleTLAS(&tlas, &d_tlas_mem, &d_instance_bases, &num_instances);

    if (tlas == 0) {
        std::cerr << "ERROR: run_bfs_hybrid failed: No Single TLAS available (check MM_FRACTION in main).\n";

        CUDA_CHECK(cudaFree(d_visited));
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_jobs));
        CUDA_CHECK(cudaFree(d_next_frontier));
        CUDA_CHECK(cudaFree(d_next_count));
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    hparams.tlas = tlas;
    hparams.instance_prim_bases = (const uint32_t*)d_instance_bases;
    hparams.num_instances = num_instances;

    if (!checkDevPtr((void*)hparams.instance_prim_bases, "instance_prim_bases")) {
        CUDA_CHECK(cudaFree(d_visited));
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_jobs));
        CUDA_CHECK(cudaFree(d_next_frontier));
        CUDA_CHECK(cudaFree(d_next_count));
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- BFS loop ---
    std::vector<uint32_t> frontier{ (uint32_t)source };
    int level = 0;
    float total_ms = 0.f;

    int total_cuda_jobs = 0;
    int total_optix_jobs = 0;
    while (!frontier.empty()) {
        const uint32_t frontier_size = (uint32_t)frontier.size();

        // Ensure current frontier fits in the bounded capacity
        if (frontier_size > max_frontier_cap) {
            std::cerr << "ERROR: Frontier size (" << frontier_size
                      << ") exceeds buffer capacity (" << max_frontier_cap
                      << "). BFS terminating early.\n";
            break;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_next_count, &zero, sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));

        int job_size = 0;
        // Build EXPAND jobs for current frontier (host-side)
        {
            std::vector<Job> jobs(frontier_size);
            for (uint32_t i = 0; i < frontier_size; ++i) {
                jobs[i] = { EXPAND, frontier[i], 0, 0, (uint32_t)level, 0, 0.0f };
            }
            CUDA_CHECK(cudaMemcpyAsync(d_jobs, jobs.data(),
                                       frontier_size * sizeof(Job),
                                       cudaMemcpyHostToDevice, stream));
            job_size = jobs.size();
        }

        if (frontier_size >= frontier_threshold) {
            // --- CUDA path ---
            total_cuda_jobs += job_size;
            cudaEventRecord(e0, stream);
            // Call the updated launch signature that includes capacity & visited_words
            launch_bfs_expand(
                d_jobs, frontier_size,
                d_row_ptr, d_nbrs,
                d_next_frontier, d_next_count, hparams.next_capacity,
                d_visited, hparams.visited_words,
                d_dist,
                stream
            );
            cudaEventRecord(e1, stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));

            float ms = elapsedMs(e0, e1);
            total_ms += ms;

            CUDA_CHECK(cudaGetLastError());
        } else {
            total_optix_jobs += job_size;
            // --- OptiX path ---
            hparams.jobs          = d_jobs;
            hparams.num_rays      = frontier_size;
            hparams.next_frontier = d_next_frontier;
            hparams.next_count    = d_next_count;

            // Copy Params to device (using the same stream as OptiX launch)
            CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &hparams, sizeof(Params),
                                       cudaMemcpyHostToDevice, stream));

            cudaEventRecord(e0, stream);
            OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt,
                                    frontier_size, 1, 1));
            cudaEventRecord(e1, stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));

            float ms = elapsedMs(e0, e1);
            total_ms += ms;

            CUDA_CHECK(cudaGetLastError());
        }

        // --- Next frontier ---
        uint32_t next_size = 0;
        
        CUDA_CHECK(cudaMemcpy(&next_size, d_next_count, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        if (next_size == 0) break;

        if (next_size > max_frontier_cap) {
            /*std::cerr << "WARNING: Next frontier size (" << next_size
                      << ") exceeds buffer capacity (" << max_frontier_cap
                      << "). Clipping frontier to prevent host memory overrun.\n";*/
            next_size = max_frontier_cap;
        }

        std::vector<uint32_t> next(next_size);
        CUDA_CHECK(cudaMemcpy(next.data(), d_next_frontier,
                              next_size * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        frontier.swap(next);
        level++;
    }

    // --- Final distances (optional copy-out retained) ---
    std::vector<uint32_t> hdist(N);
    CUDA_CHECK(cudaMemcpy(hdist.data(), d_dist, N * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    mm.bvh_build_ms = total_ms;

    
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_jobs));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_count));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "  Total CUDA jobs: " << total_cuda_jobs << std::endl;
    std::cout << "  Total Optix jobs: " << total_optix_jobs << std::endl;
}
