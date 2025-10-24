#include "pr.hpp"
#include <iostream>
#include <chrono>
#include <numeric>
#include <iomanip>

#include "shared.h"
#include "common.hpp"
#include "partition.hpp"
#include "kernels/cuda/algorithms.cuh"

#include "memory/buffer.hpp"

std::vector<float> run_pagerank_optix(
  OptixPipeline pipe,
  const OptixShaderBindingTable& sbt,
  CUdeviceptr d_params,              
  const Params& baseParams,          
  int num_vertices,
  int iters,
  float damping,
  GPUMemoryManager& mm,
  const std::vector<uint32_t>& uasp_first, 
  CUstream streamOptix)
{
  const int N = num_vertices;
  if (N <= 0) return {};

  cudaStream_t s = streamOptix;

  DeviceBuffer<float> d_pr_curr(N);
  DeviceBuffer<float> d_pr_next(N);

  {
      std::vector<float> h_init(N, 1.0f / float(N));
      d_pr_curr.uploadAsync(h_init.data(), N, s);
      CUDA_CHECK(cudaMemsetAsync(d_pr_next.ptr, 0, N * sizeof(float), s));
  }

  OptixTraversableHandle tlas = 0;
  CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
  uint32_t num_instances = 0;

  CUdeviceptr d_instance_bases_streaming = 0, d_tlas_mem_streaming = 0;
  bool built_streaming_once = false;

  if (mm.hasSingleTLAS()) {
      CUdeviceptr bases_single = 0, tlas_mem_single = 0;
      uint32_t ninst_single = 0;
      mm.getSingleTLAS(&tlas, &tlas_mem_single, &bases_single, &ninst_single);
      d_instance_bases = bases_single;
      d_tlas_mem       = tlas_mem_single;
      num_instances    = ninst_single;
  } else {
      // Build a global TLAS once for PageRank (pseudo-frontier = all vertices)
      std::vector<uint32_t> pseudo_frontier(N);
      std::iota(pseudo_frontier.begin(), pseudo_frontier.end(), 0u);
      prepare_tlas_for_frontier(mm, pseudo_frontier, uasp_first,
                                &tlas, &d_tlas_mem_streaming, &d_instance_bases_streaming,
                                &num_instances, s);
      d_instance_bases = d_instance_bases_streaming;
      d_tlas_mem       = d_tlas_mem_streaming;
      built_streaming_once = true;
  }

  Params hparams = baseParams;                 
  hparams.num_vertices        = (uint32_t)N;
  hparams.damping             = damping;
  hparams.invN                = 1.0f / float(N);
  hparams.tlas                = tlas;
  hparams.instance_prim_bases = (const uint32_t*)d_instance_bases;
  hparams.num_instances       = num_instances;

  hparams.pr_curr = d_pr_curr.ptr;
  hparams.pr_next = d_pr_next.ptr;

  CUdeviceptr d_params_local = d_params;
  bool params_local_owned = false;
  if (d_params_local == 0) {
      CUDA_CHECK(cudaMalloc((void**)&d_params_local, sizeof(Params)));
      params_local_owned = true;
  }

  DeviceBuffer<Job> d_jobs;
  {
      size_t freeB = 0, totalB = 0;
      CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
      const double JOB_FRACTION = 0.20;
      const size_t jobsBudgetB  = (size_t)(JOB_FRACTION * (double)freeB);
      const size_t bytes_per_job = sizeof(Job);
      size_t batch_cap = (bytes_per_job ? jobsBudgetB / bytes_per_job : 0);
      if (batch_cap == 0) batch_cap = std::min<size_t>(N, 64ull * 1024ull);
      batch_cap = std::min(batch_cap, (size_t)N);
      d_jobs.allocate(batch_cap);
  }
  std::vector<Job> h_jobs; h_jobs.reserve(d_jobs.count);

  cudaEvent_t e0{}, e1{};
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  float total_ms = 0.f;

  const float base_val = (1.0f - damping) / float(N);
  const float epsilon  = 1e-3f;

  std::vector<float> h_prev(N), h_curr(N);
  CUDA_CHECK(cudaMemcpyAsync(h_prev.data(), d_pr_curr.ptr, N*sizeof(float), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  bool converged = false;
  int  final_iter = iters;
  int  total_jobs = 0;

  for (int it = 0; it < iters; ++it) {
      launch_memset_f32(d_pr_next.ptr, base_val, (uint32_t)N, s);

      hparams.pr_curr = d_pr_curr.ptr;
      hparams.pr_next = d_pr_next.ptr;

      float iter_ms = 0.f;

      for (size_t off = 0; off < (size_t)N; off += d_jobs.count) {
          const uint32_t this_batch = (uint32_t)std::min((size_t)d_jobs.count, (size_t)N - off);
          h_jobs.resize(this_batch);
          for (uint32_t i = 0; i < this_batch; ++i) {
              h_jobs[i] = { PAGERANK, (uint32_t)(off + i), 0u, 0u, 0u, 0u, 0.0f };
          }
          total_jobs += (int)this_batch;

          d_jobs.uploadAsync(h_jobs.data(), this_batch, s);

          hparams.jobs     = d_jobs.ptr;
          hparams.num_rays = this_batch;

          CUDA_CHECK(cudaMemcpyAsync((void*)d_params_local, &hparams, sizeof(Params),
                                     cudaMemcpyHostToDevice, s));

          CUDA_CHECK(cudaEventRecord(e0, s));
          OPTIX_CHECK(optixLaunch(pipe, s, d_params_local, sizeof(Params), &sbt,
                                  this_batch, 1, 1));
          CUDA_CHECK(cudaEventRecord(e1, s));
          CUDA_CHECK(cudaStreamSynchronize(s));

          float ms = 0.f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
          iter_ms += ms;
      }

      total_ms += iter_ms;

      CUDA_CHECK(cudaMemcpyAsync(h_curr.data(), d_pr_next.ptr, N*sizeof(float), cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));

      double diff = 0.0;
      for (int i = 0; i < N; ++i)
          diff += std::fabs(h_curr[i] - h_prev[i]);

      if (diff < epsilon) {
          converged = true;
          final_iter = it + 1;
          std::swap(d_pr_curr.ptr, d_pr_next.ptr);  
          break;
      }

      std::swap(d_pr_curr.ptr, d_pr_next.ptr);
      std::swap(h_prev, h_curr);
  }

  std::vector<float> h_pr(N);
  CUDA_CHECK(cudaMemcpyAsync(h_pr.data(), d_pr_curr.ptr, N*sizeof(float), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  mm.bvh_build_ms = total_ms;

  std::cout << "PageRank results after " << final_iter
            << " iteration" << (final_iter == 1 ? "" : "s")
            << (converged ? " (converged)" : "") << ":\n";
  for (int i = 0; i < std::min(N, 10); ++i)
      std::cout << "[" << i << ":" << std::fixed << std::setprecision(6) << h_pr[i] << "] ";
  if (N > 10) std::cout << "...";
  std::cout << "\nTotal Rays: " << total_jobs
            << "\nTotal Time (ms): " << total_ms << "\n";

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  if (params_local_owned)
      cudaFree((void*)d_params_local);

  if (!mm.hasSingleTLAS() && built_streaming_once) {
      if (d_tlas_mem_streaming)
          CUDA_CHECK(cudaFree((void*)d_tlas_mem_streaming));
      if (d_instance_bases_streaming)
          CUDA_CHECK(cudaFree((void*)d_instance_bases_streaming));
  }

  return h_pr;
}

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
  uint32_t deg)
{
  const int N = num_vertices;
  if (N <= 0) { mm.bvh_build_ms = 0.0f; return; }

  std::vector<uint32_t> h_rowptr(N + 1);
  CUDA_CHECK(cudaMemcpyAsync(h_rowptr.data(), baseParams.row_ptr,
                             (N + 1) * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const uint64_t M = (uint64_t)h_rowptr[N];

  uint32_t DEG_T = deg ? deg : (M >= 10000000 ? 32u : 16u);
  std::vector<float>    h_inv_outdeg(N);
  std::vector<float>    h_inv_outdeg_cuda(N);
  std::vector<uint32_t> optix_vertices; optix_vertices.reserve(N);

  uint64_t M_cuda = 0; 
  for (int u = 0; u < N; ++u) {
    const uint32_t du = h_rowptr[u + 1] - h_rowptr[u];
    const float inv = (du ? 1.0f / float(du) : 0.0f);
    h_inv_outdeg[u] = inv;
    if (du >= DEG_T) { h_inv_outdeg_cuda[u] = inv; M_cuda += du; }
    else { h_inv_outdeg_cuda[u] = 0.0f; optix_vertices.push_back((uint32_t)u); }
  }
  const uint32_t O = (uint32_t)optix_vertices.size();

  float *d_pr_curr = nullptr, *d_pr_next = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pr_curr, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pr_next, N * sizeof(float)));
  {
    std::vector<float> init(N, 1.0f / float(N));
    CUDA_CHECK(cudaMemcpyAsync(d_pr_curr, init.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_pr_next, 0, N * sizeof(float), stream));
  }

  float *d_inv_outdeg = nullptr, *d_inv_outdeg_cuda = nullptr;
  CUDA_CHECK(cudaMalloc(&d_inv_outdeg,      N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_inv_outdeg_cuda, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpyAsync(d_inv_outdeg,      h_inv_outdeg.data(),      N * sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_inv_outdeg_cuda, h_inv_outdeg_cuda.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream));

  float* d_dangling_sum = nullptr; CUDA_CHECK(cudaMalloc(&d_dangling_sum, sizeof(float)));
  float* d_diff_sum     = nullptr; CUDA_CHECK(cudaMalloc(&d_diff_sum, sizeof(float)));
  float* h_dangling_pinned = nullptr; CUDA_CHECK(cudaHostAlloc(&h_dangling_pinned, sizeof(float), cudaHostAllocPortable));
  float* h_diff_pinned     = nullptr; CUDA_CHECK(cudaHostAlloc(&h_diff_pinned, sizeof(float), cudaHostAllocPortable));

  OptixTraversableHandle tlas = 0;
  CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
  uint32_t num_instances = 0;
  CUdeviceptr d_tlas_mem_streaming = 0, d_instance_bases_streaming = 0;
  {
    if (mm.hasSingleTLAS()) {
      mm.getSingleTLAS(&tlas, &d_tlas_mem, &d_instance_bases, &num_instances);
    } else {
      std::vector<uint32_t> all_nodes(N);
      std::iota(all_nodes.begin(), all_nodes.end(), 0u);
      prepare_tlas_for_frontier(mm, all_nodes, uasp_first,
                                &tlas, &d_tlas_mem_streaming, &d_instance_bases_streaming,
                                &num_instances, stream);
      d_tlas_mem = d_tlas_mem_streaming;
      d_instance_bases = d_instance_bases_streaming;
    }
  }

  Params baseP = baseParams;
  baseP.num_vertices        = (uint32_t)N;
  baseP.damping             = damping;
  baseP.invN                = 1.0f / float(N);
  baseP.tlas                = tlas;
  baseP.instance_prim_bases = (const uint32_t*)d_instance_bases;
  baseP.num_instances       = num_instances;

  CUdeviceptr d_params_local = 0;
  {
    void* tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp, sizeof(Params)));
    d_params_local = reinterpret_cast<CUdeviceptr>(tmp);
  }

  Job* d_jobs = nullptr;
  Job* h_jobs_pinned = nullptr;
  size_t optix_jobs_cap = 0;
  if (O > 0) {
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const double SAFE_FRAC = 0.20;  // use at most 20% of free VRAM
    const size_t BYTES_PER_JOB = sizeof(Job);
    size_t max_jobs_by_vram = (size_t)((freeB * SAFE_FRAC) / BYTES_PER_JOB);
    const size_t HARD_CAP = 25'000'000u; // safety cap to avoid giant launches
    optix_jobs_cap = std::min<size_t>(std::max<size_t>(1u << 16, max_jobs_by_vram), HARD_CAP);
    if (optix_jobs_cap == 0) optix_jobs_cap = 1u << 16;

    CUDA_CHECK(cudaMalloc(&d_jobs, optix_jobs_cap * sizeof(Job)));
    CUDA_CHECK(cudaHostAlloc((void**)&h_jobs_pinned, optix_jobs_cap * sizeof(Job), cudaHostAllocPortable));
  }

  PRWorkItem* d_items = nullptr;
  {
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const size_t max_safe = (size_t)(freeB * 0.25); // ≤25% of free VRAM
    const size_t need = (size_t)N * sizeof(PRWorkItem);
    size_t alloc_items = (need > max_safe) ? (max_safe / sizeof(PRWorkItem)) : (size_t)N;
    if (alloc_items == 0) alloc_items = 1u << 16;
    if (alloc_items < (size_t)N) {
      std::cerr << "[WARN] PRHybrid: PRWorkItem buffer capped to " << alloc_items << " entries (OOM-safe)\n";
    }
    CUDA_CHECK(cudaMalloc(&d_items, alloc_items * sizeof(PRWorkItem)));
  }

  const float teleport_base = (1.0f - damping) / float(N);
  const float eps = 1e-4f;
  bool  converged = false;
  int   final_iter = iters;
  float total_ms = 0.0f;

  CUstream streamCUDA = nullptr, streamOPTIX = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&streamCUDA, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&streamOPTIX, cudaStreamNonBlocking));

  cudaEvent_t startE, stopCUDA, stopOPTIX;
  CUDA_CHECK(cudaEventCreate(&startE));
  CUDA_CHECK(cudaEventCreate(&stopCUDA));
  CUDA_CHECK(cudaEventCreate(&stopOPTIX));

  uint64_t total_cuda_jobs  = 0; 
  uint64_t total_optix_jobs = 0; 

  for (int it = 0; it < iters; ++it) {
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(d_dangling_sum, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));
    launch_pr_reduce_dangling(d_pr_curr, d_inv_outdeg, N, d_dangling_sum, stream);

    CUDA_CHECK(cudaMemcpyAsync(h_dangling_pinned, d_dangling_sum, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const float base_term = teleport_base + damping * ((*h_dangling_pinned) / float(N));
    launch_pr_fill_base(d_pr_next, N, base_term, stream);

    CUDA_CHECK(cudaEventRecord(startE, 0));

    if (M_cuda > 0) {
      const size_t TILE = (size_t) ( 
        std::max<size_t>(1u, (size_t) ( (size_t) (cudaMemGetInfo,0) )) );
      const size_t TILE_VERTS = (size_t) ( //
        0 ); 
      size_t tile_verts = 0;
    }


    size_t pr_items_capacity = 0; {
      size_t freeB2 = 0, totalB2 = 0;
      CUDA_CHECK(cudaMemGetInfo(&freeB2, &totalB2));
      pr_items_capacity = std::max<size_t>(1u << 16, (size_t)(freeB2 * 0.10) / sizeof(PRWorkItem));
    }

    if (M_cuda > 0) {
      const size_t TILE = std::max<size_t>(1u << 16, pr_items_capacity);
      for (size_t off = 0; off < (size_t)N; off += TILE) {
        uint32_t batch = (uint32_t)std::min<size_t>(TILE, N - off);
        launch_pr_build_items(
          d_pr_curr + off, d_inv_outdeg_cuda + off,
          reinterpret_cast<const uint32_t*>(baseParams.row_ptr) + off,
          batch, d_items, damping, streamCUDA);
        launch_pr_scatter_items(
          d_items, batch, baseParams.nbrs, d_pr_next, streamCUDA);
        total_cuda_jobs += batch;
      }
      CUDA_CHECK(cudaEventRecord(stopCUDA, streamCUDA));
    } else {
      CUDA_CHECK(cudaEventRecord(stopCUDA, 0));
    }

    if (O > 0) {
      for (size_t off = 0; off < (size_t)O; off += optix_jobs_cap) {
        const uint32_t thisBatch = (uint32_t)std::min<size_t>(optix_jobs_cap, (size_t)O - off);

        for (uint32_t i = 0; i < thisBatch; ++i) {
          const uint32_t vtx = optix_vertices[off + i];
          h_jobs_pinned[i] = { PAGERANK, vtx, 0u, 0u, 0u, 0u, 0.0f };
        }
        CUDA_CHECK(cudaMemcpyAsync(d_jobs, h_jobs_pinned, thisBatch * sizeof(Job), cudaMemcpyHostToDevice, streamOPTIX));

        Params p = baseP;
        p.jobs     = d_jobs;
        p.num_rays = thisBatch;
        p.pr_curr  = d_pr_curr;
        p.pr_next  = d_pr_next;
        CUDA_CHECK(cudaMemcpyAsync((void*)d_params_local, &p, sizeof(Params),
                                   cudaMemcpyHostToDevice, streamOPTIX));

        OPTIX_CHECK(optixLaunch(pipe, streamOPTIX, d_params_local, sizeof(Params), &sbt, thisBatch, 1, 1));
        total_optix_jobs += thisBatch;
      }
      CUDA_CHECK(cudaEventRecord(stopOPTIX, streamOPTIX));
    } else {
      CUDA_CHECK(cudaEventRecord(stopOPTIX, 0));
    }

    CUDA_CHECK(cudaEventSynchronize(stopCUDA));
    CUDA_CHECK(cudaEventSynchronize(stopOPTIX));

    float msCUDA = 0.0f, msOPTIX = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msCUDA, startE, stopCUDA));
    CUDA_CHECK(cudaEventElapsedTime(&msOPTIX, startE, stopOPTIX));
    total_ms += (msCUDA > msOPTIX ? msCUDA : msOPTIX);


    CUDA_CHECK(cudaMemcpyAsync(d_diff_sum, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));
    launch_pr_diff_norm(d_pr_curr, d_pr_next, N, d_diff_sum, stream);
    CUDA_CHECK(cudaMemcpyAsync(h_diff_pinned, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if ((*h_diff_pinned) < eps) { converged = true; final_iter = it + 1; break; }

    std::swap(d_pr_curr, d_pr_next);
  }

  std::cout << "  CUDA edges: " << total_cuda_jobs  << std::endl;
  std::cout << "  OptiX jobs: " << total_optix_jobs << std::endl;

  std::vector<float> h_pr(N);
  CUDA_CHECK(cudaMemcpyAsync(h_pr.data(), d_pr_curr, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  mm.bvh_build_ms = total_ms;

  CUDA_CHECK(cudaEventDestroy(startE));
  CUDA_CHECK(cudaEventDestroy(stopCUDA));
  CUDA_CHECK(cudaEventDestroy(stopOPTIX));
  CUDA_CHECK(cudaStreamDestroy(streamCUDA));
  CUDA_CHECK(cudaStreamDestroy(streamOPTIX));

  if (d_params_local) CUDA_CHECK(cudaFree((void*)d_params_local));
  if (d_jobs)         CUDA_CHECK(cudaFree(d_jobs));
  if (h_jobs_pinned)  CUDA_CHECK(cudaFreeHost(h_jobs_pinned));
  if (d_items)        CUDA_CHECK(cudaFree(d_items));
  CUDA_CHECK(cudaFree(d_pr_curr));
  CUDA_CHECK(cudaFree(d_pr_next));
  CUDA_CHECK(cudaFree(d_inv_outdeg));
  CUDA_CHECK(cudaFree(d_inv_outdeg_cuda));
  CUDA_CHECK(cudaFree(d_dangling_sum));
  CUDA_CHECK(cudaFree(d_diff_sum));
  if (h_dangling_pinned) CUDA_CHECK(cudaFreeHost(h_dangling_pinned));
  if (h_diff_pinned)     CUDA_CHECK(cudaFreeHost(h_diff_pinned));

  if (!mm.hasSingleTLAS()) {
    if (d_tlas_mem_streaming)       CUDA_CHECK(cudaFree((void*)d_tlas_mem_streaming));
    if (d_instance_bases_streaming) CUDA_CHECK(cudaFree((void*)d_instance_bases_streaming));
  }
}

void run_pagerank_bench(OptixPipeline pipe,
                               const OptixShaderBindingTable& sbt,
                               CUdeviceptr d_params,
                               const Params& baseParams,
                               int num_vertices,
                               int iters,
                               float damping,
                               GPUMemoryManager& mm,
                               const std::vector<uint32_t>& uasp_first,
                               CUstream streamOptix)
{
    const int N = num_vertices;
    if (N <= 0) return;

    cudaStream_t s = streamOptix;

    float *d_pr_curr = nullptr, *d_pr_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pr_curr, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pr_next, N * sizeof(float)));

    {
        std::vector<float> h_init(N, 1.0f / float(N));
        CUDA_CHECK(cudaMemcpyAsync(d_pr_curr, h_init.data(), N*sizeof(float), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemsetAsync(d_pr_next, 0, N*sizeof(float), s));
    }

    // --- TLAS selection ---
    OptixTraversableHandle tlas = 0;
    CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
    uint32_t num_instances = 0;
    CUdeviceptr d_instance_bases_streaming = 0, d_tlas_mem_streaming = 0;
    bool built_streaming_once = false;

    if (mm.hasSingleTLAS()) {
        CUdeviceptr bases_single = 0, tlas_mem_single = 0;
        uint32_t ninst_single = 0;
        mm.getSingleTLAS(&tlas, &tlas_mem_single, &bases_single, &ninst_single);
        d_instance_bases = bases_single;
        d_tlas_mem       = tlas_mem_single;
        num_instances    = ninst_single;
    } else {
        std::vector<uint32_t> pseudo_frontier(N);
        std::iota(pseudo_frontier.begin(), pseudo_frontier.end(), 0u);
        prepare_tlas_for_frontier(mm, pseudo_frontier, uasp_first,
                                  &tlas, &d_tlas_mem_streaming, &d_instance_bases_streaming,
                                  &num_instances, s);
        d_instance_bases = d_instance_bases_streaming;
        d_tlas_mem       = d_tlas_mem_streaming;
        built_streaming_once = true;
    }

    Params hparams = baseParams;
    hparams.num_vertices        = (uint32_t)N;
    hparams.damping             = damping;
    hparams.invN                = 1.0f / float(N);
    hparams.tlas                = tlas;
    hparams.instance_prim_bases = (const uint32_t*)d_instance_bases;
    hparams.num_instances       = num_instances;

    CUdeviceptr d_params_local = d_params;
    bool params_local_owned = false;
    if (d_params_local == 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_params_local, sizeof(Params)));
        params_local_owned = true;
    }

    Job* d_jobs = nullptr;
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const double JOB_FRACTION = 0.20;
    const size_t jobsBudgetB  = (size_t)(JOB_FRACTION * (double)freeB);
    const size_t bytes_per_job = sizeof(Job);
    size_t batch_cap = bytes_per_job ? (jobsBudgetB / bytes_per_job) : 0;
    if (batch_cap == 0) batch_cap = std::min<size_t>(N, 64ull * 1024ull);
    batch_cap = std::min(batch_cap, (size_t)N);
    CUDA_CHECK(cudaMalloc(&d_jobs, batch_cap * sizeof(Job)));

    std::vector<Job> h_jobs(batch_cap);

    cudaEvent_t e0{}, e1{};
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    float total_ms = 0.f;

    const float base_val = (1.0f - damping) / float(N);
    const float epsilon = 1e-3f;

    std::vector<float> h_prev(N), h_curr(N);
    CUDA_CHECK(cudaMemcpyAsync(h_prev.data(), d_pr_curr, N*sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    bool converged = false;
    int final_iter = iters;
    int total_jobs = 0;

    CUDA_CHECK(cudaStreamSynchronize(s));

    for (int it = 0; it < iters; ++it) {
        launch_memset_f32(d_pr_next, base_val, (uint32_t)N, s);

        hparams.pr_curr = d_pr_curr;
        hparams.pr_next = d_pr_next;

        float iter_ms = 0.f;
        int iter_jobs = 0;

        for (size_t off = 0; off < (size_t)N; off += batch_cap) {
            const uint32_t this_batch = (uint32_t)std::min(batch_cap, (size_t)N - off);
            for (uint32_t i = 0; i < this_batch; ++i) {
                const uint32_t u = (uint32_t)(off + i);
                h_jobs[i] = { PAGERANK, u, 0u, 0u, 0u, 0u, 0.0f };
            }

            iter_jobs += this_batch;
            total_jobs += this_batch;

            CUDA_CHECK(cudaMemcpyAsync(d_jobs, h_jobs.data(),
                                       this_batch * sizeof(Job),
                                       cudaMemcpyHostToDevice, s));

            hparams.jobs     = d_jobs;
            hparams.num_rays = this_batch;
            CUDA_CHECK(cudaMemcpyAsync((void*)d_params_local, &hparams, sizeof(Params),
                                       cudaMemcpyHostToDevice, s));

            CUDA_CHECK(cudaEventRecord(e0, s));
            OPTIX_CHECK(optixLaunch(pipe, s, d_params_local, sizeof(Params), &sbt,
                                    (unsigned)this_batch, 1, 1));
            CUDA_CHECK(cudaEventRecord(e1, s));
            CUDA_CHECK(cudaStreamSynchronize(s));

            iter_ms += elapsedMs(e0, e1);
        }

        total_ms += iter_ms;

        double ns_per_ray = (iter_jobs > 0) ? (iter_ms * 1e6 / (double)iter_jobs) : 0.0;
        std::cout << "PageRank iteration " << it
                  << " | rays=" << iter_jobs
                  << " | " << iter_ms << " ms"
                  << " | " << ns_per_ray << " ns/ray\n";

        CUDA_CHECK(cudaMemcpyAsync(h_curr.data(), d_pr_next, N*sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        double diff = 0.0;
        for (int i = 0; i < N; ++i) diff += std::fabs(h_curr[i] - h_prev[i]);
        if (diff < epsilon) {
            converged = true;
            final_iter = it + 1;
            std::swap(d_pr_curr, d_pr_next);
            break;
        }

        std::swap(d_pr_curr, d_pr_next);
        std::swap(h_prev, h_curr);
    }

    std::vector<float> h_pr(N);
    CUDA_CHECK(cudaMemcpyAsync(h_pr.data(), d_pr_curr, N*sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    mm.bvh_build_ms = total_ms;

    std::cout << "  Total Rays: " << total_jobs << std::endl;
    std::cout << "  Total GPU time: " << total_ms << " ms\n";
    std::cout << "  Average time per ray: "
              << (total_jobs > 0 ? (total_ms * 1e6 / (double)total_jobs) : 0.0)
              << " ns/ray\n";

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    cudaFree(d_pr_curr);
    cudaFree(d_pr_next);
    cudaFree(d_jobs);
    if (params_local_owned && d_params_local) cudaFree((void*)d_params_local);
    if (!mm.hasSingleTLAS() && built_streaming_once) {
        if (d_tlas_mem_streaming)       CUDA_CHECK(cudaFree((void*)d_tlas_mem_streaming));
        if (d_instance_bases_streaming) CUDA_CHECK(cudaFree((void*)d_instance_bases_streaming));
    }
}
