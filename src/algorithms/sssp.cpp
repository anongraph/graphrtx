#include "sssp.hpp"
#include <iostream>
#include <chrono>
#include <numeric>
#include <iomanip>

#include "shared.h"
#include "common.hpp"
#include "partition.hpp"
#include "kernels/cuda/algorithms.cuh"

std::vector<float> run_sssp(OptixPipeline pipe,
                     const OptixShaderBindingTable& sbt,
                     CUdeviceptr d_params,
                     const Params& baseParams,
                     int source,
                     int num_vertices,
                     GPUMemoryManager& mm,
                     const std::vector<uint32_t>& uasp_first,
                     CUstream streamOptix)
{
    const int N = num_vertices;
    if (N <= 0) return {};

    float* d_sssp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sssp, N * sizeof(float)));

    std::vector<float> hinit(N, INFINITY);
    hinit[source] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_sssp, hinit.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

    const double HEADROOM = 0.70;
    const size_t budgetB  = (size_t)((1.0 - HEADROOM) * (double)freeB);
    const size_t fixedOverhead = 1ull << 20; // ~1 MiB
    size_t cap_candidatesB = (budgetB > fixedOverhead) ? (budgetB - fixedOverhead) : budgetB;
    uint32_t next_cap = (uint32_t)std::min<size_t>(
        std::max<size_t>(cap_candidatesB / sizeof(uint32_t), 64ull * 1024ull),
        (size_t)N
    );

    const double JOB_FRACTION = 0.15; // 15% of free for jobs
    size_t jobsBudgetB = (size_t)(JOB_FRACTION * (double)freeB);
    uint32_t job_batch_cap = (uint32_t)std::min<size_t>(
        std::max<size_t>(jobsBudgetB / sizeof(Job), 64ull * 1024ull),
        (size_t)N
    );

    uint32_t* d_next_frontier = nullptr;
    uint32_t* d_next_count    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next_frontier, (size_t)next_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_count, sizeof(uint32_t)));

    Job* d_jobs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs, (size_t)job_batch_cap * sizeof(Job)));
    std::vector<Job> h_jobs(job_batch_cap);

    Params hparams = baseParams;
    hparams.num_vertices   = (uint32_t)N;
    hparams.next_capacity  = (uint32_t)next_cap;   
    hparams.sssp_distances = d_sssp;
    hparams.next_frontier  = d_next_frontier;
    hparams.next_count     = d_next_count;

    std::vector<uint32_t> frontier{ (uint32_t)source };

    CUdeviceptr d_instance_bases_single = 0;
    CUdeviceptr d_tlas_mem_single = 0;
    OptixTraversableHandle tlas_single = 0;
    uint32_t num_instances_single = 0;
    if (mm.hasSingleTLAS()) {
        mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);
        if (tlas_single == 0) {
            std::cerr << "ERROR: SSSP: Single TLAS invalid.\n";
            cudaFree(d_sssp); cudaFree(d_next_frontier); cudaFree(d_next_count); cudaFree(d_jobs);
            return {};
        }
    }

    int level = 0;
    double total_ms_e2e = 0.0;

    cudaEvent_t e0{}, e1{};
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    auto zero_device_u32 = [&](uint32_t* ptr){
        const uint32_t z = 0;
        CUDA_CHECK(cudaMemcpyAsync(ptr, &z, sizeof(uint32_t), cudaMemcpyHostToDevice, streamOptix));
    };

    int total_jobs = 0;
    while (!frontier.empty()) {
        auto t0 = std::chrono::high_resolution_clock::now();

        CUdeviceptr d_instance_bases = 0;
        CUdeviceptr d_tlas_mem = 0;
        OptixTraversableHandle tlas = 0;
        uint32_t num_instances = 0;

        if (mm.hasSingleTLAS()) {
            tlas = tlas_single;
            d_instance_bases = d_instance_bases_single;
            d_tlas_mem = d_tlas_mem_single;
            num_instances = num_instances_single;
        } else {
            prepare_tlas_for_frontier(mm, frontier, uasp_first, &tlas, &d_tlas_mem,
                                      &d_instance_bases, &num_instances, streamOptix);
            if (tlas == 0) {
                std::cerr << "ERROR: SSSP: streaming TLAS build failed.\n";
                break;
            }
        }

        zero_device_u32(d_next_count);

        const uint32_t numJobsTotal = (uint32_t)frontier.size();
        float gpu_ms_level = 0.f;

        for (uint32_t off = 0; off < numJobsTotal; off += job_batch_cap) {
            const uint32_t thisBatch = std::min(job_batch_cap, numJobsTotal - off);

            for (uint32_t i = 0; i < thisBatch; ++i) {
                Job j{}; j.qtype = SSSP; j.src = frontier[off + i];
                h_jobs[i] = j;
            }

            CUDA_CHECK(cudaMemcpyAsync(d_jobs, h_jobs.data(),
                                       (size_t)thisBatch * sizeof(Job),
                                       cudaMemcpyHostToDevice, streamOptix));

            Params iter = hparams;
            iter.tlas                = tlas;
            iter.jobs                = d_jobs;
            iter.num_rays            = thisBatch;
            iter.instance_prim_bases = (const uint32_t*)d_instance_bases;
            iter.num_instances       = num_instances;
            total_jobs += iter.num_rays;
            CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iter, sizeof(Params),
                                       cudaMemcpyHostToDevice, streamOptix));

            CUDA_CHECK(cudaEventRecord(e0, streamOptix));
            OPTIX_CHECK(optixLaunch(pipe, streamOptix, d_params, sizeof(Params), &sbt, thisBatch, 1, 1));
            CUDA_CHECK(cudaEventRecord(e1, streamOptix));
            CUDA_CHECK(cudaStreamSynchronize(streamOptix));
            gpu_ms_level += elapsedMs(e0, e1);
        }

        uint32_t next_size = 0;
        CUDA_CHECK(cudaMemcpy(&next_size, d_next_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if (next_size > next_cap) next_size = next_cap;

        std::vector<uint32_t> next;
        if (next_size) {
            next.resize(next_size);
            CUDA_CHECK(cudaMemcpy(next.data(), d_next_frontier,
                                  (size_t)next_size * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            std::sort(next.begin(), next.end());
            next.erase(std::unique(next.begin(), next.end()), next.end());
        }

        if (!mm.hasSingleTLAS()) {
            if (d_tlas_mem)       { CUDA_CHECK(cudaFree((void*)d_tlas_mem)); d_tlas_mem = 0; }
            if (d_instance_bases) { CUDA_CHECK(cudaFree((void*)d_instance_bases)); d_instance_bases = 0; }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double level_ms_e2e = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms_e2e += level_ms_e2e;
/*
        std::cout << "[SSSP] level " << level
                  << " frontier=" << frontier.size()
                  << " next=" << (next_size ? next.size() : 0)
                  << " | gpu=" << gpu_ms_level << " ms"
                  << " | e2e=" << level_ms_e2e << " ms\n";
*/
        if (next.empty()) break;
        frontier.swap(next);
        ++level;
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    std::vector<float> hdist(N);
    CUDA_CHECK(cudaMemcpy(hdist.data(), d_sssp, N*sizeof(float), cudaMemcpyDeviceToHost));

    mm.bvh_build_ms = total_ms_e2e;

    {
        std::cout << "Source vertex: " << source << std::endl;
    
        int printed = 0;
        for (int i = 0; i < N && printed < 10; ++i) {
            float d = hdist[i];
            if (std::isinf(d))
                std::cout << "[" << i << ":inf] ";
            else
                std::cout << "[" << i << ":" << std::fixed << std::setprecision(6) << d << "] ";
            ++printed;
        }
        if (printed < N) std::cout << "...";
        std::cout << std::endl;
    
        std::cout << "Total Jobs: " << total_jobs << std::endl;
        std::cout << "Traversal Time (ms): " << total_ms_e2e << std::endl;
    }

    // cleanup
    cudaFree(d_sssp);
    cudaFree(d_next_frontier);
    cudaFree(d_next_count);
    cudaFree(d_jobs);

    std::cout << "  Total jobs: " << total_jobs << std::endl;
    return hdist;
}

void run_sssp_hybrid(OptixPipeline pipe,
                            const OptixShaderBindingTable& sbt,
                            CUdeviceptr d_params,
                            const Params& baseParams,
                            int source,
                            int num_vertices,
                            GPUMemoryManager& mm,
                            const std::vector<uint32_t>& uasp_first,
                            CUstream stream,
                            uint32_t deg)
{
    const int N = num_vertices;
    if (N <= 0) { mm.bvh_build_ms = 0.0; return; }

    std::vector<uint32_t> h_rowptr(N + 1);
    CUDA_CHECK(cudaMemcpyAsync(h_rowptr.data(), baseParams.row_ptr,
                               (N + 1) * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint32_t> degree(N);
    for (int u = 0; u < N; ++u)
        degree[u] = h_rowptr[u + 1] - h_rowptr[u];

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const double HEADROOM = 0.70;
    const size_t budgetB  = (size_t)((1.0 - HEADROOM) * (double)freeB);
    const size_t fixedOverhead = 1ull << 20;

    auto bounded_cap = [&](size_t bytes_per_item, size_t min_items, size_t max_items)->uint32_t {
        if (bytes_per_item == 0) return (uint32_t)min_items;
        size_t avail = budgetB > fixedOverhead ? (budgetB - fixedOverhead) : budgetB;
        size_t cap   = avail / bytes_per_item;
        cap = std::max(cap, min_items);
        cap = std::min(cap, max_items);
        return (uint32_t)cap;
    };

    const uint32_t next_cap_cuda = bounded_cap(sizeof(uint32_t), 64u * 1024u, (size_t)N);
    const uint32_t next_cap_optx = bounded_cap(sizeof(uint32_t), 64u * 1024u, (size_t)N);
    const uint32_t job_batch_cap = bounded_cap(sizeof(Job), 64u * 1024u, (size_t)N);

    float* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, N * sizeof(float)));

    {
        std::vector<float> hinit(N, INFINITY);
        if (source >= 0 && source < N) hinit[source] = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(d_dist, hinit.data(), N*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    Job* d_jobs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs, (size_t)job_batch_cap * sizeof(Job)));
    std::vector<Job> h_jobs(job_batch_cap);

    uint32_t *d_next_frontier_cuda = nullptr, *d_next_count_cuda = nullptr;
    uint32_t *d_next_frontier_optx = nullptr, *d_next_count_optx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next_frontier_cuda, (size_t)next_cap_cuda * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_count_cuda,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_optx, (size_t)next_cap_optx * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_count_optx,    sizeof(uint32_t)));

    uint32_t* d_frontier_cuda = nullptr;
    const uint32_t frontier_cuda_cap = std::min<uint32_t>(std::max<uint32_t>(64u * 1024u, next_cap_cuda), (uint32_t)N);
    CUDA_CHECK(cudaMalloc(&d_frontier_cuda, (size_t)frontier_cuda_cap * sizeof(uint32_t)));

    Params baseP = baseParams;
    baseP.num_vertices   = (uint32_t)N;
    baseP.sssp_distances = d_dist;

    CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
    OptixTraversableHandle tlas_single = 0;
    uint32_t num_instances_single = 0;
    if (mm.hasSingleTLAS()) {
        mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);
        if (tlas_single == 0) {
            std::cerr << "ERROR: SSSP Hybrid: Single TLAS invalid.\n";
            return;
        }
    }

    
    uint32_t DEG_CUDA = 0; 
    if(deg == 0) {
      DEG_CUDA = 32;
    } else {
      DEG_CUDA = deg;
    }

    std::vector<uint32_t> frontier{ (uint32_t)source };

    uint64_t total_cuda_jobs = 0;
    uint64_t total_optix_jobs = 0;

    double total_kernel_ms = 0.0;
    cudaEvent_t startE, stopE;
    CUDA_CHECK(cudaEventCreate(&startE));
    CUDA_CHECK(cudaEventCreate(&stopE));

    auto zero_device_u32 = [&](uint32_t* ptr){
        const uint32_t z = 0;
        CUDA_CHECK(cudaMemcpyAsync(ptr, &z, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    };

    int level = 0;
    while (!frontier.empty()) {

        std::vector<uint32_t> frontier_cuda;
        std::vector<uint32_t> frontier_optx;
        frontier_cuda.reserve(frontier.size());
        frontier_optx.reserve(frontier.size());

        for (uint32_t u : frontier) {
            if (degree[u] >= DEG_CUDA)
                frontier_cuda.push_back(u);
            else
                frontier_optx.push_back(u);
        }

        total_cuda_jobs  += frontier_cuda.size();
        total_optix_jobs += frontier_optx.size();

        zero_device_u32(d_next_count_cuda);
        zero_device_u32(d_next_count_optx);

        float iter_kernel_ms = 0.0f;

        
        if (!frontier_optx.empty()) {
            CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
            OptixTraversableHandle tlas = 0;
            uint32_t num_instances = 0;

            if (mm.hasSingleTLAS()) {
                tlas = tlas_single;
                d_instance_bases = d_instance_bases_single;
                d_tlas_mem = d_tlas_mem_single;
                num_instances = num_instances_single;
            } else {
                prepare_tlas_for_frontier(mm, frontier_optx, uasp_first,
                                          &tlas, &d_tlas_mem, &d_instance_bases, &num_instances, stream);
            }

            const uint32_t totalJobs = (uint32_t)frontier_optx.size();
            for (uint32_t off = 0; off < totalJobs; off += job_batch_cap) {
                const uint32_t thisBatch = std::min(job_batch_cap, totalJobs - off);

                for (uint32_t i = 0; i < thisBatch; ++i)
                    h_jobs[i] = { SSSP, frontier_optx[off + i], 0u, 0u, 0u, 0u, 0.0f };

                CUDA_CHECK(cudaMemcpyAsync(d_jobs, h_jobs.data(),
                                           (size_t)thisBatch * sizeof(Job),
                                           cudaMemcpyHostToDevice, stream));

                Params iter = baseP;
                iter.tlas                = tlas;
                iter.jobs                = d_jobs;
                iter.num_rays            = thisBatch;
                iter.instance_prim_bases = (const uint32_t*)d_instance_bases;
                iter.num_instances       = num_instances;
                iter.next_frontier       = d_next_frontier_optx;
                iter.next_count          = d_next_count_optx;
                iter.next_capacity       = next_cap_optx;

                CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iter, sizeof(Params),
                                           cudaMemcpyHostToDevice, stream));

                CUDA_CHECK(cudaEventRecord(startE, stream));
                OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, thisBatch, 1, 1));
                CUDA_CHECK(cudaEventRecord(stopE, stream));
                CUDA_CHECK(cudaEventSynchronize(stopE));

                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, startE, stopE));
                iter_kernel_ms += ms;
            }

            if (!mm.hasSingleTLAS()) {
                if (d_tlas_mem)       { CUDA_CHECK(cudaFree((void*)d_tlas_mem)); }
                if (d_instance_bases) { CUDA_CHECK(cudaFree((void*)d_instance_bases)); }
            }
        }

        if (!frontier_cuda.empty()) {
            CUDA_CHECK(cudaEventRecord(startE, stream));
            for (size_t off = 0; off < frontier_cuda.size(); off += frontier_cuda_cap) {
                const uint32_t thisChunk = (uint32_t)std::min<size_t>(frontier_cuda_cap, frontier_cuda.size() - off);

                CUDA_CHECK(cudaMemcpyAsync(d_frontier_cuda, frontier_cuda.data() + off,
                                           (size_t)thisChunk * sizeof(uint32_t),
                                           cudaMemcpyHostToDevice, stream));

                                           CUDA_CHECK(cudaEventRecord(startE, stream));
                launch_sssp_relax_frontier(
                    reinterpret_cast<const uint32_t*>(baseParams.row_ptr),
                    reinterpret_cast<const uint32_t*>(baseParams.nbrs),
                    d_frontier_cuda,
                    thisChunk,
                    d_dist,
                    d_next_frontier_cuda,
                    d_next_count_cuda,
                    stream
                );
            }
            CUDA_CHECK(cudaEventRecord(stopE, stream));
            CUDA_CHECK(cudaEventSynchronize(stopE));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, startE, stopE));
            iter_kernel_ms += ms;
        }

        total_kernel_ms += iter_kernel_ms;

        uint32_t next_cuda = 0, next_optx = 0;
        CUDA_CHECK(cudaMemcpy(&next_cuda, d_next_count_cuda, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&next_optx, d_next_count_optx, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (next_cuda > next_cap_cuda) next_cuda = next_cap_cuda;
        if (next_optx > next_cap_optx) next_optx = next_cap_optx;

        std::vector<uint32_t> next;
        next.reserve((size_t)next_cuda + (size_t)next_optx);

        if (next_cuda) {
            std::vector<uint32_t> tmp(next_cuda);
            CUDA_CHECK(cudaMemcpy(tmp.data(), d_next_frontier_cuda,
                                  (size_t)next_cuda * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            next.insert(next.end(), tmp.begin(), tmp.end());
        }
        if (next_optx) {
            std::vector<uint32_t> tmp(next_optx);
            CUDA_CHECK(cudaMemcpy(tmp.data(), d_next_frontier_optx,
                                  (size_t)next_optx * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            next.insert(next.end(), tmp.begin(), tmp.end());
        }

        if (!next.empty()) {
            std::sort(next.begin(), next.end());
            next.erase(std::unique(next.begin(), next.end()), next.end());
        }

        if (next.empty()) break;
        frontier.swap(next);
        ++level;
    }

    CUDA_CHECK(cudaEventDestroy(startE));
    CUDA_CHECK(cudaEventDestroy(stopE));

    std::vector<float> hdist(N);
    CUDA_CHECK(cudaMemcpy(hdist.data(), d_dist, N*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  Total CUDA jobs " << total_cuda_jobs << "\n";
    std::cout << "  Total OptiX jobs " << total_optix_jobs << "\n";

    mm.bvh_build_ms = total_kernel_ms;

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_jobs));
    CUDA_CHECK(cudaFree(d_frontier_cuda));
    CUDA_CHECK(cudaFree(d_next_frontier_cuda));
    CUDA_CHECK(cudaFree(d_next_count_cuda));
    CUDA_CHECK(cudaFree(d_next_frontier_optx));
    CUDA_CHECK(cudaFree(d_next_count_optx));
}