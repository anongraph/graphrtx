#include "wcc.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "shared.h"
#include "common.hpp"
#include "partition.hpp"
#include "kernels/cuda/algorithms.cuh"

#ifndef DEBUG_WCC
#define DEBUG_WCC 0
#endif

std::vector<uint32_t> run_wcc_optix(
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
#if DEBUG_WCC
        std::cout << "[SYNC] " << where << "\n";
#endif
    };

    cudaEvent_t startEvt{}, stopEvt{};
    CUDA_CHECK(cudaEventCreate(&startEvt));
    CUDA_CHECK(cudaEventCreate(&stopEvt));

    uint32_t* d_comp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_comp, (size_t)N * sizeof(uint32_t)));

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const double HEADROOM = 0.35;
    size_t budgetB = static_cast<size_t>((1.0 - HEADROOM) * (double)freeB);
    if (budgetB < (32ull << 20)) budgetB = (32ull << 20);

    const size_t jobBytes = sizeof(Job);
    const size_t perItemBytes = sizeof(uint32_t) * 2 + jobBytes;
    const size_t fixedOverhead = 1ull << 20;

    uint32_t frontier_cap = 0;
    if (budgetB > fixedOverhead && perItemBytes > 0) {
        frontier_cap = (uint32_t)std::min<size_t>(
            std::max<size_t>((budgetB - fixedOverhead) / perItemBytes, 64ull * 1024ull),
            (size_t)N);
    } else {
        frontier_cap = std::min<uint32_t>(N, 64u * 1024u);
    }

#if DEBUG_WCC
    std::cout << "[DBG] N=" << N << " frontier_cap=" << frontier_cap
              << " free=" << (double)freeB/1e9 << "GB total=" << (double)totalB/1e9 << "GB\n";
#endif

    uint32_t* d_frontier      = nullptr; CUDA_CHECK(cudaMalloc(&d_frontier,      (size_t)frontier_cap * sizeof(uint32_t)));
    uint32_t* d_next_frontier = nullptr; CUDA_CHECK(cudaMalloc(&d_next_frontier, (size_t)frontier_cap * sizeof(uint32_t)));
    uint32_t* d_next_count    = nullptr; CUDA_CHECK(cudaMalloc(&d_next_count,    sizeof(uint32_t)));
    Job*      d_jobs          = nullptr; CUDA_CHECK(cudaMalloc(&d_jobs,          (size_t)frontier_cap * sizeof(Job)));

    CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
    OptixTraversableHandle tlas_single = 0;
    uint32_t num_instances_single = 0;
    const bool has_single_tlas = mm.hasSingleTLAS();
    if (has_single_tlas)
        mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);

    Params hparams = baseParams;
    hparams.num_vertices   = (uint32_t)N;
    hparams.next_capacity  = frontier_cap;
    hparams.next_frontier  = d_next_frontier;
    hparams.next_count     = d_next_count;
    hparams.wcc_comp       = d_comp;

    launch_memset_u32(d_comp, 0xFFFFFFFFu, (uint32_t)N, streamOptix);
    SAFE_SYNC("init comp");

    uint64_t total_jobs = 0;
    uint32_t total_optix_launches = 0;
    uint32_t num_components = 0;

    uint32_t* d_next_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next_src, sizeof(uint32_t)));
    float ms = 0.f;
    
    for (;;) {
        
        launch_find_next_unassigned_u32(d_comp, (uint32_t)N, d_next_src, streamOptix);

        uint32_t src = 0xFFFFFFFFu;
        CUDA_CHECK(cudaMemcpyAsync(&src, d_next_src, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, streamOptix));
        SAFE_SYNC("read next_src");

        if (src == 0xFFFFFFFFu) break; 

        const uint32_t comp_id = num_components++;
#if DEBUG_WCC
        std::cout << "[WCC] start comp=" << comp_id << " src=" << src << "\n";
#endif

        {
            launch_wcc_set_seed(d_comp, src, comp_id, streamOptix);
            SAFE_SYNC("seed comp[src]");
        }
        
        CUDA_CHECK(cudaMemcpyAsync(d_frontier, &src, sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, streamOptix));
        SAFE_SYNC("init frontier");

        uint32_t frontier_size = 1u;
        
        while (frontier_size > 0u) {
            total_jobs += frontier_size;

            uint32_t z = 0;
            CUDA_CHECK(cudaMemcpyAsync(d_next_count, &z, sizeof(uint32_t),
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
                std::vector<uint32_t> curFront(frontier_size);
                CUDA_CHECK(cudaMemcpy(curFront.data(), d_frontier,
                                      (size_t)frontier_size * sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost));
                SAFE_SYNC("copy frontier to host (tlas prep)");
                prepare_tlas_for_frontier(mm, curFront, uasp_first,
                                          &tlas, &d_tlas_mem,
                                          &d_instance_bases, &num_instances, streamOptix);
            }

            launch_build_jobs_from_nodes(d_frontier, frontier_size, d_jobs, WCC, streamOptix);
            SAFE_SYNC("build jobs wcc");

            Params iter = hparams;
            iter.tlas                = tlas;
            iter.instance_prim_bases = (const uint32_t*)d_instance_bases;
            iter.num_instances       = num_instances;
            iter.jobs                = d_jobs;
            iter.num_rays            = frontier_size;
            iter.wcc_current_comp    = comp_id;
            iter.next_frontier       = d_next_frontier;
            iter.next_count          = d_next_count;

            CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iter, sizeof(Params),
                                       cudaMemcpyHostToDevice, streamOptix));
            
            CUDA_CHECK(cudaEventRecord(startEvt, streamOptix));
            OPTIX_CHECK(optixLaunch(pipe, streamOptix, d_params, sizeof(Params), &sbt,
                                    frontier_size, 1, 1));
            CUDA_CHECK(cudaEventRecord(stopEvt, streamOptix));
            CUDA_CHECK(cudaEventSynchronize(stopEvt));
            float mst = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&mst, startEvt, stopEvt));
            ms += mst;

            ++total_optix_launches;
            SAFE_SYNC("optix wcc launch");

            uint32_t next_size = 0;
            CUDA_CHECK(cudaMemcpyAsync(&next_size, d_next_count, sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, streamOptix));
            SAFE_SYNC("read next_size");

            std::swap(d_frontier, d_next_frontier);
            frontier_size = next_size;

            if (!has_single_tlas) {
                if (d_tlas_mem)       cudaFree((void*)d_tlas_mem);
                if (d_instance_bases) cudaFree((void*)d_instance_bases);
            }
        }
    }

    const auto wall_t1 = clk::now();
    const double e2e_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

    std::vector<uint32_t> h_comp((size_t)N);
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_comp, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    SAFE_SYNC("copy comp back");

    mm.bvh_build_ms = ms;

    //std::cout << "WCC results\n";
    //std::cout << "  Components: " << num_components << "\n";
    std::cout << "  Total jobs: " << total_jobs << "\n";
    std::cout << "  OptiX launches: " << total_optix_launches << "\n";
    std::cout << "  Traversal time (ms): " << ms << "\n";

    {
        int printed = 0;
        std::cout << "  comp[0:10] ";
        for (int i = 0; i < N && printed < 10; ++i, ++printed)
            std::cout << "[" << i << ":" << h_comp[i] << "] ";
        if (N > 10) std::cout << "...";
        std::cout << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(startEvt));
    CUDA_CHECK(cudaEventDestroy(stopEvt));
    if (d_next_src) cudaFree(d_next_src);
    cudaFree(d_comp);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_count);
    cudaFree(d_jobs);

    return h_comp;
}

std::vector<uint32_t> run_wcc_hybrid(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream stream,
    uint32_t thres)
{
    using clk = std::chrono::high_resolution_clock;
    const auto wall_t0 = clk::now();

    const int N = num_vertices;
    if (N <= 0) { mm.bvh_build_ms = 0.0; return {}; }

    const uint32_t frontier_threshold = (thres == 0u) ? 100000u : thres;

    auto SAFE_SYNC = [&](const char* where){
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
#if DEBUG_WCC
        std::cout << "[SYNC] " << where << "\n";
#endif
    };

    cudaEvent_t devStart{}, devStop{};
    CUDA_CHECK(cudaEventCreate(&devStart));
    CUDA_CHECK(cudaEventCreate(&devStop));

    uint32_t* d_comp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_comp, (size_t)N * sizeof(uint32_t)));

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
    } else {
        frontier_cap = std::min<uint32_t>(N, 64u * 1024u);
    }

    uint32_t* d_frontier      = nullptr; CUDA_CHECK(cudaMalloc(&d_frontier,      (size_t)frontier_cap * sizeof(uint32_t)));
    uint32_t* d_next_frontier = nullptr; CUDA_CHECK(cudaMalloc(&d_next_frontier, (size_t)frontier_cap * sizeof(uint32_t)));
    uint32_t* d_next_count    = nullptr; CUDA_CHECK(cudaMalloc(&d_next_count,    sizeof(uint32_t)));
    Job*      d_jobs          = nullptr; CUDA_CHECK(cudaMalloc(&d_jobs,          (size_t)frontier_cap * sizeof(Job)));

    CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
    OptixTraversableHandle tlas_single = 0;
    uint32_t num_instances_single = 0;
    const bool has_single_tlas = mm.hasSingleTLAS();
    if (has_single_tlas)
        mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);

    Params hparams = baseParams;
    hparams.num_vertices   = (uint32_t)N;
    hparams.next_capacity  = frontier_cap;
    hparams.wcc_comp       = d_comp;

    hparams.next_frontier  = nullptr;
    hparams.next_count     = d_next_count;

    launch_memset_u32(d_comp, 0xFFFFFFFFu, (uint32_t)N, stream);
    SAFE_SYNC("init comp");

    uint32_t* d_next_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next_src, sizeof(uint32_t)));

    uint64_t total_cuda_jobs = 0;
    uint64_t total_optix_jobs = 0;
    uint32_t total_optix_launches = 0;
    uint32_t num_components = 0;
    float device_active_ms = 0.f;

    auto process_level_in_batches =
        [&](uint32_t comp_id,
            const uint32_t* d_src_frontier,
            uint32_t fsize,
            uint32_t* d_dst_frontier) -> uint32_t
    {
        uint32_t z = 0u;
        CUDA_CHECK(cudaMemcpyAsync(d_next_count, &z, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const uint32_t B = frontier_cap;
        for (uint32_t off = 0; off < fsize; off += B) {
            const uint32_t batch = std::min<uint32_t>(B, fsize - off);

            if (batch >= frontier_threshold) {
                total_cuda_jobs += batch;

                launch_wcc_expand_frontier(
                    d_src_frontier + off,
                    batch,
                    baseParams.row_ptr,
                    baseParams.nbrs,
                    d_comp,
                    comp_id,
                    d_dst_frontier,
                    d_next_count,
                    frontier_cap,
                    stream);

                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                total_optix_jobs += batch;

                CUdeviceptr d_instance_bases = 0, d_tlas_mem = 0;
                OptixTraversableHandle tlas = 0;
                uint32_t num_instances = 0;

                if (has_single_tlas) {
                    tlas = tlas_single;
                    d_instance_bases = d_instance_bases_single;
                    d_tlas_mem = d_tlas_mem_single;
                    num_instances = num_instances_single;
                } else {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    std::vector<uint32_t> curFront(batch);
                    CUDA_CHECK(cudaMemcpy(curFront.data(),
                                          d_src_frontier + off,
                                          (size_t)batch * sizeof(uint32_t),
                                          cudaMemcpyDeviceToHost));
                    prepare_tlas_for_frontier(mm, curFront, uasp_first,
                                              &tlas, &d_tlas_mem, &d_instance_bases, &num_instances, stream);
                }

                launch_build_jobs_from_nodes(d_src_frontier + off, batch, d_jobs, WCC, stream);

                Params iter = hparams;
                iter.tlas                = tlas;
                iter.instance_prim_bases = (const uint32_t*)d_instance_bases;
                iter.num_instances       = num_instances;
                iter.jobs                = d_jobs;
                iter.num_rays            = batch;
                iter.wcc_current_comp    = comp_id;

                iter.next_frontier       = d_dst_frontier;
                iter.next_count          = d_next_count;
                iter.next_capacity       = frontier_cap;

                CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iter, sizeof(Params),
                                           cudaMemcpyHostToDevice, stream));

                CUDA_CHECK(cudaEventRecord(devStart, stream));
                OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, batch, 1, 1));
                ++total_optix_launches;                
                CUDA_CHECK(cudaEventRecord(devStop, stream));
                CUDA_CHECK(cudaEventSynchronize(devStop));
                float mst = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&mst, devStart, devStop));
                device_active_ms += mst;

                CUDA_CHECK(cudaStreamSynchronize(stream));

                if (!has_single_tlas) {
                    if (d_tlas_mem)       CUDA_CHECK(cudaFree((void*)d_tlas_mem));
                    if (d_instance_bases) CUDA_CHECK(cudaFree((void*)d_instance_bases));
                }
            }
        }

        uint32_t produced_total = 0;
        CUDA_CHECK(cudaMemcpyAsync(&produced_total, d_next_count, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (produced_total > frontier_cap) produced_total = frontier_cap;

        return produced_total;
    };

    

    for (;;) {
        launch_find_next_unassigned_u32(d_comp, (uint32_t)N, d_next_src, stream);

        uint32_t src = 0xFFFFFFFFu;
        CUDA_CHECK(cudaMemcpyAsync(&src, d_next_src, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (src == 0xFFFFFFFFu) break;

        const uint32_t comp_id = num_components++;
        launch_wcc_set_seed(d_comp, src, comp_id, stream);
        CUDA_CHECK(cudaMemcpyAsync(d_frontier, &src, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        uint32_t frontier_size = 1u;

        
        while (frontier_size > 0u) {
            const uint32_t next_size =
                process_level_in_batches(comp_id, d_frontier, frontier_size, d_next_frontier);
            std::swap(d_frontier, d_next_frontier);
            frontier_size = next_size;
        }

    }

;

    const auto wall_t1 = clk::now();
    const double e2e_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

    std::vector<uint32_t> h_comp((size_t)N);
    CUDA_CHECK(cudaMemcpy(h_comp.data(), d_comp, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mm.bvh_build_ms = device_active_ms;

    //std::cout << "WCC Hybrid results\n";
    //std::cout << "  Components: " << num_components << "\n";
    std::cout << "  Total CUDA jobs: " << total_cuda_jobs << "\n";
    std::cout << "  Total OptiX jobs: " << total_optix_jobs << "\n";
    //std::cout << "  OptiX launches: " << total_optix_launches << "\n";
    std::cout << "  Device active time (ms): " << device_active_ms << "\n";

    CUDA_CHECK(cudaEventDestroy(devStart));
    CUDA_CHECK(cudaEventDestroy(devStop));
    if (d_next_src) cudaFree(d_next_src);

    cudaFree(d_comp);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_count);
    cudaFree(d_jobs);

    return h_comp;
}
