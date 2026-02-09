#include "cdlp.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "common.hpp"
#include "partition.hpp"
#include "kernels/cuda/algorithms.cuh"

#ifndef DEBUG_CDLP
#define DEBUG_CDLP 0
#endif

static void sync_check(CUstream s, const char* where) {
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));
#if DEBUG_CDLP
    std::cout << "[SYNC] " << where << "\n";
#endif
}

std::vector<uint32_t> run_cdlp_optix(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream stream,
    int max_iters)
{
    using clk = std::chrono::high_resolution_clock;
    const auto wall_t0 = clk::now();

    const int N = num_vertices;
    if (N <= 0) { mm.bvh_build_ms = 0.0; return {}; }
    if (max_iters <= 0) max_iters = 1;

    cudaEvent_t startEvt{}, stopEvt{};
    CUDA_CHECK(cudaEventCreate(&startEvt));
    CUDA_CHECK(cudaEventCreate(&stopEvt));

    uint32_t* d_labels_a = nullptr; CUDA_CHECK(cudaMalloc(&d_labels_a, (size_t)N * sizeof(uint32_t)));
    uint32_t* d_labels_b = nullptr; CUDA_CHECK(cudaMalloc(&d_labels_b, (size_t)N * sizeof(uint32_t)));
    uint32_t* d_changed  = nullptr; CUDA_CHECK(cudaMalloc(&d_changed,  sizeof(uint32_t)));

    uint32_t* d_nodes = nullptr; CUDA_CHECK(cudaMalloc(&d_nodes, (size_t)N * sizeof(uint32_t)));
    launch_iota_u32(d_nodes, (uint32_t)N, stream);
    launch_cdlp_init_labels(d_labels_a, (uint32_t)N, stream);
    launch_copy_u32(d_labels_b, d_labels_a, (uint32_t)N, stream);
    sync_check(stream, "init labels/nodes");

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const double HEADROOM = 0.35;
    size_t budgetB = (size_t)((1.0 - HEADROOM) * (double)freeB);
    if (budgetB < (32ull << 20)) budgetB = (32ull << 20);

    const size_t jobBytes = sizeof(Job);
    const size_t perItemBytes = jobBytes;
    const size_t fixedOverhead = 1ull << 20;

    uint32_t job_cap = 0;
    if (budgetB > fixedOverhead && perItemBytes > 0) {
        size_t avail = (budgetB - fixedOverhead) / perItemBytes;
        job_cap = (uint32_t)std::min<size_t>(std::max<size_t>(avail, 64ull * 1024ull), (size_t)N);
    } else {
        job_cap = std::min<uint32_t>((uint32_t)N, 64u * 1024u);
    }

    Job* d_jobs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs, (size_t)job_cap * sizeof(Job)));

    CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
    OptixTraversableHandle tlas_single = 0;
    uint32_t num_instances_single = 0;
    const bool has_single_tlas = mm.hasSingleTLAS();
    if (has_single_tlas)
        mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);

    Params hparams = baseParams;
    hparams.num_vertices  = (uint32_t)N;
    hparams.cdlp_labels_curr = d_labels_a;
    hparams.cdlp_labels_next = d_labels_b;
    hparams.cdlp_changed     = d_changed;

    uint64_t total_jobs = 0;
    uint32_t optix_launches = 0;

    CUDA_CHECK(cudaEventRecord(startEvt, stream));

    for (int it = 0; it < max_iters; ++it) {
        launch_set_u32(d_changed, 0u, stream);
        sync_check(stream, "zero changed");

        uint32_t remaining = (uint32_t)N;
        uint32_t offset = 0;

        while (remaining) {
            const uint32_t batch = std::min<uint32_t>(remaining, job_cap);

            OptixTraversableHandle tlas = tlas_single;
            CUdeviceptr d_instance_bases = d_instance_bases_single;
            uint32_t num_instances = num_instances_single;

            launch_build_jobs_from_nodes(d_nodes + offset, batch, d_jobs, CDLP, stream);
            sync_check(stream, "build jobs cdlp");

            Params iterP = hparams;
            iterP.tlas                = tlas;
            iterP.instance_prim_bases = (const uint32_t*)d_instance_bases;
            iterP.num_instances       = num_instances;
            iterP.jobs                = d_jobs;
            iterP.num_rays            = batch;

            CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iterP, sizeof(Params), cudaMemcpyHostToDevice, stream));

            OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, batch, 1, 1));
            ++optix_launches;
            total_jobs += batch;

            sync_check(stream, "optix cdlp launch");

            offset += batch;
            remaining -= batch;
        }

        uint32_t h_changed = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_changed, d_changed, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        sync_check(stream, "read changed");

#if DEBUG_CDLP
        std::cout << "[CDLP] iter=" << it << " changed=" << h_changed << "\n";
#endif

        std::swap(d_labels_a, d_labels_b);
        hparams.cdlp_labels_curr = d_labels_a;
        hparams.cdlp_labels_next = d_labels_b;

        if (h_changed == 0u) break;
    }

    CUDA_CHECK(cudaEventRecord(stopEvt, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvt));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, startEvt, stopEvt));

    const auto wall_t1 = clk::now();
    const double e2e_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
    mm.bvh_build_ms = ms;

    std::vector<uint32_t> h_labels((size_t)N);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels_a, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    sync_check(stream, "copy labels back");

    //std::cout << "CDLP (Label Propagation) results\n";
    std::cout << "  Total jobs: " << total_jobs << "\n";
    std::cout << "  OptiX launches: " << optix_launches << "\n";
    std::cout << "  Traversal time (ms): " << ms << "\n";
    {
        int printed = 0;
        std::cout << "  labels[0:10] ";
        for (int i = 0; i < N && printed < 10; ++i, ++printed)
            std::cout << "[" << i << ":" << h_labels[i] << "] ";
        if (N > 10) std::cout << "...";
        std::cout << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(startEvt));
    CUDA_CHECK(cudaEventDestroy(stopEvt));
    cudaFree(d_nodes);
    cudaFree(d_jobs);
    cudaFree(d_changed);
    cudaFree(d_labels_a);
    cudaFree(d_labels_b);

    return h_labels;
}

std::vector<uint32_t> run_cdlp_hybrid(
    OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream stream,
    int max_iters,
    uint32_t thres)
{
    using clk = std::chrono::high_resolution_clock;
    const auto wall_t0 = clk::now();

    const int N = num_vertices;
    if (N <= 0) { mm.bvh_build_ms = 0.0; return {}; }
    if (max_iters <= 0) max_iters = 1;

    const uint32_t batch_threshold = (thres == 0u) ? 100000u : thres;

    cudaEvent_t devStart{}, devStop{};
    CUDA_CHECK(cudaEventCreate(&devStart));
    CUDA_CHECK(cudaEventCreate(&devStop));

    uint32_t* d_labels_a = nullptr; CUDA_CHECK(cudaMalloc(&d_labels_a, (size_t)N * sizeof(uint32_t)));
    uint32_t* d_labels_b = nullptr; CUDA_CHECK(cudaMalloc(&d_labels_b, (size_t)N * sizeof(uint32_t)));
    uint32_t* d_changed  = nullptr; CUDA_CHECK(cudaMalloc(&d_changed,  sizeof(uint32_t)));

    uint32_t* d_nodes = nullptr; CUDA_CHECK(cudaMalloc(&d_nodes, (size_t)N * sizeof(uint32_t)));
    launch_iota_u32(d_nodes, (uint32_t)N, stream);
    launch_cdlp_init_labels(d_labels_a, (uint32_t)N, stream);
    launch_copy_u32(d_labels_b, d_labels_a, (uint32_t)N, stream);
    sync_check(stream, "init labels/nodes");

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const double HEADROOM = 0.35;
    size_t budgetB = (size_t)((1.0 - HEADROOM) * (double)freeB);
    if (budgetB < (32ull << 20)) budgetB = (32ull << 20);

    const size_t jobBytes = sizeof(Job);
    const size_t fixedOverhead = 1ull << 20;
    uint32_t job_cap = 0;
    if (budgetB > fixedOverhead && jobBytes > 0) {
        size_t avail = (budgetB - fixedOverhead) / jobBytes;
        job_cap = (uint32_t)std::min<size_t>(std::max<size_t>(avail, 64ull * 1024ull), (size_t)N);
    } else {
        job_cap = std::min<uint32_t>((uint32_t)N, 64u * 1024u);
    }

    Job* d_jobs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs, (size_t)job_cap * sizeof(Job)));

    CUdeviceptr d_instance_bases_single = 0, d_tlas_mem_single = 0;
    OptixTraversableHandle tlas_single = 0;
    uint32_t num_instances_single = 0;
    const bool has_single_tlas = mm.hasSingleTLAS();
    if (has_single_tlas)
        mm.getSingleTLAS(&tlas_single, &d_tlas_mem_single, &d_instance_bases_single, &num_instances_single);

    Params hparams = baseParams;
    hparams.num_vertices  = (uint32_t)N;
    hparams.cdlp_labels_curr = d_labels_a;
    hparams.cdlp_labels_next = d_labels_b;
    hparams.cdlp_changed     = d_changed;

    uint64_t total_cuda_jobs = 0;
    uint64_t total_optix_jobs = 0;
    uint32_t optix_launches = 0;

    CUDA_CHECK(cudaEventRecord(devStart, stream));

    for (int it = 0; it < max_iters; ++it) {
        launch_set_u32(d_changed, 0u, stream);
        sync_check(stream, "zero changed");

        uint32_t remaining = (uint32_t)N;
        uint32_t offset = 0;

        while (remaining) {
            const uint32_t batch = std::min<uint32_t>(remaining, job_cap);

            if (batch >= batch_threshold) {
                total_cuda_jobs += batch;

                launch_cdlp_iterate_nodes(
                    d_nodes + offset, batch,
                    baseParams.row_ptr, baseParams.nbrs,
                    d_labels_a, d_labels_b,
                    d_changed, stream);

                sync_check(stream, "cuda cdlp batch");
            } else {
                total_optix_jobs += batch;

                OptixTraversableHandle tlas = tlas_single;
                CUdeviceptr d_instance_bases = d_instance_bases_single;
                uint32_t num_instances = num_instances_single;

                launch_build_jobs_from_nodes(d_nodes + offset, batch, d_jobs, CDLP, stream);
                sync_check(stream, "build jobs cdlp");

                Params iterP = hparams;
                iterP.tlas                = tlas;
                iterP.instance_prim_bases = (const uint32_t*)d_instance_bases;
                iterP.num_instances       = num_instances;
                iterP.jobs                = d_jobs;
                iterP.num_rays            = batch;

                CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &iterP, sizeof(Params), cudaMemcpyHostToDevice, stream));

                OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(Params), &sbt, batch, 1, 1));
                ++optix_launches;
                sync_check(stream, "optix cdlp batch");
            }

            offset += batch;
            remaining -= batch;
        }

        uint32_t h_changed = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_changed, d_changed, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        sync_check(stream, "read changed");

#if DEBUG_CDLP
        std::cout << "[CDLP-H] iter=" << it << " changed=" << h_changed << "\n";
#endif
        std::swap(d_labels_a, d_labels_b);
        hparams.cdlp_labels_curr = d_labels_a;
        hparams.cdlp_labels_next = d_labels_b;

        if (h_changed == 0u) break;
    }

    CUDA_CHECK(cudaEventRecord(devStop, stream));
    CUDA_CHECK(cudaEventSynchronize(devStop));
    float device_active_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&device_active_ms, devStart, devStop));

    const auto wall_t1 = clk::now();
    const double e2e_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
    mm.bvh_build_ms = device_active_ms;

    std::vector<uint32_t> h_labels((size_t)N);
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels_a, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    sync_check(stream, "copy labels back");

    //std::cout << "CDLP Hybrid results\n";
    std::cout << "  Total CUDA jobs: " << total_cuda_jobs << "\n";
    std::cout << "  Total OptiX jobs: " << total_optix_jobs << "\n";
    //std::cout << "  OptiX launches: " << optix_launches << "\n";
    std::cout << "  Device active time (ms): " << device_active_ms << "\n";

    CUDA_CHECK(cudaEventDestroy(devStart));
    CUDA_CHECK(cudaEventDestroy(devStop));
    cudaFree(d_nodes);
    cudaFree(d_jobs);
    cudaFree(d_changed);
    cudaFree(d_labels_a);
    cudaFree(d_labels_b);

    return h_labels;
}
