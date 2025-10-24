#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <chrono>

#include <sstream>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <tuple>
#include <cfloat>
#include <thread>
#include <mutex>
#include <future>
#include <random>

#include <unordered_set>

#include <optix_function_table_definition.h>

#include "shared.h"
#include "memory/gpu_manager.hpp"
#include "memory/buffer.hpp"
#include "memory/register.hpp"

#include "common.hpp"

#include "graph/graph.hpp"

#include "algorithms/partition.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/pr.hpp"
#include "algorithms/bc.hpp"
#include "algorithms/sssp.hpp"
#include "algorithms/tc.hpp"
#include "kernels/cuda/algorithms.cuh"

#include "rt/rt_pipeline.hpp"

int main(int argc, char** argv)
{
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int SRC = 0;          // BFS & SSSP source node
    constexpr int PR_ITERS = 20;     // PageRank iterations
    constexpr float PR_DAMP = 0.85f; // PageRank damping factor
    uint32_t MAX_SEG_LEN = 1024;
    int num_partitions = 0;          // 0 = auto
    int num_dummy_nodes = 0;

    std::cout << "Config: MAX_SEG=" << MAX_SEG_LEN
              << " PARTS=" << num_partitions
              << " DUMMIES=" << num_dummy_nodes << '\n';

    // --- Load graph on host ---
    const std::string file_path = "../data/graph.mtx";

    auto graph = std::make_shared<graph_rtx>();
    const int N = graph->load_mtx_graph(file_path);
    if (N <= 0) {
        std::cerr << "Graph load returned N=" << N << " — nothing to do.\n";
        return 1;
    }

    std::vector<uint32_t>& row_ptr = graph->get_row_ptr();
    std::vector<uint32_t>& nbrs    = graph->get_nbrs_ptr();
    std::vector<float>&    wts     = graph->get_wts();

    const size_t row_ptr_bytes = row_ptr.size() * sizeof(uint32_t);
    const size_t nbrs_bytes    = nbrs.size()    * sizeof(uint32_t);
    const size_t wts_bytes     = wts.size()     * sizeof(float);

    std::cout << "Nodes: " << N << '\n'
              << "Edges: " << nbrs.size() << '\n'
              << "Graph size: " << toMB(row_ptr_bytes + nbrs_bytes + wts_bytes) << " MB\n";

    // --- GPU memory check (safe now; context already has MapHost enabled) ---
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU mem: free=" << toGB(free_mem)
              << " GB / total="    << toGB(total_mem) << " GB\n";

    const size_t graph_bytes = row_ptr_bytes + nbrs_bytes + wts_bytes;
    const bool use_gpu_memory = (graph_bytes < static_cast<size_t>(free_mem * 0.8));

    std::cout << "[MM] " << (use_gpu_memory
        ? "Enough GPU memory, copying graph to device"
        : "Not enough GPU memory, registering host memory") << "...\n";

    // --- Allocate or register graph memory ---
    uint32_t *d_row_ptr = nullptr, *d_nbrs = nullptr;
    float    *d_wts     = nullptr;

    cudaStream_t streamCompute, streamTransfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamCompute,  cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamTransfer, cudaStreamNonBlocking));

    cudaEvent_t h2dDone; // sync point for graph uploads
    CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone, cudaEventDisableTiming));

    if (use_gpu_memory) {
        // Temporary host pinning for faster HtoD copies
        PinnedRegister pin_row(row_ptr.data(), row_ptr_bytes);
        PinnedRegister pin_nbr(nbrs.data(),    nbrs_bytes);
        PinnedRegister pin_wts(wts.data(),     wts_bytes);

        CUDA_CHECK(cudaMalloc(&d_row_ptr, row_ptr_bytes));
        CUDA_CHECK(cudaMalloc(&d_nbrs,    nbrs_bytes));
        CUDA_CHECK(cudaMalloc(&d_wts,     wts_bytes));

        CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, row_ptr.data(), row_ptr_bytes, cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaMemcpyAsync(d_nbrs,    nbrs.data(),    nbrs_bytes,    cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaMemcpyAsync(d_wts,     wts.data(),     wts_bytes,     cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaEventRecord(h2dDone, streamTransfer));
    } else {
        // Guard against zero-sized ranges (cudaHostRegister(size=0) is invalid)
        if (!row_ptr.empty()) CUDA_CHECK(cudaHostRegister(row_ptr.data(), row_ptr_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!nbrs.empty())    CUDA_CHECK(cudaHostRegister(nbrs.data(),    nbrs_bytes,    cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!wts.empty())     CUDA_CHECK(cudaHostRegister(wts.data(),     wts_bytes,     cudaHostRegisterMapped | cudaHostRegisterPortable));

        if (!row_ptr.empty()) CUDA_CHECK(cudaHostGetDevicePointer(&d_row_ptr, row_ptr.data(), 0));
        if (!nbrs.empty())    CUDA_CHECK(cudaHostGetDevicePointer(&d_nbrs,    nbrs.data(),    0));
        if (!wts.empty())     CUDA_CHECK(cudaHostGetDevicePointer(&d_wts,     wts.data(),     0));
        CUDA_CHECK(cudaEventRecord(h2dDone, streamTransfer)); // no copies, but keep the event path
    }

    std::cout << "[MM] Graph memory prepared.\n";

    auto rt_pipe = std::make_shared<rt_pipeline>();
    // OptiX context & pipeline (OptiX will use current CUDA context with MapHost enabled)
    auto ctx      = rt_pipe->get_context();
    auto module   = rt_pipe->get_module();
    auto pipeline = rt_pipe->get_pipeline();
    auto sbt      = rt_pipe->get_sbt();

    // --- Launch params ---
    CUdeviceptr d_params = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));
    Params base{};
    base.num_vertices = static_cast<uint32_t>(N);
    base.row_ptr      = d_row_ptr;   // device mem or mapped host-DEV pointer
    base.nbrs         = d_nbrs;
    base.weights      = d_wts;
    base.num_nbrs     = static_cast<uint32_t>(nbrs.size());

    // --- Build UASP & AABBs (host) ---
    std::vector<uint32_t>& uasp_first = graph->get_uasp_first();
    std::vector<uint32_t>& uasp_count = graph->get_uasp_count();
    std::vector<UASP>&     uasps_host = graph->get_uasp_host();
    std::vector<float>&    aabbs6     = graph->get_aabb();

    auto t0 = std::chrono::high_resolution_clock::now();
    graph->build_uasps(MAX_SEG_LEN);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "UASP build: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    const double total_MB = graph->uasp_total_size();
    std::cout << "Total UASP size: " << total_MB << " MB\n";
    std::cout << "Total UASP count: " << uasps_host.size() << "\n";

    aabbs6.reserve(uasps_host.size() * 6);
    auto t2 = std::chrono::high_resolution_clock::now();
    graph->build_aabbs();
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "AABB build: "
              << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";
    std::cout << "AABB count: " << aabbs6.size() << '\n';

    double aabb_total_MB = graph->aabbs6_total_size();
    std::cout << "Total AABB size: " << aabb_total_MB << " MB\n";

    std::cout << "Create dummies of count: " << num_dummy_nodes << '\n';
    auto t4 = std::chrono::high_resolution_clock::now();
    graph->append_dummy_aabbs_tagged(num_dummy_nodes);
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Dummy build: "
              << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms\n";

    aabb_total_MB = graph->aabbs6_total_size();
    std::cout << "Total AABB size with dummies: " << aabb_total_MB << " MB\n";

    
    std::vector<uint8_t>& aabb_mask = graph->get_mask();

    DeviceBuffer<uint8_t> d_aabb_mask(aabb_mask.size());
    d_aabb_mask.uploadAsync(aabb_mask.data(), aabb_mask.size(), streamTransfer);

    base.aabb_mask = (const uint8_t*)d_aabb_mask.ptr;

    DeviceBuffer<UASP>      d_uasps(uasps_host.size());
    DeviceBuffer<float>     d_aabbs(aabbs6.size());
    DeviceBuffer<uint32_t>  d_uasp_first(uasp_first.size());
    DeviceBuffer<uint32_t>  d_uasp_count(uasp_count.size());

    d_uasps.uploadAsync(uasps_host.data(), uasps_host.size(), streamTransfer);
    d_aabbs.uploadAsync(aabbs6.data(),     aabbs6.size(),     streamTransfer);
    d_uasp_first.uploadAsync(uasp_first.data(), uasp_first.size(), streamTransfer);
    d_uasp_count.uploadAsync(uasp_count.data(), uasp_count.size(), streamTransfer);

    base.uasps       = (const UASP*)d_uasps.ptr;
    base.aabbs       = (const float*)d_aabbs.ptr;
    base.num_uasps   = (uint32_t)uasps_host.size();
    base.num_aabbs   = (uint32_t)(aabbs6.size() / 6);
    base.uasp_first  = (const uint32_t*)d_uasp_first.ptr;
    base.uasp_count  = (const uint32_t*)d_uasp_count.ptr;
    
    // --- Memory manager & streams ---
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    std::cout << "GPU mem: free=" << toGB(freeB)
              << " GB / total="    << toGB(totalB) << " GB\n";

    constexpr float MM_FRACTION = 0.85f;
    GPUMemoryManager mm(ctx, uasps_host, aabbs6, (uint32_t)uasps_host.size(),
                        MM_FRACTION, num_partitions, &aabb_mask);
    base.uasps        = (const UASP*)d_uasps.ptr;
    base.aabbs        = (const float*)d_aabbs.ptr;
    base.num_uasps    = (uint32_t)uasps_host.size();
    base.num_aabbs    = (uint32_t)(aabbs6.size() / 6);
    base.uasp_first   = (const uint32_t*)d_uasp_first.ptr;
    base.uasp_count   = (const uint32_t*)d_uasp_count.ptr;
    base.aabb_mask    = (const uint8_t*)d_aabb_mask.ptr;
                        
    // Ensure compute waits for transfers if any
    CUDA_CHECK(cudaStreamWaitEvent(streamCompute, h2dDone, 0));
    CUDA_CHECK(cudaEventDestroy(h2dDone));

    auto prefetch_next_partition = [&](uint32_t){ /* no-op for AABB path */ };

    std::cout << "<------ RUN BC ------>\n";
    auto res = graph->bc(rt_pipe, mm, d_params, base, N, streamCompute);


    CUDA_CHECK(cudaStreamSynchronize(streamCompute));
    CUDA_CHECK(cudaStreamSynchronize(streamTransfer));

    CUDA_CHECK(cudaStreamDestroy(streamCompute));
    CUDA_CHECK(cudaStreamDestroy(streamTransfer));

    if (mm.hasSingleTLAS()) {
        OptixTraversableHandle t; CUdeviceptr mem = 0, bases = 0; uint32_t n = 0;
        mm.getSingleTLAS(&t, &mem, &bases, &n);
        if (mem)   CUDA_CHECK(cudaFree((void*)mem));
        if (bases) CUDA_CHECK(cudaFree((void*)bases));
    }

    CUDA_CHECK(cudaFree((void*)d_params));

    if (!use_gpu_memory) {
        if (!row_ptr.empty()) CUDA_CHECK(cudaHostUnregister(row_ptr.data()));
        if (!nbrs.empty())    CUDA_CHECK(cudaHostUnregister(nbrs.data()));
        if (!wts.empty())     CUDA_CHECK(cudaHostUnregister(wts.data()));
    } else {
        
        if (d_row_ptr) CUDA_CHECK(cudaFree((void*)d_row_ptr));
        if (d_nbrs)    CUDA_CHECK(cudaFree((void*)d_nbrs));
        if (d_wts)     CUDA_CHECK(cudaFree((void*)d_wts));
    }

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(ctx));
    return 0;
}
