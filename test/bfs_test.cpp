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
#include "kernels/cuda/algorithms.cuh"
#include "rt/rt_pipeline.hpp"

static inline std::vector<uint32_t> bfs_cpu_ref(
    int N, const std::vector<uint32_t>& row_ptr, const std::vector<uint32_t>& nbrs, int src)
{
    std::vector<uint32_t> dist(N, UINT32_MAX);
    std::vector<uint32_t> q; q.reserve(N);
    dist[src] = 0;
    q.push_back(src);

    size_t head = 0;
    while (head < q.size()) {
        uint32_t u = q[head++];
        uint32_t beg = row_ptr[u], end = row_ptr[u + 1];
        for (uint32_t e = beg; e < end; ++e) {
            uint32_t v = nbrs[e];
            if (dist[v] == UINT32_MAX) {
                dist[v] = dist[u] + 1;
                q.push_back(v);
            }
        }
    }
    return dist;
}

static inline size_t check_bfs_result(
    const std::vector<uint32_t>& got, const std::vector<uint32_t>& exp, size_t max_show = 8)
{
    if (got.size() != exp.size()) {
        std::cerr << "Size mismatch: got " << got.size() << " vs exp " << exp.size() << "\n";
        return std::max(got.size(), exp.size());
    }
    size_t mism = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != exp[i]) {
            if (mism < max_show)
                std::cerr << "  i=" << i << " got=" << got[i] << " exp=" << exp[i] << "\n";
            ++mism;
        }
    }
    if (mism)
        std::cerr << "Total mismatches: " << mism << "/" << got.size() << "\n";
    return mism;
}

struct RunStats {
    double ms = 0.0;
    double teps = 0.0; // traversed edges per second
};

static inline RunStats time_bfs_gpu(
    std::function<std::vector<uint32_t>()> runner, size_t edges, int warmup = 1, int reps = 5)
{
    for (int i = 0; i < warmup; ++i) (void)runner(); // warmup

    double total_ms = 0.0;
    std::vector<uint32_t> last;

    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        last = runner();
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    RunStats s;
    s.ms = total_ms / std::max(1, reps);
    s.teps = (s.ms > 0.0) ? (double)edges / (s.ms * 1e-3) : 0.0;
    return s;
}

int main()
{
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int SRC = 0;
    uint32_t MAX_SEG_LEN = 1024;
    int num_partitions = 0;
    int num_dummy_nodes = 0;

    std::cout << "Config: MAX_SEG=" << MAX_SEG_LEN
              << " PARTS=" << num_partitions
              << " DUMMIES=" << num_dummy_nodes << '\n';

    // --- Load graph ---
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
    const size_t nbrs_bytes    = nbrs.size() * sizeof(uint32_t);
    const size_t wts_bytes     = wts.size() * sizeof(float);

    std::cout << "Nodes: " << N << '\n'
              << "Edges: " << nbrs.size() << '\n'
              << "Graph size: " << toMB(row_ptr_bytes + nbrs_bytes + wts_bytes) << " MB\n";

    // --- GPU memory check ---
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU mem: free=" << toGB(free_mem)
              << " GB / total=" << toGB(total_mem) << " GB\n";

    const size_t graph_bytes = row_ptr_bytes + nbrs_bytes + wts_bytes;
    const bool use_gpu_memory = (graph_bytes < static_cast<size_t>(free_mem * 0.8));

    std::cout << "[MM] " << (use_gpu_memory
        ? "Enough GPU memory, copying graph to device"
        : "Not enough GPU memory, registering host memory") << "...\n";

    // --- Allocate or register ---
    uint32_t *d_row_ptr = nullptr, *d_nbrs = nullptr;
    float    *d_wts     = nullptr;

    cudaStream_t streamCompute, streamTransfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamCompute,  cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamTransfer, cudaStreamNonBlocking));

    cudaEvent_t h2dDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone, cudaEventDisableTiming));

    if (use_gpu_memory) {
        PinnedRegister pin_row(row_ptr.data(), row_ptr_bytes);
        PinnedRegister pin_nbr(nbrs.data(),    nbrs_bytes);
        PinnedRegister pin_wts(wts.data(),     wts_bytes);

        CUDA_CHECK(cudaMalloc(&d_row_ptr, row_ptr_bytes));
        CUDA_CHECK(cudaMalloc(&d_nbrs,    nbrs_bytes));
        CUDA_CHECK(cudaMalloc(&d_wts,     wts_bytes));

        CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, row_ptr.data(), row_ptr_bytes,
                                   cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaMemcpyAsync(d_nbrs, nbrs.data(), nbrs_bytes,
                                   cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaMemcpyAsync(d_wts, wts.data(), wts_bytes,
                                   cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaEventRecord(h2dDone, streamTransfer));
    } else {
        if (!row_ptr.empty())
            CUDA_CHECK(cudaHostRegister(row_ptr.data(), row_ptr_bytes,
                                        cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!nbrs.empty())
            CUDA_CHECK(cudaHostRegister(nbrs.data(), nbrs_bytes,
                                        cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!wts.empty())
            CUDA_CHECK(cudaHostRegister(wts.data(), wts_bytes,
                                        cudaHostRegisterMapped | cudaHostRegisterPortable));

        if (!row_ptr.empty()) CUDA_CHECK(cudaHostGetDevicePointer(&d_row_ptr, row_ptr.data(), 0));
        if (!nbrs.empty())    CUDA_CHECK(cudaHostGetDevicePointer(&d_nbrs,    nbrs.data(),    0));
        if (!wts.empty())     CUDA_CHECK(cudaHostGetDevicePointer(&d_wts,     wts.data(),     0));
        CUDA_CHECK(cudaEventRecord(h2dDone, streamTransfer));
    }

    std::cout << "[MM] Graph memory prepared.\n";

    // --- OptiX setup ---
    auto rt_pipe = std::make_shared<rt_pipeline>();
    auto ctx      = rt_pipe->get_context();
    auto module   = rt_pipe->get_module();
    auto pipeline = rt_pipe->get_pipeline();
    auto sbt      = rt_pipe->get_sbt();

    CUdeviceptr d_params = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));
    Params base{};
    base.num_vertices = static_cast<uint32_t>(N);
    base.row_ptr = d_row_ptr;
    base.nbrs    = d_nbrs;
    base.weights = d_wts;

    // --- Build UASP/AABBs ---
    graph->build_uasps(MAX_SEG_LEN);
    graph->build_aabbs();
    graph->append_dummy_aabbs_tagged(num_dummy_nodes);

    std::vector<uint32_t>& uasp_first = graph->get_uasp_first();
    std::vector<uint32_t>& uasp_count = graph->get_uasp_count();
    std::vector<UASP>&     uasps_host = graph->get_uasp_host();
    std::vector<float>&    aabbs6     = graph->get_aabb();
    std::vector<uint8_t>&  aabb_mask  = graph->get_mask();

    DeviceBuffer<uint8_t>  d_aabb_mask(aabb_mask.size());
    DeviceBuffer<UASP>     d_uasps(uasps_host.size());
    DeviceBuffer<float>    d_aabbs(aabbs6.size());
    DeviceBuffer<uint32_t> d_uasp_first(uasp_first.size());
    DeviceBuffer<uint32_t> d_uasp_count(uasp_count.size());

    d_aabb_mask.uploadAsync(aabb_mask.data(), aabb_mask.size(), streamTransfer);
    d_uasps.uploadAsync(uasps_host.data(), uasps_host.size(), streamTransfer);
    d_aabbs.uploadAsync(aabbs6.data(), aabbs6.size(), streamTransfer);
    d_uasp_first.uploadAsync(uasp_first.data(), uasp_first.size(), streamTransfer);
    d_uasp_count.uploadAsync(uasp_count.data(), uasp_count.size(), streamTransfer);

    base.aabb_mask  = (const uint8_t*)d_aabb_mask.ptr;
    base.uasps      = (const UASP*)d_uasps.ptr;
    base.aabbs      = (const float*)d_aabbs.ptr;
    base.num_uasps  = (uint32_t)uasps_host.size();
    base.num_aabbs  = (uint32_t)(aabbs6.size() / 6);
    base.uasp_first = (const uint32_t*)d_uasp_first.ptr;
    base.uasp_count = (const uint32_t*)d_uasp_count.ptr;

    GPUMemoryManager mm(ctx, uasps_host, aabbs6, (uint32_t)uasps_host.size(),
                        0.85f, num_partitions, &aabb_mask);

    CUDA_CHECK(cudaStreamWaitEvent(streamCompute, h2dDone, 0));
    CUDA_CHECK(cudaEventDestroy(h2dDone));

    std::cout << "<------ RUN BFS ------>\n";

    // --- CPU reference ---
    auto dist_ref = bfs_cpu_ref(N, row_ptr, nbrs, SRC);

    // --- GPU BFS runner ---
    auto runner = [&]() -> std::vector<uint32_t> {
        auto out = graph->bfs(rt_pipe, mm, d_params, base, SRC, N, streamCompute);
        CUDA_CHECK(cudaStreamSynchronize(streamCompute));
        return out;
    };

    // correctness check
    auto dist_gpu = runner();
    size_t mism = check_bfs_result(dist_gpu, dist_ref);
    if (mism == 0)
        std::cout << "BFS correctness: OK\n";
    else
        std::cerr << "BFS correctness: FAILED\n";

    // performance
    RunStats stats = time_bfs_gpu(runner, nbrs.size(), 1, 5);
    std::cout << "BFS avg time: " << stats.ms << " ms, TEPS: " << stats.teps << "\n";

    // print first few results
    std::cout << "Distances (first 16): ";
    for (int i = 0; i < std::min(N, 16); ++i)
        std::cout << "[" << i << ":" << dist_gpu[i] << "] ";
    std::cout << "\n";

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
