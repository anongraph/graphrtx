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
#include <queue>
#include <limits>
#include <functional>

#include <optix_function_table_definition.h>

#include "shared.h"
#include "memory/gpu_manager.hpp"
#include "memory/buffer.hpp"
#include "memory/register.hpp"
#include "common.hpp"
#include "graph/graph.hpp"
#include "algorithms/partition.hpp"
#include "algorithms/sssp.hpp"
#include "kernels/cuda/algorithms.cuh"
#include "rt/rt_pipeline.hpp"

static inline std::vector<float> sssp_cpu_ref(
    int N,
    const std::vector<uint32_t>& row_ptr,
    const std::vector<uint32_t>& nbrs,
    const std::vector<float>& weights,
    int src)
{
    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> dist(N, INF);
    dist[src] = 0.0f;

    using Node = std::pair<float, uint32_t>; // (distance, node)
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.emplace(0.0f, src);

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;

        uint32_t beg = row_ptr[u];
        uint32_t end = row_ptr[u + 1];
        for (uint32_t e = beg; e < end; ++e) {
            uint32_t v = nbrs[e];
            float w = weights.empty() ? 1.0f : weights[e];
            float nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.emplace(nd, v);
            }
        }
    }
    return dist;
}

static inline void analyze_diff_sssp(const std::vector<float>& got, const std::vector<float>& ref)
{
    if (got.size() != ref.size()) {
        std::cerr << "SSSP size mismatch: got " << got.size() << " vs ref " << ref.size() << "\n";
        return;
    }
    double l1 = 0.0, maxabs = 0.0;
    int count = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float a = got[i], b = ref[i];
        if (std::isinf(a) && std::isinf(b)) continue;
        const double d = std::abs(double(a) - double(b));
        l1 += d;
        if (d > maxabs) maxabs = d;
        ++count;
    }
    std::cout.setf(std::ios::scientific); std::cout.precision(3);
    std::cout << "SSSP diff: L1=" << l1 << ", max|Δ|=" << maxabs << " (" << count << " compared)\n";
    std::cout.unsetf(std::ios::scientific);
}

static inline void print_sssp_firstn(const std::vector<float>& dist, int n = 8)
{
    std::cout << "Distances: ";
    for (int i = 0; i < std::min<int>(n, dist.size()); ++i) {
        if (std::isinf(dist[i])) std::cout << "[" << i << ":INF] ";
        else {
            std::cout.setf(std::ios::fixed); std::cout.precision(3);
            std::cout << "[" << i << ":" << dist[i] << "] ";
            std::cout.unsetf(std::ios::fixed);
        }
    }
    std::cout << "\n";
}

struct RunStats {
    double ms = 0.0;
    double teps = 0.0;
};

static inline RunStats time_sssp_gpu(
    const std::function<std::vector<float>()>& runner,
    size_t edges,
    int warmup = 1,
    int reps = 3)
{
    for (int i = 0; i < warmup; ++i) (void)runner();

    double total_ms = 0.0;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        (void)runner();
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    RunStats s{};
    s.ms = total_ms / std::max(1, reps);
    s.teps = (s.ms > 0.0) ? (double)edges / (s.ms * 1e-3) : 0.0;
    return s;
}

static inline void print_units_teps(double teps) {
    const char* unit = "TEPS";
    double v = teps;
    if (v >= 1e12) { v /= 1e12; unit = "TTEPS"; }
    else if (v >= 1e9) { v /= 1e9; unit = "GTEPS"; }
    else if (v >= 1e6) { v /= 1e6; unit = "MTEPS"; }
    std::cout.setf(std::ios::fixed); std::cout.precision(3);
    std::cout << v << " " << unit;
    std::cout.unsetf(std::ios::fixed);
}

int main(int argc, char** argv)
{
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int SRC = 1;
    uint32_t MAX_SEG_LEN = 1024;
    int num_partitions = 1;
    int num_dummy_nodes = 0;

    std::cout << "Config: MAX_SEG=" << MAX_SEG_LEN
              << " PARTS=" << num_partitions
              << " DUMMIES=" << num_dummy_nodes << '\n';

    // --- Load graph ---
    const std::string file_path = "../data/graph.mtx";
    auto graph = std::make_shared<graph_rtx>();
    const int N = graph->load_mtx_graph(file_path);
    if (N <= 0) {
        std::cerr << "Graph load returned N=" << N << "\n";
        return 1;
    }

    std::vector<uint32_t>& row_ptr = graph->get_row_ptr();
    std::vector<uint32_t>& nbrs    = graph->get_nbrs_ptr();
    std::vector<float>&    wts     = graph->get_wts();

    std::cout << "Nodes: " << N << "\nEdges: " << nbrs.size() << "\n";

    // --- GPU memory setup ---
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const size_t row_ptr_bytes = row_ptr.size() * sizeof(uint32_t);
    const size_t nbrs_bytes    = nbrs.size()    * sizeof(uint32_t);
    const size_t wts_bytes     = wts.size()     * sizeof(float);
    const size_t graph_bytes   = row_ptr_bytes + nbrs_bytes + wts_bytes;
    const bool use_gpu_memory  = (graph_bytes < static_cast<size_t>(free_mem * 0.8));

    uint32_t *d_row_ptr = nullptr, *d_nbrs = nullptr;
    float    *d_wts     = nullptr;

    cudaStream_t streamCompute, streamTransfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamCompute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamTransfer, cudaStreamNonBlocking));

    cudaEvent_t h2dDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone, cudaEventDisableTiming));

    if (use_gpu_memory) {
        CUDA_CHECK(cudaMalloc(&d_row_ptr, row_ptr_bytes));
        CUDA_CHECK(cudaMalloc(&d_nbrs, nbrs_bytes));
        CUDA_CHECK(cudaMalloc(&d_wts, wts_bytes));

        CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, row_ptr.data(), row_ptr_bytes, cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaMemcpyAsync(d_nbrs, nbrs.data(), nbrs_bytes, cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaMemcpyAsync(d_wts, wts.data(), wts_bytes, cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaEventRecord(h2dDone, streamTransfer));
    } else {
        CUDA_CHECK(cudaHostRegister(row_ptr.data(), row_ptr_bytes, cudaHostRegisterMapped));
        CUDA_CHECK(cudaHostRegister(nbrs.data(), nbrs_bytes, cudaHostRegisterMapped));
        CUDA_CHECK(cudaHostRegister(wts.data(), wts_bytes, cudaHostRegisterMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_row_ptr, row_ptr.data(), 0));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_nbrs,    nbrs.data(),    0));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_wts,     wts.data(),     0));
        CUDA_CHECK(cudaEventRecord(h2dDone, streamTransfer));
    }

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

    // --- Build AABBs & memory manager ---
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

    std::cout << "<------ RUN SSSP ------>\n";

    auto dist_ref = sssp_cpu_ref(N, row_ptr, nbrs, wts, SRC);

    auto sssp_runner = [&]() -> std::vector<float> {
        auto out = graph->sssp(rt_pipe, mm, d_params, base, SRC, N, streamCompute);
        CUDA_CHECK(cudaStreamSynchronize(streamCompute));
        return out;
    };

    auto dist_gpu = sssp_runner();
    analyze_diff_sssp(dist_gpu, dist_ref);
    print_sssp_firstn(dist_gpu);

    RunStats stats = time_sssp_gpu(sssp_runner, nbrs.size(), 1, 3);
    std::cout << "SSSP avg time: " << stats.ms << " ms, TEPS: ";
    print_units_teps(stats.teps);
    std::cout << " (edges=" << nbrs.size() << ")\n";

    
    CUDA_CHECK(cudaStreamDestroy(streamCompute));
    CUDA_CHECK(cudaStreamDestroy(streamTransfer));
    CUDA_CHECK(cudaFree((void*)d_params));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(ctx));
    return 0;
}
