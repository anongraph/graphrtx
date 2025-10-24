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
#include <functional>
#include <limits>

#include <optix_function_table_definition.h>

#include "shared.h"
#include "memory/gpu_manager.hpp"
#include "memory/buffer.hpp"
#include "memory/register.hpp"
#include "common.hpp"
#include "graph/graph.hpp"
#include "algorithms/partition.hpp"
#include "algorithms/pr.hpp"
#include "kernels/cuda/algorithms.cuh"
#include "rt/rt_pipeline.hpp"

static inline std::vector<float> pr_cpu_ref(
    int N,
    const std::vector<uint32_t>& row_ptr,
    const std::vector<uint32_t>& nbrs,
    int iters,
    float damping = 0.85f)
{
    if (N <= 0) return {};

    std::vector<float> rank(N, 1.0f / std::max(1, N));
    std::vector<float> next(N, 0.0f);
    std::vector<uint32_t> outdeg(N);
    for (int u = 0; u < N; ++u) outdeg[u] = row_ptr[u + 1] - row_ptr[u];

    const float base = (1.0f - damping) / std::max(1, N);

    for (int it = 0; it < iters; ++it) {
        double dangling_sum = 0.0;
        for (int u = 0; u < N; ++u) if (outdeg[u] == 0) dangling_sum += rank[u];

        const float dangling_term = damping * float(dangling_sum / std::max(1, N));

        std::fill(next.begin(), next.end(), base + dangling_term);

        for (int u = 0; u < N; ++u) {
            const uint32_t deg = outdeg[u];
            if (deg == 0) continue;
            const float contrib = damping * rank[u] / float(deg);
            const uint32_t beg = row_ptr[u], end = row_ptr[u + 1];
            for (uint32_t e = beg; e < end; ++e) {
                const uint32_t v = nbrs[e];
                next[v] += contrib;
            }
        }

        rank.swap(next);
    }

    double s = 0.0;
    for (float x : rank) s += x;
    if (s > 0) {
        const float inv = float(1.0 / s);
        for (float &x : rank) x *= inv;
    }
    return rank;
}

struct RunStats {
    double ms = 0.0;
    double teps = 0.0; // traversed edges per second
};

static inline RunStats time_pr_gpu(
    const std::function<std::vector<float>()>& runner,
    size_t edges,
    int iters,
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
    
    const double total_edges_processed = double(edges) * double(std::max(1, iters));
    s.teps = (s.ms > 0.0) ? total_edges_processed / (s.ms * 1e-3) : 0.0;
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

static inline void analyze_diff(const std::vector<float>& got, const std::vector<float>& ref) {
    if (got.size() != ref.size()) {
        std::cerr << "PR size mismatch: got " << got.size() << " vs ref " << ref.size() << "\n";
        return;
    }
    
    auto norm = [](std::vector<float> v) {
        double s = 0.0; for (float x : v) s += x;
        if (s > 0) { float inv = float(1.0 / s); for (float &x : v) x *= inv; }
        return v;
    };
    auto a = norm(got), b = norm(ref);

    double l1 = 0.0, maxabs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::abs(double(a[i]) - double(b[i]));
        l1 += d;
        if (d > maxabs) maxabs = d;
    }
    std::cout.setf(std::ios::scientific); std::cout.precision(3);
    std::cout << "PR diff: L1=" << l1 << ", max|Δ|=" << maxabs << "\n";
    std::cout.unsetf(std::ios::scientific);
}

static inline void print_topk(const std::vector<float>& rank, int k = 5) {
    std::vector<int> idx(rank.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(k, idx.size()), idx.end(),
        [&](int i, int j){ return rank[i] > rank[j]; });
    std::cout << "Top-" << k << " ranks: ";
    for (int i = 0; i < std::min<int>(k, (int)idx.size()); ++i) {
        int v = idx[i];
        std::cout.setf(std::ios::fixed); std::cout.precision(6);
        std::cout << "(" << v << ":" << rank[v] << ") ";
        std::cout.unsetf(std::ios::fixed);
    }
    std::cout << "\n";
}

int main()
{
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int   PR_ITERS = 20;     // default iterations
    constexpr float PR_DAMP  = 0.85f;  // damping factor

    uint32_t MAX_SEG_LEN = 1024;
    int num_partitions   = 0;          // 0 = auto
    int num_dummy_nodes  = 0;

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
              << "Edges (CSR entries): " << nbrs.size() << '\n'
              << "Graph size: " << toMB(row_ptr_bytes + nbrs_bytes + wts_bytes) << " MB\n";

    // --- GPU memory check ---
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU mem: free=" << toGB(free_mem)
              << " GB / total="    << toGB(total_mem) << " GB\n";

    const size_t graph_bytes = row_ptr_bytes + nbrs_bytes + wts_bytes;
    const bool use_gpu_memory = (graph_bytes < static_cast<size_t>(free_mem * 0.8));

    std::cout << "[MM] " << (use_gpu_memory
        ? "Enough GPU memory, copying graph to device"
        : "Not enough GPU memory, registering host memory") << "...\n";

    uint32_t *d_row_ptr = nullptr, *d_nbrs = nullptr;
    float    *d_wts     = nullptr;

    cudaStream_t streamCompute, streamTransfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamCompute,  cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamTransfer, cudaStreamNonBlocking));

    cudaEvent_t h2dDone; // sync point for graph uploads
    CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone, cudaEventDisableTiming));

    if (use_gpu_memory) {
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
        if (!row_ptr.empty()) CUDA_CHECK(cudaHostRegister(row_ptr.data(), row_ptr_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!nbrs.empty())    CUDA_CHECK(cudaHostRegister(nbrs.data(),    nbrs_bytes,    cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!wts.empty())     CUDA_CHECK(cudaHostRegister(wts.data(),     wts_bytes,     cudaHostRegisterMapped | cudaHostRegisterPortable));

        if (!row_ptr.empty()) CUDA_CHECK(cudaHostGetDevicePointer(&d_row_ptr, row_ptr.data(), 0));
        if (!nbrs.empty())    CUDA_CHECK(cudaHostGetDevicePointer(&d_nbrs,    nbrs.data(),    0));
        if (!wts.empty())     CUDA_CHECK(cudaHostGetDevicePointer(&d_wts,     wts.data(),     0));
        CUDA_CHECK(cudaEventRecord(h2dDone, streamTransfer)); 
    }

    std::cout << "[MM] Graph memory prepared.\n";

    auto rt_pipe = std::make_shared<rt_pipeline>();
    auto ctx      = rt_pipe->get_context();
    auto module   = rt_pipe->get_module();
    auto pipeline = rt_pipe->get_pipeline();
    auto sbt      = rt_pipe->get_sbt();

    CUdeviceptr d_params = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));
    Params base{};
    base.num_vertices = static_cast<uint32_t>(N);
    base.row_ptr      = d_row_ptr;   // device mem or mapped host-DEV pointer
    base.nbrs         = d_nbrs;
    base.weights      = d_wts;

    std::vector<uint32_t>& uasp_first = graph->get_uasp_first();
    std::vector<uint32_t>& uasp_count = graph->get_uasp_count();
    std::vector<UASP>&     uasps_host = graph->get_uasp_host();
    std::vector<float>&    aabbs6     = graph->get_aabb();

    {
        auto t0 = std::chrono::high_resolution_clock::now();
        graph->build_uasps(MAX_SEG_LEN);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "UASP build: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
    }
    std::cout << "Total UASP size: " << graph->uasp_total_size() << " MB\n";
    std::cout << "Total UASP count: " << uasps_host.size() << "\n";

    aabbs6.reserve(uasps_host.size() * 6);
    {
        auto t2 = std::chrono::high_resolution_clock::now();
        graph->build_aabbs();
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "AABB build: " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";
    }
    std::cout << "AABB count: " << aabbs6.size() << '\n';
    std::cout << "Total AABB size: " << graph->aabbs6_total_size() << " MB\n";

    std::cout << "Create dummies of count: " << num_dummy_nodes << '\n';
    {
        auto t4 = std::chrono::high_resolution_clock::now();
        graph->append_dummy_aabbs_tagged(num_dummy_nodes);
        auto t5 = std::chrono::high_resolution_clock::now();
        std::cout << "Dummy build: " << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms\n";
    }
    std::cout << "Total AABB size with dummies: " << graph->aabbs6_total_size() << " MB\n";

    std::vector<uint8_t>& aabb_mask = graph->get_mask();

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

    const int   iters = PR_ITERS;
    const float damp  = PR_DAMP;

    std::cout << "<------ RUN PageRank (CPU ref) ------>\n";
    auto pr_ref = pr_cpu_ref(N, row_ptr, nbrs, iters, damp);

    // GPU PR runner (standard)
    auto pr_runner = [&]() -> std::vector<float> {
        auto out = graph->pr(rt_pipe, mm, d_params, base, N, iters, damp, streamCompute);
        CUDA_CHECK(cudaStreamSynchronize(streamCompute));
        return out; // assumes graph->pr returns std::vector<float>
    };

    // GPU PR runner (hybrid = true)
    auto pr_runner_hybrid = [&]() -> std::vector<float> {
        auto out = graph->pr(rt_pipe, mm, d_params, base, N, iters, damp, streamCompute, true);
        CUDA_CHECK(cudaStreamSynchronize(streamCompute));
        return out;
    };

    // --- Standard PR: correctness + perf
    std::cout << "<------ RUN PR (standard) ------>\n";
    auto pr_gpu = pr_runner();

    analyze_diff(pr_gpu, pr_ref);

    RunStats stats = time_pr_gpu(pr_runner, nbrs.size(), iters, /*warmup=*/1, /*reps=*/3);
    std::cout << "PR avg time: " << stats.ms << " ms, TEPS: ";
    print_units_teps(stats.teps);
    std::cout << " (edges=" << nbrs.size() << ", iters=" << iters << ")\n";
    print_topk(pr_gpu, 5);

    
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
