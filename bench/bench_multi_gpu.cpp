// bench_multi_gpu.cpp
//
// Multi-GPU benchmark driver for your OptiX+CUDA graph pipeline.
// SHARDED BVH (partitions owned per GPU) + REPLICATED algorithm execution (BFS only here).
//
// FIXES APPLIED:
//  1) Removed manual driver primary-ctx retain/setcurrent (bind_primary_ctx()).
//     Multi-thread + driver context juggling is a common source of “wrong device” OptiX handles.
//     We now bind per-thread with cudaSetDevice + cudaFree(0), then (optionally) grab CUcontext.
//  2) Ensure rt_pipeline (OptiX context/pipeline/SBT) is constructed AFTER device bind in each thread.
//     Each GPU thread must have its own OptiX objects on that device.
//  3) Removed explicit optixDeviceContextDestroy() to avoid double-destroy if rt_pipeline owns it.
//     If your rt_pipeline does NOT destroy its OptiX context, see the comment near cleanup.
//  4) Added an explicit optixInit() safety in main as before.
//  5) Kept peer access logic.
//
// If your rt_pipeline constructor builds the OptiX context/pipeline/SBT immediately, you are done.
// If your rt_pipeline requires an explicit init call, add it right after construction in worker_run().
//
// Example:
//   ./bench_multi_gpu graph.mtx --gpus 0,1 --sharded --parts 8000 --src 0 --runs 3

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "shared.h"
#include "common.hpp"
#include "graph/graph.hpp"
#include "rt/rt_pipeline.hpp"

#include "memory/gpu_manager.hpp"
#include "memory/buffer.hpp"


struct HostPin {
    void*  ptr = nullptr;
    size_t bytes = 0;
    bool   pinned = false;

    HostPin() = default;
    HostPin(void* p, size_t b) : ptr(p), bytes(b) {
        if (ptr && bytes) {
            CUDA_CHECK(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable));
            pinned = true;
        }
    }
    ~HostPin() {
        if (pinned) CUDA_CHECK(cudaHostUnregister(ptr));
    }
    HostPin(const HostPin&) = delete;
    HostPin& operator=(const HostPin&) = delete;
    HostPin(HostPin&& o) noexcept {
        ptr=o.ptr; bytes=o.bytes; pinned=o.pinned;
        o.ptr=nullptr; o.bytes=0; o.pinned=false;
    }
    HostPin& operator=(HostPin&& o) noexcept {
        if (this == &o) return *this;
        if (pinned) CUDA_CHECK(cudaHostUnregister(ptr));
        ptr=o.ptr; bytes=o.bytes; pinned=o.pinned;
        o.ptr=nullptr; o.bytes=0; o.pinned=false;
        return *this;
    }
};


struct Cli {
    std::string input;

    std::vector<int> gpus;
    bool sharded = false;
    ShardMode shard_mode = ShardMode::Contiguous;

    int runs = 1;
    bool peer = false;

    uint32_t max_seg = 1024;
    int parts = 0;    
    int dummies = 0;

    int src = 0;

    bool quiet = false;
};

static void print_help(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <graph.mtx> [OPTIONS]\n\n"
        << "Options:\n"
        << "  --gpus LIST          CUDA devices list, e.g. 0,1 or 0,1,2,3\n"
        << "  --sharded            Enable BVH partition sharding\n"
        << "  --shard MODE         contiguous|rr (default contiguous)\n"
        << "  --parts N            Global partition count (REQUIRED for --sharded)\n"
        << "  --runs N             Repetitions per GPU (default 1)\n"
        << "  --peer               Enable cudaDeviceEnablePeerAccess (optional)\n"
        << "  --max-seg N          Max UASP segment length (default 1024)\n"
        << "  --dummies N          Dummy AABBs to append (default 0)\n"
        << "  --src N              BFS source vertex (default 0)\n"
        << "  -q, --quiet          Minimal logging\n"
        << "  -h, --help           Show this help\n";
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        std::string t = s.substr(p, q == std::string::npos ? q : (q - p));
        if (!t.empty()) out.push_back(std::stoi(t));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

static std::optional<Cli> parse_cli(int argc, char** argv) {
    if (argc < 2) { print_help(argv[0]); return std::nullopt; }

    Cli c;
    c.input = argv[1];
    if (c.input == "-h" || c.input == "--help") { print_help(argv[0]); return std::nullopt; }

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; std::exit(1); }
            return argv[++i];
        };

        if (a == "--gpus") c.gpus = parse_int_list(val(a.c_str()));
        else if (a == "--sharded") c.sharded = true;
        else if (a == "--shard") {
            std::string m = val(a.c_str());
            if (m == "contiguous") c.shard_mode = ShardMode::Contiguous;
            else if (m == "rr") c.shard_mode = ShardMode::RoundRobin;
            else { std::cerr << "Unknown --shard: " << m << "\n"; std::exit(1); }
        }
        else if (a == "--parts") c.parts = std::stoi(val(a.c_str()));
        else if (a == "--runs") c.runs = std::stoi(val(a.c_str()));
        else if (a == "--peer") c.peer = true;
        else if (a == "--max-seg") c.max_seg = std::stoul(val(a.c_str()));
        else if (a == "--dummies") c.dummies = std::stoi(val(a.c_str()));
        else if (a == "--src") c.src = std::stoi(val(a.c_str()));
        else if (a == "-q" || a == "--quiet") c.quiet = true;
        else if (a == "-h" || a == "--help") { print_help(argv[0]); return std::nullopt; }
        else { std::cerr << "Unknown option: " << a << "\n"; return std::nullopt; }
    }

    if (!std::filesystem::exists(c.input)) {
        std::cerr << "File not found: " << c.input << "\n";
        return std::nullopt;
    }
    if (c.gpus.empty()) c.gpus = {0};
    if (c.runs < 1) c.runs = 1;

    if (c.sharded && c.parts <= 0) {
        std::cerr << "[bench] ERROR: --sharded requires --parts N so all GPUs share the same global partitioning.\n";
        return std::nullopt;
    }

    return c;
}

static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static CUcontext bind_cuda_device(int gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));
    CUDA_CHECK(cudaFree(0)); 
    CUcontext ctx = nullptr;
    CUresult rc = cuCtxGetCurrent(&ctx);
    if (rc != CUDA_SUCCESS || ctx == nullptr) {
        std::cerr << "cuCtxGetCurrent failed / null context on gpu " << gpu << "\n";
        std::exit(1);
    }
    return ctx;
}

static void try_enable_peer_access(const std::vector<int>& gpus) {
    for (int src : gpus) {
        CUDA_CHECK(cudaSetDevice(src));
        CUDA_CHECK(cudaFree(0));
        for (int dst : gpus) if (src != dst) {
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, src, dst));
            if (can) {
                cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
                if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                else CUDA_CHECK(e);
            }
        }
    }
}

struct GpuResult {
    int gpu = -1;
    int rank = 0;
    int num_gpus = 1;

    double load_ms = 0.0;
    double uasps_ms = 0.0;
    double aabbs_ms = 0.0;

    double bvh_build_ms = 0.0;
    double bfs_ms = 0.0;

    double bvh_mb = 0.0;

    size_t freeB = 0, totalB = 0;
};


static void worker_run(const Cli& cli, int gpu, int rank, int num_gpus,
                       GpuResult& out, std::mutex& print_mu)
{
    out.gpu = gpu;
    out.rank = rank;
    out.num_gpus = num_gpus;

    (void)bind_cuda_device(gpu);

    if (!cli.quiet) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));
        std::lock_guard<std::mutex> lk(print_mu);
        std::cout << "[GPU " << gpu << "] " << prop.name
                  << " total_mem=" << toGB((size_t)prop.totalGlobalMem) << " GB\n";
    }

    auto graph = std::make_shared<graph_rtx>();

    {
        const double t0 = now_ms();
        if (graph->load_mtx_graph(cli.input) <= 0) {
            std::lock_guard<std::mutex> lk(print_mu);
            std::cerr << "[GPU " << gpu << "] Graph load failed.\n";
            return;
        }
        out.load_ms = now_ms() - t0;
    }

    {
        const double t0 = now_ms();
        graph->build_uasps(cli.max_seg);
        out.uasps_ms = now_ms() - t0;
    }
    {
        const double t0 = now_ms();
        graph->build_aabbs();
        out.aabbs_ms = now_ms() - t0;
    }
    if (cli.dummies > 0) {
        graph->append_dummy_aabbs_tagged(cli.dummies);
    }

    auto& row_ptr    = graph->get_row_ptr();
    auto& nbrs       = graph->get_nbrs_ptr();
    auto& wts        = graph->get_wts();

    auto& uasp_first = graph->get_uasp_first();
    auto& uasps_host = graph->get_uasp_host();
    auto& aabbs6     = graph->get_aabb();
    auto& aabb_mask  = graph->get_mask();

    const size_t row_bytes = row_ptr.size() * sizeof(uint32_t);
    const size_t nbr_bytes = nbrs.size() * sizeof(uint32_t);
    const size_t wt_bytes  = wts.size()  * sizeof(float);

    CUDA_CHECK(cudaMemGetInfo(&out.freeB, &out.totalB));

    ScopedStream streamCompute, streamTransfer;

    uint32_t *d_row = nullptr, *d_nbr = nullptr;
    float *d_wt = nullptr;

    HostPin pin_row((void*)row_ptr.data(), row_bytes);
    HostPin pin_nbr((void*)nbrs.data(),    nbr_bytes);
    HostPin pin_wts((void*)wts.data(),     wt_bytes);

    CUDA_CHECK(cudaMalloc(&d_row, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_nbr, nbr_bytes));
    CUDA_CHECK(cudaMalloc(&d_wt,  wt_bytes));

    CUDA_CHECK(cudaMemcpyAsync(d_row, row_ptr.data(), row_bytes, cudaMemcpyHostToDevice, streamTransfer));
    CUDA_CHECK(cudaMemcpyAsync(d_nbr, nbrs.data(),    nbr_bytes, cudaMemcpyHostToDevice, streamTransfer));
    CUDA_CHECK(cudaMemcpyAsync(d_wt,  wts.data(),     wt_bytes,  cudaMemcpyHostToDevice, streamTransfer));

    auto rt_pipe = std::make_shared<rt_pipeline>();

    CUdeviceptr d_params = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));

    Params base{};
    base.row_ptr = d_row;
    base.nbrs    = d_nbr;
    base.weights = d_wt;
    base.num_vertices = (uint32_t)(row_ptr.size() - 1);

    DeviceBuffer<uint8_t>  d_mask(aabb_mask.size());
    DeviceBuffer<UASP>     d_uasps(uasps_host.size());
    DeviceBuffer<float>    d_aabbs(aabbs6.size());
    DeviceBuffer<uint32_t> d_first(uasp_first.size());
    DeviceBuffer<uint32_t> d_count(graph->get_uasp_count().size());

    d_mask.uploadAsync(aabb_mask.data(), aabb_mask.size(), streamTransfer);
    d_uasps.uploadAsync(uasps_host.data(), uasps_host.size(), streamTransfer);
    d_aabbs.uploadAsync(aabbs6.data(), aabbs6.size(), streamTransfer);
    d_first.uploadAsync(uasp_first.data(), uasp_first.size(), streamTransfer);
    d_count.uploadAsync(graph->get_uasp_count().data(), graph->get_uasp_count().size(), streamTransfer);

    base.aabb_mask  = d_mask.ptr;
    base.uasps      = d_uasps.ptr;
    base.aabbs      = d_aabbs.ptr;
    base.uasp_first = d_first.ptr;
    base.uasp_count = d_count.ptr;
    base.num_uasps  = (uint32_t)uasps_host.size();
    base.num_aabbs  = (uint32_t)(aabbs6.size() / 6);

    CUDA_CHECK(cudaStreamSynchronize(streamTransfer));

    const double t_bvh0 = now_ms();
    GPUMemoryManager mm(
        rt_pipe->get_context(),
        uasps_host,
        aabbs6,
        (uint32_t)uasps_host.size(),
        0.85f,
        (uint32_t)(cli.sharded ? cli.parts : 0), /
        &aabb_mask,
        cli.sharded,
        (uint32_t)num_gpus,
        (uint32_t)rank,
        cli.shard_mode
    );
    CUDA_CHECK(cudaStreamSynchronize(streamCompute));
    out.bvh_build_ms = now_ms() - t_bvh0;
    out.bvh_mb = mm.getTotalBVHBytes() / (1024.0 * 1024.0);

    double best_bfs = 1e100;
    for (int r = 0; r < cli.runs; ++r) {
        const double t0 = now_ms();
        graph->bfs(rt_pipe, mm, d_params, base, cli.src, base.num_vertices, streamCompute);
        CUDA_CHECK(cudaStreamSynchronize(streamCompute));
        best_bfs = std::min(best_bfs, now_ms() - t0);
    }
    out.bfs_ms = best_bfs;

    {
        std::lock_guard<std::mutex> lk(print_mu);
        std::cout << "[GPU " << gpu << "] rank=" << rank << "/" << num_gpus
                  << " load=" << out.load_ms << " ms"
                  << " uasps=" << out.uasps_ms << " ms"
                  << " aabbs=" << out.aabbs_ms << " ms"
                  << " bvh_build=" << out.bvh_build_ms << " ms"
                  << " bvh_size≈" << out.bvh_mb << " MB"
                  << " bfs_best=" << out.bfs_ms << " ms"
                  << "\n";
    }

    CUDA_CHECK(cudaFree((void*)d_params));


    CUDA_CHECK(cudaFree(d_row));
    CUDA_CHECK(cudaFree(d_nbr));
    CUDA_CHECK(cudaFree(d_wt));
}


int main(int argc, char** argv) {
    auto cliOpt = parse_cli(argc, argv);
    if (!cliOpt) return 1;
    const Cli cli = *cliOpt;

    int nDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    for (int g : cli.gpus) {
        if (g < 0 || g >= nDevices) {
            std::cerr << "Invalid GPU id " << g << " (have " << nDevices << " devices)\n";
            return 1;
        }
    }

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    for (int g : cli.gpus) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(0));
    }

    OPTIX_CHECK(optixInit());

    if (cli.peer && cli.gpus.size() > 1) {
        try_enable_peer_access(cli.gpus);
    }

    if (!cli.quiet) {
        std::cout << "Config: GPUS=";
        for (size_t i=0;i<cli.gpus.size();++i) std::cout << cli.gpus[i] << (i+1<cli.gpus.size()? ",":"");
        std::cout << " SHARDED=" << (cli.sharded ? "on" : "off")
                  << " SHARD=" << (cli.shard_mode == ShardMode::Contiguous ? "contiguous" : "rr")
                  << " PARTS=" << cli.parts
                  << " RUNS=" << cli.runs
                  << " MAX_SEG=" << cli.max_seg
                  << " SRC=" << cli.src
                  << "\n";
    }

    const int num_gpus = (int)cli.gpus.size();
    std::mutex print_mu;
    std::vector<GpuResult> results((size_t)num_gpus);
    std::vector<std::thread> threads;
    threads.reserve((size_t)num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        threads.emplace_back([&, i]{
            worker_run(cli, cli.gpus[i], i, num_gpus, results[(size_t)i], print_mu);
        });
    }
    for (auto& t : threads) t.join();

    std::cout << "\n=== Summary ===\n";
    for (auto& r : results) {
        std::cout << "GPU " << r.gpu
                  << " bvh_build=" << r.bvh_build_ms << " ms"
                  << " bfs_best=" << r.bfs_ms << " ms"
                  << " bvh_size≈" << r.bvh_mb << " MB\n";
    }

    return 0;
}
