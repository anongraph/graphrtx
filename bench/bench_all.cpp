#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

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
#include "algorithms/wcc.hpp"              
#include "algorithms/cdlp.hpp"             
#include "kernels/cuda/algorithms.cuh"
#include "rt/rt_pipeline.hpp"


struct Cli {
    std::string input;
    int device = 0;
    uint32_t max_seg = 1024;
    int partitions = 0;
    int dummies = 0;
    int src = 0;
    int pr_iters = 20;
    float pr_damp = 0.85f;

    int cdlp_iters = 20; 

    bool run_bfs = true, run_pr = true, run_sssp = true, run_bc = true, run_tc = true, run_wcc = true, run_cdlp = true; 
    bool run_hybrid = true;
    bool quiet = false;
};

static void print_help(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " <graph.mtx> [OPTIONS]\n\n"
        "Options:\n"
        "  --device N           CUDA device (default 0)\n"
        "  --max-seg N          Max UASP segment length (default 1024)\n"
        "  --parts N            Number of partitions (0=auto)\n"
        "  --dummies N          Dummy AABBs to append (default 0)\n"
        "  --src N              Source vertex for BFS/SSSP (default 0)\n"
        "  --pr-iters N         PageRank iterations (default 20)\n"
        "  --pr-damp F          PageRank damping (default 0.85)\n"
        "  --cdlp-iters N       CDLP iterations (default 20)\n"
        "  --algo LIST          e.g., bfs,pr,sssp,bc,tc,wcc,cdlp,all (default all)\n"
        "  --no-hybrid          Disable hybrid variants\n"
        "  -q, --quiet          Minimal logging\n"
        "  -h, --help           Show this help\n";
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
        if (a == "--device")       c.device     = std::stoi(val(a.c_str()));
        else if (a == "--max-seg") c.max_seg    = std::stoul(val(a.c_str()));
        else if (a == "--parts")   c.partitions = std::stoi(val(a.c_str()));
        else if (a == "--dummies") c.dummies    = std::stoi(val(a.c_str()));
        else if (a == "--src")     c.src        = std::stoi(val(a.c_str()));
        else if (a == "--pr-iters")c.pr_iters   = std::stoi(val(a.c_str()));
        else if (a == "--pr-damp") c.pr_damp    = std::stof(val(a.c_str())); 
        else if (a == "--cdlp-iters") c.cdlp_iters = std::stoi(val(a.c_str()));
        else if (a == "--no-hybrid") c.run_hybrid = false;
        else if (a == "--algo") {
            std::string list = val(a.c_str());
            c.run_bfs = c.run_pr = c.run_sssp = c.run_bc = c.run_tc = c.run_wcc = c.run_cdlp = false;
            for (size_t p = 0; p < list.size();) {
                size_t q = list.find(',', p);
                auto t = list.substr(p, q == std::string::npos ? q : q - p);
                if (t == "bfs") c.run_bfs = true;
                else if (t == "pr") c.run_pr = true;
                else if (t == "sssp") c.run_sssp = true;
                else if (t == "bc") c.run_bc = true;
                else if (t == "tc") c.run_tc = true;
                else if (t == "wcc") c.run_wcc = true;
                else if (t == "cdlp") c.run_cdlp = true;
                else if (t == "all") c.run_bfs = c.run_pr = c.run_sssp = c.run_bc = c.run_tc = c.run_wcc = c.run_cdlp = true;
                else { std::cerr << "Unknown algo: " << t << "\n"; std::exit(1); }
                if (q == std::string::npos) break;
                p = q + 1;
            }
        }
        else if (a == "-q" || a == "--quiet") c.quiet = true;
        else if (a == "-h" || a == "--help") { print_help(argv[0]); return std::nullopt; }
        else { std::cerr << "Unknown option: " << a << "\n"; return std::nullopt; }
    }
    if (!std::filesystem::exists(c.input)) {
        std::cerr << "File not found: " << c.input << "\n";
        return std::nullopt;
    }
    return c;
}

int main(int argc, char** argv) {
    auto cliOpt = parse_cli(argc, argv);
    if (!cliOpt) return 1;
    const Cli cli = *cliOpt;

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(cli.device));

    std::cout << "Config: DEVICE=" << cli.device
              << " MAX_SEG=" << cli.max_seg
              << " PARTS=" << cli.partitions
              << " DUMMIES=" << cli.dummies
              << " SRC=" << cli.src
              << " PR_ITERS=" << cli.pr_iters
              << " PR_DAMP=" << cli.pr_damp
              << " CDLP_ITERS=" << cli.cdlp_iters
              << " HYBRID=" << (cli.run_hybrid ? "on" : "off") << "\n";

    auto graph = std::make_shared<graph_rtx>();
    {
        ScopedTimer t("Load graph");
        if (graph->load_mtx_graph(cli.input) <= 0) {
            std::cerr << "Graph load failed.\n";
            return 1;
        }
    }

    auto& row_ptr = graph->get_row_ptr();
    auto& nbrs    = graph->get_nbrs_ptr();
    auto& wts     = graph->get_wts();

    const size_t row_bytes = row_ptr.size() * sizeof(uint32_t);
    const size_t nbr_bytes = nbrs.size() * sizeof(uint32_t);
    const size_t wt_bytes  = wts.size()  * sizeof(float);

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    const size_t graph_bytes = row_bytes + nbr_bytes + wt_bytes;
    const bool use_gpu_memory = graph_bytes < (size_t)(free_mem * 0.8);
    std::cout << "GPU mem: free=" << toGB(free_mem) << " GB, graph=" << toMB(graph_bytes)
              << " MB → " << (use_gpu_memory ? "device copy" : "mapped host") << "\n";

    ScopedStream streamCompute, streamTransfer;
    ScopedEvent h2dDone;

    uint32_t *d_row_ptr = nullptr, *d_nbrs = nullptr;
    float *d_wts = nullptr;

    if (use_gpu_memory) {
        PinnedRegister pin_row(row_ptr.data(), row_bytes);
        PinnedRegister pin_nbr(nbrs.data(),    nbr_bytes);
        PinnedRegister pin_wts(wts.data(),     wt_bytes);

        CUDA_CHECK(cudaMalloc(&d_row_ptr, row_bytes));
        CUDA_CHECK(cudaMalloc(&d_nbrs,    nbr_bytes));
        CUDA_CHECK(cudaMalloc(&d_wts,     wt_bytes));

        CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, row_ptr.data(), row_bytes, cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaMemcpyAsync(d_nbrs,    nbrs.data(),    nbr_bytes, cudaMemcpyHostToDevice, streamTransfer));
        CUDA_CHECK(cudaMemcpyAsync(d_wts,     wts.data(),     wt_bytes,  cudaMemcpyHostToDevice, streamTransfer));
    } else {
        if (!row_ptr.empty()) CUDA_CHECK(cudaHostRegister(row_ptr.data(), row_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!nbrs.empty())    CUDA_CHECK(cudaHostRegister(nbrs.data(),    nbr_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!wts.empty())     CUDA_CHECK(cudaHostRegister(wts.data(),     wt_bytes,  cudaHostRegisterMapped | cudaHostRegisterPortable));
        if (!row_ptr.empty()) CUDA_CHECK(cudaHostGetDevicePointer(&d_row_ptr, row_ptr.data(), 0));
        if (!nbrs.empty())    CUDA_CHECK(cudaHostGetDevicePointer(&d_nbrs,    nbrs.data(),    0));
        if (!wts.empty())     CUDA_CHECK(cudaHostGetDevicePointer(&d_wts,     wts.data(),     0));
    }
    CUDA_CHECK(cudaEventRecord(h2dDone, streamTransfer));

    auto rt_pipe = std::make_shared<rt_pipeline>();
    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));
    Params base{};
    base.row_ptr = d_row_ptr;
    base.nbrs    = d_nbrs;
    base.weights = d_wts;
    base.num_vertices = (uint32_t)(row_ptr.size() - 1);

    {
        ScopedTimer t("Build UASPs");
        graph->build_uasps(cli.max_seg);
    }
    {
        ScopedTimer t("Build AABBs");
        graph->build_aabbs();
    }
    if (cli.dummies > 0) {
        ScopedTimer t("Build dummy AABBs");
        graph->append_dummy_aabbs_tagged(cli.dummies);
    }

    auto& uasp_first = graph->get_uasp_first();
    auto& uasp_count = graph->get_uasp_count();
    auto& uasps_host = graph->get_uasp_host();
    auto& aabbs6     = graph->get_aabb();
    auto& aabb_mask  = graph->get_mask();

    DeviceBuffer<uint8_t>  d_mask(aabb_mask.size());
    DeviceBuffer<UASP>     d_uasps(uasps_host.size());
    DeviceBuffer<float>    d_aabbs(aabbs6.size());
    DeviceBuffer<uint32_t> d_first(uasp_first.size());
    DeviceBuffer<uint32_t> d_count(uasp_count.size());

    d_mask.uploadAsync(aabb_mask.data(), aabb_mask.size(), streamTransfer);
    d_uasps.uploadAsync(uasps_host.data(), uasps_host.size(), streamTransfer);
    d_aabbs.uploadAsync(aabbs6.data(), aabbs6.size(), streamTransfer);
    d_first.uploadAsync(uasp_first.data(), uasp_first.size(), streamTransfer);
    d_count.uploadAsync(uasp_count.data(), uasp_count.size(), streamTransfer);

    base.aabb_mask  = d_mask.ptr;
    base.uasps      = d_uasps.ptr;
    base.aabbs      = d_aabbs.ptr;
    base.uasp_first = d_first.ptr;
    base.uasp_count = d_count.ptr;
    base.num_uasps  = (uint32_t)uasps_host.size();
    base.num_aabbs  = (uint32_t)(aabbs6.size() / 6);

    CUDA_CHECK(cudaStreamWaitEvent(streamCompute, h2dDone, 0));

    GPUMemoryManager mm(rt_pipe->get_context(), uasps_host, aabbs6, (uint32_t)uasps_host.size(),
                        0.85f, cli.partitions, &aabb_mask);

    auto run = [&](const char* name, auto fn) {
        std::cout << "<------ RUN " << name << " ------>\n";
        ScopedTimer t(name);
        fn();
        CUDA_CHECK(cudaStreamSynchronize(streamCompute));
    };

    if (cli.run_bfs) {
        run("BFS", [&]{ graph->bfs(rt_pipe, mm, d_params, base, cli.src, base.num_vertices, streamCompute); });
        if (cli.run_hybrid) run("BFS Hybrid", [&]{ graph->bfs(rt_pipe, mm, d_params, base, cli.src, base.num_vertices, streamCompute, true); });
    }

    if (cli.run_pr)  {
        run("PR",  [&]{ graph->pr(rt_pipe, mm, d_params, base, base.num_vertices, cli.pr_iters, cli.pr_damp, streamCompute); });
        if (cli.run_hybrid) run("PR Hybrid", [&]{ graph->pr(rt_pipe, mm, d_params, base, base.num_vertices, cli.pr_iters, cli.pr_damp, streamCompute, true); });
    }

    if (cli.run_sssp){
        run("SSSP", [&]{ graph->sssp(rt_pipe, mm, d_params, base, cli.src, base.num_vertices, streamCompute); });
        if (cli.run_hybrid) run("SSSP Hybrid", [&]{ graph->sssp(rt_pipe, mm, d_params, base, cli.src, base.num_vertices, streamCompute, true); });
    }

    if (cli.run_bc)  {
        run("BC",  [&]{ graph->bc(rt_pipe, mm, d_params, base, base.num_vertices, streamCompute); });
        if (cli.run_hybrid) run("BC Hybrid", [&]{ graph->bc(rt_pipe, mm, d_params, base, base.num_vertices, streamCompute, true); });
    }

    if (cli.run_tc)  {
        run("TC",  [&]{ graph->tc(rt_pipe, mm, d_params, base, base.num_vertices, streamCompute); });
        if (cli.run_hybrid) run("TC Hybrid", [&]{ graph->tc(rt_pipe, mm, d_params, base, base.num_vertices, streamCompute, true); });
    }

    if (cli.run_wcc) {
        run("WCC", [&]{
            (void)run_wcc_optix(rt_pipe->get_pipeline(), rt_pipe->get_sbt(), d_params,
                                base, (int)base.num_vertices, mm, uasp_first, streamCompute);
        });

        if (cli.run_hybrid) {
            run("WCC Hybrid", [&]{
                (void)run_wcc_hybrid(rt_pipe->get_pipeline(), rt_pipe->get_sbt(), d_params,
                                     base, (int)base.num_vertices, mm, uasp_first, streamCompute);
            });
        }
    }

    if (cli.run_cdlp) {
        run("CDLP", [&]{
            (void)run_cdlp_optix(rt_pipe->get_pipeline(), rt_pipe->get_sbt(), d_params,
                                 base, (int)base.num_vertices, mm, uasp_first, streamCompute,
                                 cli.cdlp_iters);
        });

        if (cli.run_hybrid) {
            run("CDLP Hybrid", [&]{
                (void)run_cdlp_hybrid(rt_pipe->get_pipeline(), rt_pipe->get_sbt(), d_params,
                                      base, (int)base.num_vertices, mm, uasp_first, streamCompute,
                                      cli.cdlp_iters);
            });
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(streamTransfer));
    CUDA_CHECK(cudaFree((void*)d_params));
    OPTIX_CHECK(optixDeviceContextDestroy(rt_pipe->get_context()));
    return 0;
}
