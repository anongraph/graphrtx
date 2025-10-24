#include "graph.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cfloat>
#include <random>

#include "algorithms/partition.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/pr.hpp"
#include "algorithms/bc.hpp"
#include "algorithms/sssp.hpp"
#include "algorithms/tc.hpp"
#include "kernels/cuda/algorithms.cuh"

std::vector<uint32_t> graph_rtx::bfs(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams,
    uint32_t source, uint32_t N, CUstream stream, bool hybrid) {
    std::vector<uint32_t> res;
    if(hybrid) {
        run_bfs_hybrid(pipe->get_pipeline(), pipe->get_sbt(), mm, d_params, baseParams, source, N, uasp_first_, uasp_count_);
    } else {
        res = run_bfs(pipe->get_pipeline(), pipe->get_sbt(), mm, d_params, baseParams, source, N, uasp_first_, uasp_count_);
    }
    return res;
}

std::vector<float> graph_rtx::pr(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams, uint32_t N, int iters, float damp, CUstream stream, bool hybrid) {
    std::vector<float> res;
    if(hybrid) {
        run_pagerank_hybrid(pipe->get_pipeline(), pipe->get_sbt(), d_params, baseParams, N, iters, damp, mm, uasp_first_, stream);
    } else {
        res = run_pagerank_optix(pipe->get_pipeline(), pipe->get_sbt(), d_params, baseParams, N, iters, damp, mm, uasp_first_, stream);
    }
    return res;
}

std::vector<float> graph_rtx::sssp(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams,
    uint32_t source, uint32_t N, CUstream stream, bool hybrid) {
    std::vector<float> res;
    if(hybrid) {
        run_sssp_hybrid(pipe->get_pipeline(), pipe->get_sbt(), d_params, baseParams, source, N, mm, uasp_first_, stream);
    } else {
        res = run_sssp(pipe->get_pipeline(), pipe->get_sbt(), d_params, baseParams, source, N, mm, uasp_first_, stream);
    }
    return res;
}

std::vector<float> graph_rtx::bc(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams,
    uint32_t N, CUstream stream, bool hybrid) {
    std::vector<float> res;
    if(hybrid) {
        run_betweenness_hybrid(pipe->get_pipeline(), pipe->get_sbt(), d_params, baseParams, N, mm, uasp_first_, stream);
    } else {
        res = run_betweenness_optix(pipe->get_pipeline(), pipe->get_sbt(), d_params, baseParams, N, mm, uasp_first_, stream);
    }
    return res;
}

uint64_t graph_rtx::tc(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams,
    uint32_t N, CUstream stream, bool hybrid) {
    uint64_t res;
    if(hybrid) {
        run_triangle_counting_hybrid_safe(pipe->get_pipeline(), pipe->get_sbt(), d_params, baseParams, N, row_ptr_, nbrs_ptr_, uasp_first_, uasp_count_, mm, stream);
    } else {
        res = run_triangle_counting_optix(pipe->get_pipeline(), pipe->get_sbt(), d_params, baseParams, N, row_ptr_, nbrs_ptr_, uasp_first_, uasp_count_, mm, stream);
    }
    return res;
}