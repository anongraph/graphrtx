#ifndef GRAPH_HPP
#define GRAPH_HPP
#include <cstdint>
#include <vector>
#include <string>
#include <thread>
#include <memory>

#include "../memory/gpu_manager.hpp"
#include "../rt/rt_pipeline.hpp"
#include "../shared.h"
#include "../common.hpp"

class graph_rtx {
public:
    graph_rtx() = default;

    void set_graph_from_csr(const std::vector<uint32_t>& row_ptr,
                            const std::vector<uint32_t>& nbrs,
                            const std::vector<float>& weights) {
        row_ptr_ = row_ptr;
        nbrs_ptr_ = nbrs;
        wts_ = weights;
    }

    int load_mtx_graph(
                const std::string& filename,
                int num_threads = std::thread::hardware_concurrency());

    void build_aabbs();

    void build_uasps(uint32_t MAX_SEG_LEN);

    void append_dummy_aabbs_tagged(
                uint32_t count,
                float far_multiplier = 1e6f,
                float tiny_extent    = 1.0f);

    std::vector<uint32_t> bfs(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams,
        uint32_t source, uint32_t N, CUstream stream, bool hybrid = false);
    std::vector<float> pr(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams, uint32_t N, int iters, float damp, CUstream stream, bool hybrid = false);
    std::vector<float> sssp(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams,
        uint32_t source, uint32_t N, CUstream stream, bool hybrid = false);
    std::vector<float> bc(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams,
        uint32_t N, CUstream stream, bool hybrid = false);
    uint64_t tc(std::shared_ptr<rt_pipeline>& pipe, GPUMemoryManager& mm, CUdeviceptr d_params, Params& baseParams,
        uint32_t N, CUstream stream, bool hybrid = false);
    
    std::vector<uint32_t>& get_row_ptr() { return row_ptr_; }
    std::vector<uint32_t>& get_nbrs_ptr() { return nbrs_ptr_; }
    std::vector<float>& get_wts() { return wts_; }

    std::vector<uint32_t>& get_uasp_first() { return uasp_first_; }
    std::vector<uint32_t>& get_uasp_count() { return uasp_count_; }
    std::vector<UASP>& get_uasp_host() { return uasps_host_; }

    std::vector<float>& get_aabb() { return aabbs6_; }
    std::vector<uint8_t>& get_mask() { return aabb_mask_; }
    
    double uasp_total_size();

    double aabbs6_total_size();
      
    void generate_random_graph(int n, float p);
      
private:
    int num_parts_;
    int num_dummies_;

    // CSR 
    std::vector<uint32_t>   row_ptr_;
    std::vector<uint32_t>   nbrs_ptr_;
    std::vector<float>      wts_;

    // UASP
    std::vector<uint32_t> uasp_first_;
    std::vector<uint32_t> uasp_count_;
    std::vector<UASP> uasps_host_;

    // Primitives
    std::vector<float> aabbs6_;  
    std::vector<uint8_t> aabb_mask_;
};

#endif