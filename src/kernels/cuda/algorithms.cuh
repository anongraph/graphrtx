#ifndef CUDA_ALGORITHMS
#define CUDA_ALGORITHMS

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>


extern "C" void launch_tc_join_warp_coop(const Job* jobs,
    uint32_t   num_jobs,
    const uint32_t* row_ptr,
    const uint32_t* nbrs,
    unsigned long long* tri_count);

extern "C" void launch_bfs_expand(const Job* jobs, uint32_t num_jobs,
const uint32_t* row_ptr, const uint32_t* nbrs,
uint32_t* next_frontier, uint32_t* next_count,
uint32_t  next_capacity,
uint32_t* visited_bitmap, uint32_t  visited_words,
uint32_t* distances,
CUstream stream);

extern "C" void launch_memset_u32(uint32_t* arr, uint32_t value, uint32_t n, CUstream stream);
extern "C" void launch_memset_f32(float* arr, float value, uint32_t n, CUstream stream);
extern "C" void launch_add_delta_to_bc(const uint32_t* nodes,
                            uint32_t count,
                            int source,
                            const float* delta,
                            float* bc,
                            uint32_t N,
                            CUstream stream);
extern "C" {
void launch_d2d_copy_u32(const uint32_t* src, uint32_t* dst, uint32_t count, CUstream stream);
void launch_build_jobs_from_nodes(const uint32_t* nodes, uint32_t count,
Job* jobs, QueryType qtype, CUstream stream);
}

extern "C"
void launch_pr_reduce_dangling(const float* pr_curr, const float* inv_outdeg,
int N, float* d_out, CUstream stream);

extern "C"
void launch_pr_fill_base(float* pr_next, int N, float base_term, CUstream stream);

extern "C"
void launch_pr_scatter_items(const PRWorkItem* items, uint32_t num_items,
const uint32_t* nbrs, float* pr_next, CUstream stream);

extern "C" void launch_pr_build_items(
  const float* d_pr_curr,
  const float* d_inv_outdeg,
  const uint32_t* d_row_ptr,
  int N,
  PRWorkItem* d_items,
  float damping,
  CUstream s);

extern "C" void launch_pr_diff_norm(
  const float* d_a,
  const float* d_b,
  int n,
  float* d_out_sum,
  CUstream s);

extern "C" void launch_sssp_relax_frontier(
  const uint32_t* d_row_ptr,
  const uint32_t* d_nbrs,
  const uint32_t* d_frontier,
  uint32_t frontier_size,
  float* d_dist,
  uint32_t* d_next_frontier,
  uint32_t* d_next_count,
  CUstream s);

extern "C"
void launch_bc_forward_expand_nodes(const uint32_t* nodes,
                                    uint32_t        count,
                                    const uint32_t* row_ptr,
                                    const uint32_t* nbrs,
                                    uint32_t*       next_frontier,
                                    uint32_t*       next_count,
                                    uint32_t*       dist,
                                    uint32_t*       sigma,
                                    uint32_t        cur_depth,
                                    CUstream        stream);

extern "C" void launch_bc_backward_accumulate_nodes(
    const uint32_t* level_nodes, uint32_t count,
    const uint32_t* row_ptr,     const uint32_t* nbrs,
    const uint32_t* dist,        const uint32_t* sigma,
    const float*    delta,       float* delta_out,
    CUstream        stream);

    
#endif