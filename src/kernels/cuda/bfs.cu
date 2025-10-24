#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "shared.h"

extern "C" __global__
void bfs_expand_from_segments_kernel(const uint32_t* __restrict__ seg_ids,
                                     uint32_t                     num_segs,
                                     const UASP* __restrict__ uasps,
                                     const uint32_t* __restrict__ row_ptr,
                                     const uint32_t* __restrict__ nbrs,
                                     uint32_t* __restrict__ next_frontier,
                                     uint32_t* __restrict__ next_count,
                                     uint32_t* __restrict__ visited_bitmap,
                                     uint32_t* __restrict__ distances,
                                     uint32_t                     cur_depth)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_segs) return;

    const uint32_t sid = seg_ids[tid];
    const UASP seg = uasps[sid];

    const uint32_t start = seg.start;
    const uint32_t end   = start + seg.len;

    for (uint32_t k = start; k < end; ++k) {
        const uint32_t v = __ldg(&nbrs[k]);
        const uint32_t w = v >> 5;
        const uint32_t m = 1u << (v & 31);
        const uint32_t old = atomicOr(&visited_bitmap[w], m);
        if ((old & m) == 0) {
            distances[v] = cur_depth + 1;
            const uint32_t idx = atomicAdd(next_count, 1u);
            next_frontier[idx] = v;
        }
    }
}

extern "C" void launch_bfs_expand_segments(const uint32_t* seg_ids,
                                           uint32_t        num_segs,
                                           const UASP* uasps,
                                           const uint32_t* row_ptr,
                                           const uint32_t* nbrs,
                                           uint32_t* next_frontier,
                                           uint32_t* next_count,
                                           uint32_t* visited_bitmap,
                                           uint32_t* distances,
                                           uint32_t        cur_depth,
                                           CUstream        stream)
{
    if (num_segs == 0) return;
    const int block = 256;
    const int grid  = (num_segs + block - 1) / block;
    bfs_expand_from_segments_kernel<<<grid, block, 0, stream>>>(
        seg_ids, num_segs, uasps, row_ptr, nbrs,
        next_frontier, next_count, visited_bitmap, distances, cur_depth
    );
}

extern "C" __global__
void bfs_expand_kernel(const Job* __restrict__ jobs,
                       uint32_t                 num_jobs,
                       const uint32_t* __restrict__ row_ptr,
                       const uint32_t* __restrict__ nbrs,
                       uint32_t* __restrict__ next_frontier,
                       uint32_t* __restrict__ next_count,
                       uint32_t                 next_capacity,
                       uint32_t* __restrict__ visited_bitmap,
                       uint32_t                 visited_words,
                       uint32_t* __restrict__ distances)
{
    const unsigned global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned warp_id    = global_tid >> 5;          
    if (warp_id >= num_jobs) return;

    const unsigned lane_id = threadIdx.x & 31;            
    const unsigned mask    = __activemask();

    Job j;
    if (lane_id == 0) {
        j = jobs[warp_id];
    }
    j.src   = __shfl_sync(mask, j.src,   0);
    j.depth = __shfl_sync(mask, j.depth, 0);

    const uint32_t u          = j.src;
    const uint32_t row_start  = __ldg(&row_ptr[u]);
    const uint32_t row_end    = __ldg(&row_ptr[u + 1]);
    const uint32_t u_degree   = row_end - row_start;
    const uint32_t* u_neighbors = nbrs + row_start;
    const uint32_t new_dist   = j.depth + 1;

    for (uint32_t i = 0; i < u_degree; i += 32) {
        uint32_t neighbor_v = 0;
        bool is_new = false;

        const uint32_t idx_in_row = i + lane_id;
        if (idx_in_row < u_degree) {
            neighbor_v = __ldg(&u_neighbors[idx_in_row]);

            const uint32_t word_idx = neighbor_v >> 5;     
            const uint32_t bit_mask = 1u << (neighbor_v & 31);

            if (word_idx < visited_words) {
                const uint32_t old_word = atomicOr(&visited_bitmap[word_idx], bit_mask);
                if ((old_word & bit_mask) == 0) {
                    is_new = true;
                }
            }
        }

        const unsigned new_nodes_ballot = __ballot_sync(mask, is_new);
        if (new_nodes_ballot == 0) continue; 
        const int warp_total_new = __popc(new_nodes_ballot);

        uint32_t write_idx_base = 0;
        if (lane_id == 0) {
            write_idx_base = atomicAdd(next_count, (uint32_t)warp_total_new);
        }
        write_idx_base = __shfl_sync(mask, write_idx_base, 0);
        uint32_t write_n = 0;
        if (write_idx_base < next_capacity) {
            const uint32_t remaining = next_capacity - write_idx_base;
            write_n = (uint32_t)min(remaining, (uint32_t)warp_total_new);
        } else {
            write_n = 0;
        }

        write_n = __shfl_sync(mask, write_n, 0);

        const int lane_rank = __popc(new_nodes_ballot & ((1u << lane_id) - 1));

        if (is_new && (uint32_t)lane_rank < write_n) {
            const uint32_t out_idx = write_idx_base + (uint32_t)lane_rank;
            distances[neighbor_v] = new_dist;
            next_frontier[out_idx] = neighbor_v;
        }

        if (lane_id == 0 && (uint32_t)warp_total_new > write_n) {
            atomicMax(next_count, next_capacity);
        }
    }
}

extern "C"
void launch_bfs_expand(const Job* jobs, uint32_t num_jobs,
                       const uint32_t* row_ptr, const uint32_t* nbrs,
                       uint32_t* next_frontier, uint32_t* next_count,
                       uint32_t  next_capacity,
                       uint32_t* visited_bitmap, uint32_t visited_words,
                       uint32_t* distances,
                       CUstream stream)
{
    if (num_jobs == 0) return;

    constexpr int threads_per_block = 1024;
    constexpr int warps_per_block   = threads_per_block / 32; 

    const int blocks = (int)((num_jobs + warps_per_block - 1) / warps_per_block);

    bfs_expand_kernel<<<blocks, threads_per_block, 0, stream>>>(
        jobs, num_jobs,
        row_ptr, nbrs,
        next_frontier, next_count, next_capacity,
        visited_bitmap, visited_words,
        distances
    );
}

