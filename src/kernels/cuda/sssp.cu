#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "shared.h"


struct SSSPFrontierView {
    const uint32_t* __restrict__ row_ptr;
    const uint32_t* __restrict__ nbrs;
};

__global__ void sssp_relax_frontier_kernel(
    SSSPFrontierView g,
    const uint32_t* __restrict__ frontier,  
    uint32_t frontier_size,
    float* __restrict__ dist,               
    uint32_t* __restrict__ next_frontier,  
    uint32_t* __restrict__ next_count)      
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    const uint32_t u = frontier[tid];
    const float du   = dist[u];
    const uint32_t beg = g.row_ptr[u];
    const uint32_t end = g.row_ptr[u + 1];

    const float cand_base = du + 1.0f;

    for (uint32_t k = beg; k < end; ++k) {
        const uint32_t v = g.nbrs[k];

        float old = __int_as_float(atomicMin(
            reinterpret_cast<int*>(dist + v),
            __float_as_int(cand_base)
        ));

        if (old > cand_base) {
            const uint32_t idx = atomicAdd(next_count, 1u);
            next_frontier[idx] = v;
        }
    }
}

extern "C" void launch_sssp_relax_frontier(
    const uint32_t* d_row_ptr,
    const uint32_t* d_nbrs,
    const uint32_t* d_frontier,
    uint32_t frontier_size,
    float* d_dist,
    uint32_t* d_next_frontier,
    uint32_t* d_next_count,
    CUstream s)
{
    if (frontier_size == 0) return;
    const int TPB = 128;
    const int blocks = (frontier_size + TPB - 1) / TPB;
    SSSPFrontierView gv{ d_row_ptr, d_nbrs };
    sssp_relax_frontier_kernel<<<blocks, TPB, 0, s>>>(
        gv, d_frontier, frontier_size, d_dist, d_next_frontier, d_next_count
    );
}

