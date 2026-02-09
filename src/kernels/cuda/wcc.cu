#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

__global__ void k_find_next_unassigned_u32(const uint32_t* comp, uint32_t n, uint32_t* out_min_idx)
{
    const uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t stride = (uint32_t)(blockDim.x * gridDim.x);

    for (uint32_t i = tid; i < n; i += stride) {
        if (comp[i] == 0xFFFFFFFFu) {
            atomicMin(out_min_idx, i);
        }
    }
}

extern "C" void launch_find_next_unassigned_u32(const uint32_t* d_comp, uint32_t n, uint32_t* d_out, cudaStream_t s)
{
    const uint32_t init = 0xFFFFFFFFu;
    cudaMemcpyAsync(d_out, &init, sizeof(uint32_t), cudaMemcpyHostToDevice, s);

    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    if (grid > 4096) grid = 4096;
    k_find_next_unassigned_u32<<<grid, block, 0, s>>>(d_comp, n, d_out);
}

__global__ void k_wcc_set_seed(uint32_t* comp, uint32_t src, uint32_t comp_id)
{
    atomicCAS(&comp[src], 0xFFFFFFFFu, comp_id);
}

extern "C" void launch_wcc_set_seed(uint32_t* d_comp, uint32_t src, uint32_t comp_id, cudaStream_t s)
{
    k_wcc_set_seed<<<1,1,0,s>>>(d_comp, src, comp_id);
}

__global__ void k_wcc_expand_frontier(
    const uint32_t* __restrict__ frontier,
    uint32_t frontier_size,
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ nbrs,
    uint32_t* __restrict__ comp,
    uint32_t comp_id,
    uint32_t* __restrict__ next_frontier,
    uint32_t* __restrict__ next_count,
    uint32_t next_capacity)
{
    const uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= frontier_size) return;

    const uint32_t u = frontier[tid];
    const uint32_t beg = __ldg(&row_ptr[u]);
    const uint32_t end = __ldg(&row_ptr[u + 1]);

    for (uint32_t off = beg; off < end; ++off) {
        const uint32_t v = __ldg(&nbrs[off]);
        const uint32_t old = atomicCAS(&comp[v], 0xFFFFFFFFu, comp_id);
        if (old == 0xFFFFFFFFu) {
            const uint32_t idx = atomicAdd(next_count, 1u);
            if (idx < next_capacity) next_frontier[idx] = v;
            else atomicSub(next_count, 1u);
        }
    }
}

extern "C" void launch_wcc_expand_frontier(
    const uint32_t* d_frontier,
    uint32_t frontier_size,
    const uint32_t* row_ptr,
    const uint32_t* nbrs,
    uint32_t* d_comp,
    uint32_t comp_id,
    uint32_t* d_next_frontier,
    uint32_t* d_next_count,
    uint32_t next_capacity,
    cudaStream_t s)
{
    const int block = 256;
    int grid = (int)((frontier_size + block - 1) / block);
    if (grid > 65535) grid = 65535;

    k_wcc_expand_frontier<<<grid, block, 0, s>>>(
        d_frontier, frontier_size, row_ptr, nbrs,
        d_comp, comp_id,
        d_next_frontier, d_next_count, next_capacity);
}