
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "shared.h"


extern "C" __global__
void pr_scatter_items_kernel(
    const PRWorkItem* __restrict__ items,
    uint32_t                      num_items,
    const uint32_t* __restrict__ nbrs,
    float* __restrict__ pr_next)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_items; i += blockDim.x * gridDim.x) {
        const PRWorkItem w = items[i];
        const float c = w.contrib;
        const uint32_t end = w.start + w.len;
        for (uint32_t k = w.start; k < end; ++k) {
            const uint32_t v = nbrs[k];
            atomicAdd(&pr_next[v], c);
        }
    }
}

extern "C" __global__
void pr_scatter_items_kernel_warp(
    const PRWorkItem* __restrict__ items,
    uint32_t                      num_items,
    const uint32_t*  __restrict__ nbrs,
    float*            __restrict__ pr_next)
{
    const uint32_t gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t warp_id = gtid >> 5;                 
    if (warp_id >= num_items) return;

    const unsigned mask = __activemask();

    PRWorkItem w;
    if (lane == 0) w = items[warp_id];
    const uint32_t start = __shfl_sync(mask, w.start,   0);
    const uint32_t len   = __shfl_sync(mask, w.len,     0);
    const float    c     = __shfl_sync(mask, w.contrib, 0);

    for (uint32_t k = lane; k < len; k += 32) {
        const uint32_t v = __ldg(&nbrs[start + k]);
        atomicAdd(&pr_next[v], c);
    }
}

extern "C"
void launch_pr_scatter_items(const PRWorkItem* items, uint32_t num_items,
                             const uint32_t* nbrs, float* pr_next, CUstream stream)
{
    if (num_items == 0) return;

    
    constexpr int WARPS_PER_BLOCK = 8;   
    constexpr int BLOCK = WARPS_PER_BLOCK * 32;
    const uint32_t num_warps = (num_items + 0u) ; 
    const uint32_t blocks = ( (num_items + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK );

    pr_scatter_items_kernel_warp<<<blocks, BLOCK, 0, stream>>>(items, num_items, nbrs, pr_next);
}


extern "C" __global__
void pr_reduce_dangling_kernel(
    const float* __restrict__ pr_curr,
    const float* __restrict__ inv_outdeg,
    int N,
    float* __restrict__ out_sum)
{
    __shared__ float s[256];
    float local = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        if (inv_outdeg[i] == 0.0f) local += pr_curr[i];
    }
    const int tid = threadIdx.x;
    s[tid] = local; __syncthreads();
    for (int off = blockDim.x>>1; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid+off];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out_sum, s[0]);
}

extern "C"
void launch_pr_reduce_dangling(const float* pr_curr, const float* inv_outdeg,
                               int N, float* d_out, CUstream stream)
{
    const int block = 256;
    const int grid  = std::min( (N + block - 1)/block, 65535 );
    pr_reduce_dangling_kernel<<<grid, block, 0, stream>>>(pr_curr, inv_outdeg, N, d_out);
}

extern "C" __global__
void pr_fill_base_kernel(float* pr_next, int N, float base_term) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        pr_next[i] = base_term;
    }
}

extern "C"
void launch_pr_fill_base(float* pr_next, int N, float base_term, CUstream stream)
{
    const int block = 256;
    const int grid  = std::min( (N + block - 1)/block, 65535 );
    pr_fill_base_kernel<<<grid, block, 0, stream>>>(pr_next, N, base_term);
}

__global__ void pr_build_items_kernel(
    const float* __restrict__ pr_curr,
    const float* __restrict__ inv_outdeg,
    const uint32_t* __restrict__ row_ptr,
    int N,
    PRWorkItem* __restrict__ items,
    float damping)
{
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if (u >= N) return;

    uint32_t start = row_ptr[u];
    uint32_t end   = row_ptr[u + 1];
    uint32_t len   = end - start;

    float invdeg  = inv_outdeg[u];
    float pr      = pr_curr[u];
    float contrib = (invdeg > 0.0f) ? damping * pr * invdeg : 0.0f;

    items[u] = { start, len, contrib };
}

__global__ void pr_diff_norm_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n,
    float* __restrict__ out_sum)
{
    extern __shared__ float sdata[];
    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;

    float v = 0.0f;
    if (idx < n) v = fabsf(a[idx] - b[idx]);
    sdata[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out_sum, sdata[0]);
}

extern "C"  void launch_pr_build_items(
    const float* d_pr_curr,
    const float* d_inv_outdeg,
    const uint32_t* d_row_ptr,
    int N,
    PRWorkItem* d_items,
    float damping,
    CUstream s)
{
    const int TPB = 256;
    const int blocks = (N + TPB - 1) / TPB;
    pr_build_items_kernel<<<blocks, TPB, 0, s>>>(d_pr_curr, d_inv_outdeg, d_row_ptr, N, d_items, damping);
}

extern "C"  void launch_pr_diff_norm(
    const float* d_a,
    const float* d_b,
    int n,
    float* d_out_sum,
    CUstream s)
{
    const int TPB = 256;
    const int blocks = (n + TPB - 1) / TPB;
    const size_t shmem = TPB * sizeof(float);
    pr_diff_norm_kernel<<<blocks, TPB, shmem, s>>>(d_a, d_b, n, d_out_sum);
}