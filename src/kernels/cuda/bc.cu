
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <cstdio>

#include "shared.h"


// -----------------------------------------------------------------------------
extern "C" __global__
void add_delta_to_bc_kernel_safe(const uint32_t* __restrict__ nodes,
                                 uint32_t count,
                                 int source,
                                 const float* __restrict__ delta,
                                 float* __restrict__ bc,
                                 uint32_t N)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint32_t u = nodes[i];
    if (u >= N || u == static_cast<uint32_t>(source))
        return;

    const float d = delta[u];
    if (d != 0.0f) {
        atomicAdd(&bc[u], d);
    }
}

extern "C"
void launch_add_delta_to_bc(const uint32_t* nodes,
                            uint32_t count,
                            int source,
                            const float* delta,
                            float* bc,
                            uint32_t N,
                            CUstream stream)
{
    if (count == 0) return;

    constexpr int block = 128;
    const int grid = (count + block - 1) / block;

    add_delta_to_bc_kernel_safe<<<grid, block, 0, stream>>>(
        nodes, count, source, delta, bc, N);
}

extern "C" __global__
void memset_kernel_u32(uint32_t* arr, uint32_t value, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = value;
}

extern "C"
void launch_memset_u32(uint32_t* arr, uint32_t value, uint32_t n, CUstream stream)
{
    if (n == 0) return;
    int block = 256;
    int grid  = (n + block - 1) / block;
    memset_kernel_u32<<<grid, block, 0, stream>>>(arr, value, n);
}

extern "C" __global__
void memset_kernel_f32(float* arr, float value, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = value;
}

extern "C"
void launch_memset_f32(float* arr, float value, uint32_t n, CUstream stream)
{
    if (n == 0) return;
    int block = 256;
    int grid  = (n + block - 1) / block;
    memset_kernel_f32<<<grid, block, 0, stream>>>(arr, value, n);
}

extern "C" __global__
void build_jobs_from_nodes_kernel(const uint32_t* __restrict__ nodes,
                                  uint32_t                     count,
                                  Job* __restrict__ jobs,
                                  uint8_t                      qtype )
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Job j{};
    j.qtype = (QueryType)qtype;
    j.src   = nodes[i];
    jobs[i] = j;
}

extern "C" void launch_build_jobs_from_nodes(const uint32_t* nodes,
                                             uint32_t        count,
                                             Job* jobs,
                                             QueryType       qtype,
                                             CUstream        stream)
{
    if (count == 0) return;
    const int block = 256;
    const int grid  = (count + block - 1) / block;
    build_jobs_from_nodes_kernel<<<grid, block, 0, stream>>>(nodes, count, jobs, (uint8_t)qtype);
}

extern "C" __global__
void d2d_copy_u32_kernel(const uint32_t* __restrict__ src,
                         uint32_t* __restrict__ dst,
                         uint32_t                      count)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[i];
}

extern "C" void launch_d2d_copy_u32(const uint32_t* src,
                                    uint32_t* dst,
                                    uint32_t        count,
                                    CUstream        stream)
{
    if (count == 0) return;
    const int block = 256;
    const int grid  = (count + block - 1) / block;
    d2d_copy_u32_kernel<<<grid, block, 0, stream>>>(src, dst, count);
}

extern "C" __global__
void bc_forward_expand_nodes_kernel(
    const uint32_t* __restrict__ frontier, uint32_t frontier_size,
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ nbrs,
    uint32_t*       __restrict__ next_frontier,
    uint32_t*       __restrict__ next_count,
    uint32_t*       __restrict__ dist,
    uint32_t*       __restrict__ sigma,
    uint32_t                    cur_depth)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    const uint32_t u   = frontier[tid];
    const uint32_t beg = row_ptr[u];
    const uint32_t end = row_ptr[u+1];
    const uint32_t su  = sigma[u];      // σ(u)
    const uint32_t du1 = cur_depth + 1; // depth+1

    for (uint32_t k = beg; k < end; ++k) {
        const uint32_t w = __ldg(&nbrs[k]);
        
        uint32_t old = atomicCAS(&dist[w], 0xFFFFFFFFu, du1);
        if (old == 0xFFFFFFFFu) {
           
            const uint32_t idx = atomicAdd(next_count, 1u);
            next_frontier[idx] = w;
        }
        
        if (__ldg(&dist[w]) == du1) {
            atomicAdd(&sigma[w], su);
        }
    }
}

extern "C" void launch_bc_forward_expand_nodes(
    const uint32_t* frontier, uint32_t frontier_size,
    const uint32_t* row_ptr,   const uint32_t* nbrs,
    uint32_t* next_frontier,   uint32_t* next_count,
    uint32_t* dist,            uint32_t* sigma,
    uint32_t cur_depth,        CUstream stream)
{
    if (frontier_size == 0) return;
    const int block = 256;
    const int grid  = (frontier_size + block - 1) / block;
    bc_forward_expand_nodes_kernel<<<grid, block, 0, stream>>>(
        frontier, frontier_size, row_ptr, nbrs,
        next_frontier, next_count, dist, sigma, cur_depth
    );
}

extern "C" __global__
void bc_backward_accumulate_nodes_kernel(
    const uint32_t* __restrict__ level_nodes, uint32_t count,
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ nbrs,
    const uint32_t* __restrict__ dist,
    const uint32_t* __restrict__ sigma,
    const float*    __restrict__ delta,  
    float*          __restrict__ delta_out) 
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const uint32_t u   = level_nodes[tid];
    const uint32_t beg = row_ptr[u];
    const uint32_t end = row_ptr[u+1];
    const uint32_t du  = dist[u];
    const uint32_t su  = sigma[u];

    if (su == 0u) return; 

    float acc = 0.0f;
    for (uint32_t k = beg; k < end; ++k) {
        const uint32_t w = __ldg(&nbrs[k]);
        if (__ldg(&dist[w]) == du + 1u) {
            const uint32_t sw = __ldg(&sigma[w]);
            if (sw > 0u) {
                const float dw = __ldg(&delta[w]);
                acc += (float(su) / float(sw)) * (1.0f + dw);
            }
        }
    }
    if (acc != 0.0f) atomicAdd(&delta_out[u], acc);
}

extern "C" void launch_bc_backward_accumulate_nodes(
    const uint32_t* level_nodes, uint32_t count,
    const uint32_t* row_ptr,     const uint32_t* nbrs,
    const uint32_t* dist,        const uint32_t* sigma,
    const float*    delta,       float* delta_out,
    CUstream        stream)
{
    if (count == 0) return;
    const int block = 256;
    const int grid  = (count + block - 1) / block;
    bc_backward_accumulate_nodes_kernel<<<grid, block, 0, stream>>>(
        level_nodes, count, row_ptr, nbrs, dist, sigma, delta, delta_out
    );
}
