#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "shared.h"


__global__ void k_iota_u32(uint32_t* out, uint32_t n) {
    uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < n) out[tid] = tid;
}

extern "C" void launch_iota_u32(uint32_t* d_nodes, uint32_t n, cudaStream_t s) {
    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    if (grid > 65535) grid = 65535;
    k_iota_u32<<<grid, block, 0, s>>>(d_nodes, n);
}

__global__ void k_set_u32(uint32_t* p, uint32_t v) { *p = v; }

extern "C" void launch_set_u32(uint32_t* d_ptr, uint32_t v, cudaStream_t s) {
    k_set_u32<<<1,1,0,s>>>(d_ptr, v);
}

__global__ void k_cdlp_init_labels(uint32_t* labels, uint32_t n) {
    uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < n) labels[tid] = tid;
}

extern "C" void launch_cdlp_init_labels(uint32_t* d_labels, uint32_t n, cudaStream_t s) {
    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    if (grid > 65535) grid = 65535;
    k_cdlp_init_labels<<<grid, block, 0, s>>>(d_labels, n);
}

__global__ void k_copy_u32(uint32_t* dst, const uint32_t* src, uint32_t n) {
    uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < n) dst[tid] = src[tid];
}

extern "C" void launch_copy_u32(uint32_t* dst, const uint32_t* src, uint32_t n, cudaStream_t s) {
    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    if (grid > 65535) grid = 65535;
    k_copy_u32<<<grid, block, 0, s>>>(dst, src, n);
}

static __device__ __forceinline__ void map_init(uint32_t* keys, uint32_t* cnts) {
    for (int i = 0; i < 16; ++i) { keys[i] = 0xFFFFFFFFu; cnts[i] = 0u; }
}

static __device__ __forceinline__ void map_add(uint32_t* keys, uint32_t* cnts, uint32_t key) {

    for (int i = 0; i < 16; ++i) {
        uint32_t k = keys[i];
        if (k == key) { cnts[i]++; return; }
        if (k == 0xFFFFFFFFu) { keys[i] = key; cnts[i] = 1u; return; }
    }
}

static __device__ __forceinline__ uint32_t map_argmax_label(const uint32_t* keys, const uint32_t* cnts, uint32_t fallback) {
    uint32_t bestLabel = fallback;
    uint32_t bestCnt   = 0u;

    for (int i = 0; i < 16; ++i) {
        uint32_t k = keys[i];
        uint32_t c = cnts[i];
        if (k == 0xFFFFFFFFu) continue;
        if (c > bestCnt || (c == bestCnt && k < bestLabel)) {
            bestCnt = c;
            bestLabel = k;
        }
    }
    return bestLabel;
}

__global__ void k_cdlp_iterate_nodes(
    const uint32_t* __restrict__ nodes,
    uint32_t num_nodes,
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ nbrs,
    const uint32_t* __restrict__ labels_curr,
    uint32_t* __restrict__ labels_next,
    uint32_t* __restrict__ changed)
{
    uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= num_nodes) return;

    const uint32_t u = nodes[tid];

    const uint32_t beg = __ldg(&row_ptr[u]);
    const uint32_t end = __ldg(&row_ptr[u + 1]);

    const uint32_t cur = __ldg(&labels_curr[u]);

    if (end <= beg) {
        labels_next[u] = cur;
        return;
    }

    uint32_t keys[16];
    uint32_t cnts[16];
    map_init(keys, cnts);

    for (uint32_t off = beg; off < end; ++off) {
        const uint32_t v = __ldg(&nbrs[off]);
        const uint32_t lbl = __ldg(&labels_curr[v]);
        map_add(keys, cnts, lbl);
    }

    const uint32_t next = map_argmax_label(keys, cnts, cur);
    labels_next[u] = next;

    if (next != cur) atomicAdd(changed, 1u);
}

extern "C" void launch_cdlp_iterate_nodes(
    const uint32_t* d_nodes,
    uint32_t num_nodes,
    const uint32_t* row_ptr,
    const uint32_t* nbrs,
    const uint32_t* labels_curr,
    uint32_t* labels_next,
    uint32_t* changed,
    cudaStream_t s)
{
    int block = 256;
    int grid  = (int)((num_nodes + block - 1) / block);
    if (grid > 65535) grid = 65535;

    k_cdlp_iterate_nodes<<<grid, block, 0, s>>>(
        d_nodes, num_nodes, row_ptr, nbrs, labels_curr, labels_next, changed);
}
