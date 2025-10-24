
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "shared.h"

static __device__ __forceinline__ uint32_t bsearch_in(const uint32_t* arr, uint32_t n, uint32_t key){
    int lo = 0, hi = (int)n - 1;
    while (lo <= hi){
        int mid = (lo + hi) >> 1;
        uint32_t x = __ldg(&arr[mid]);
        if (x == key) return 1u;
        lo = (x < key) ? (mid + 1) : lo;
        hi = (x > key) ? (mid - 1) : hi;
    }
    return 0u;
}

static __forceinline__ __device__
uint32_t lb_u32(const uint32_t* __restrict__ a, uint32_t n, uint32_t key)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        const uint32_t x = __ldg(&a[mid]);
        if (x < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static __forceinline__ __device__
uint32_t gallop_lb_from(const uint32_t* __restrict__ a, uint32_t n,
                        uint32_t key, uint32_t base)
{
    if (base >= n) return n;
    if (__ldg(&a[base]) >= key) return base;

    uint32_t step = 1;
    uint32_t lo = base + 1;
    uint32_t hi = lo;

    while (hi < n && __ldg(&a[hi]) < key) {
        lo = hi;
        const uint32_t nh = hi + step;
        hi = nh < n ? nh : n;
        step <<= 1;
    }
    
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1);
        const uint32_t x = __ldg(&a[mid]);
        if (x < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static __forceinline__ __device__
uint32_t contains_with_base(const uint32_t* __restrict__ B, uint32_t B_len,
                            uint32_t key, uint32_t &base)
{
    base = gallop_lb_from(B, B_len, key, base);
    return (base < B_len && __ldg(&B[base]) == key) ? 1u : 0u;
}

extern "C" __global__
void tc_join_warp_coop_kernel(const Job* __restrict__ jobs,
                              uint32_t                 num_jobs,
                              const uint32_t* __restrict__ row_ptr,
                              const uint32_t* __restrict__ nbrs,
                              unsigned long long* tri_count)
{
    const unsigned gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane    = threadIdx.x & 31u;
    const unsigned warp_id = gtid >> 5;
    const unsigned mask    = __activemask();

    if (warp_id >= num_jobs) return;

    Job j;
    if (lane == 0) j = jobs[warp_id];
    j.qtype   = (QueryType)__shfl_sync(mask, (int)j.qtype, 0);
    j.src     = __shfl_sync(mask, j.src,     0);
    j.partner = __shfl_sync(mask, j.partner, 0);

    const uint32_t u = j.src;
    const uint32_t v = j.partner;

    const uint32_t u_beg = __ldg(&row_ptr[u]);
    const uint32_t u_end = __ldg(&row_ptr[u + 1]);
    const uint32_t v_beg = __ldg(&row_ptr[v]);
    const uint32_t v_end = __ldg(&row_ptr[v + 1]);

    const uint32_t len_u = u_end - u_beg;
    const uint32_t len_v = v_end - v_beg;
    if (len_u == 0 || len_v == 0) return;

    const bool       a_is_u = (len_u <= len_v);
    const uint32_t*  A      = a_is_u ? (nbrs + u_beg) : (nbrs + v_beg);
    const uint32_t*  B      = a_is_u ? (nbrs + v_beg) : (nbrs + u_beg);
    const uint32_t   A_len  = a_is_u ?  len_u         :  len_v;
    const uint32_t   B_len  = a_is_u ?  len_v         :  len_u;

    uint32_t i0 = 0;
    if (lane == 0) i0 = lb_u32(A, A_len, v + 1);
    i0 = __shfl_sync(mask, i0, 0);
    if (i0 >= A_len) return;

    uint32_t local = 0;
    uint32_t base  = 0; 

    for (uint32_t idx = i0 + lane; idx < A_len; idx += 32) {
        const uint32_t key = __ldg(&A[idx]);
        local += contains_with_base(B, B_len, key, base);
    }

    unsigned warp_sum = local;
    #pragma unroll
    for (int offs = 16; offs > 0; offs >>= 1)
        warp_sum += __shfl_down_sync(mask, warp_sum, offs);

    if (lane == 0 && warp_sum)
        atomicAdd(tri_count, (unsigned long long)warp_sum);
}


extern "C" void launch_tc_join_warp_coop(const Job* d_jobs,
                                         uint32_t   num_jobs,
                                         const uint32_t* d_row_ptr,
                                         const uint32_t* d_nbrs,
                                         unsigned long long* d_tri_count)
{
    if (num_jobs == 0) return;
    const int warps   = (int)num_jobs;
    const int threads = 512;
    const int blocks  = (warps * 64 + threads - 1) / threads;
    tc_join_warp_coop_kernel<<<blocks, threads>>>(d_jobs, num_jobs, d_row_ptr, d_nbrs, d_tri_count);
    cudaDeviceSynchronize();
}