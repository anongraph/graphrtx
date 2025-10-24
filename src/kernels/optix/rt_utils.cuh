#ifndef RT_UTILS_CUH
#define RT_UTILS_CUH
#include <optix_device.h>
#include <math_constants.h>
#include "shared.h"

extern "C" __constant__ Params params;

static __forceinline__ __device__
uint32_t u32_lower_bound(const uint32_t* __restrict__ a, uint32_t n, uint32_t key)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        uint32_t x = __ldg(&a[mid]);
        if (x < key) lo = mid + 1;
        else         hi = mid;
    }
    return lo;
}

__device__ inline float atomicMinFloat(float* addr, float val)
{
    int* ai = reinterpret_cast<int*>(addr);
    int old_int = *ai;
    while (true) {
        float old_float = __int_as_float(old_int);
        if (!(val < old_float)) return old_float;
        int assumed = old_int;
        int desired = __float_as_int(val);
        old_int = atomicCAS(ai, assumed, desired);
        if (old_int == assumed) return old_float;
    }
}

static __forceinline__ __device__
uint32_t global_prim_index()
{
    const uint32_t primLocal = optixGetPrimitiveIndex();
    if (params.num_instances <= 1 || params.instance_prim_bases == nullptr)
        return primLocal;

    const uint32_t instId = optixGetInstanceId();
    const uint32_t* bases = params.instance_prim_bases;
    const uint32_t base = bases[instId];
    return base + primLocal;
}


static __forceinline__ __device__
uint32_t count_common_bitset_safe(const uint32_t* __restrict__ A, uint32_t lenA,
                                  const uint32_t* __restrict__ B, uint32_t lenB)
{
    uint32_t minv = (lenA ? __ldg(&A[0]) : 0);
    if (lenB) minv = min(minv, __ldg(&B[0]));
    uint64_t mA = 0ull, mB = 0ull;
    for (uint32_t i = 0; i < lenA; ++i) mA |= (1ull << ( (__ldg(&A[i])) - minv ));
    for (uint32_t i = 0; i < lenB; ++i) mB |= (1ull << ( (__ldg(&B[i])) - minv ));
    return __popcll(mA & mB);
}

static __forceinline__ __device__
uint32_t count_common_scalar(const uint32_t* __restrict__ A, uint32_t lenA,
                             const uint32_t* __restrict__ B, uint32_t lenB)
{
    if (lenA == 0 || lenB == 0) return 0;

    // Use binary search only with strong imbalance to reduce wasted probes.
    const bool binsearch = (lenA * 8u < lenB) || (lenB * 8u < lenA);
    uint32_t c = 0;

    if (binsearch) {
        const bool aSmall = (lenA <= lenB);
        const uint32_t* __restrict__ small  = aSmall ? A : B;
        const uint32_t* __restrict__ big    = aSmall ? B : A;
        const uint32_t  smallN = aSmall ? lenA : lenB;
        const uint32_t  bigN   = aSmall ? lenB : lenA;

        uint32_t base = 0;
        for (uint32_t i = 0; i < smallN; ++i) {
            const uint32_t key = __ldg(&small[i]);
            uint32_t lo = base, hi = bigN;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                const uint32_t x = __ldg(&big[mid]);
                if (x < key) lo = mid + 1;
                else         hi = mid;
            }
            base = lo;
            if (base < bigN && __ldg(&big[base]) == key) ++c;
        }
    } else {
        uint32_t i = 0, j = 0;
        while (i < lenA && j < lenB) {
            const uint32_t a = __ldg(&A[i]);
            const uint32_t b = __ldg(&B[j]);
            if (a < b)      ++i;
            else if (a > b) ++j;
            else { ++c; ++i; ++j; }
        }
    }
    return c;
}

static __forceinline__ __device__
uint32_t count_common_gt_partner(uint32_t partner,
                                 const uint32_t* __restrict__ seg, uint32_t seg_len,
                                 const uint32_t* __restrict__ row_ptr,
                                 const uint32_t* __restrict__ nbrs)
{
    // Fast exits
    if (seg_len == 0) return 0;

    const uint32_t p_beg = __ldg(&row_ptr[partner]);
    const uint32_t p_end = __ldg(&row_ptr[partner + 1]);
    const uint32_t p_len = p_end - p_beg;
    if (p_len == 0) return 0;

    // If the segment's max element <= partner, no elements greater than partner exist.
    if (__ldg(&seg[seg_len - 1]) <= partner) return 0;

    // Lower-bound both sides to > partner
    const uint32_t s_off = u32_lower_bound(seg, seg_len, partner + 1);
    if (s_off >= seg_len) return 0;
    const uint32_t* __restrict__ S = seg + s_off;
    const uint32_t S_len = seg_len - s_off;

    const uint32_t* __restrict__ P = nbrs + p_beg;
    const uint32_t p_off = u32_lower_bound(P, p_len, partner + 1);
    if (p_off >= p_len) return 0;
    const uint32_t* __restrict__ PN = P + p_off;
    const uint32_t PN_len = p_len - p_off;

    // Small ranges → cheap bitset intersection
    if (S_len <= 32 && PN_len <= 32) {
        const uint32_t minv = min(__ldg(&S[0]), __ldg(&PN[0]));
        const uint32_t maxv = max(__ldg(&S[S_len - 1]), __ldg(&PN[PN_len - 1]));
        if (maxv - minv < 64)
            return count_common_bitset_safe(S, S_len, PN, PN_len);
    }

    // Otherwise: scalar with tuned heuristic and cached loads
    return count_common_scalar(S, S_len, PN, PN_len);
}

#endif
