#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <type_traits>
#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>


// common CUDA constants
#define WARPSIZE (32)
#define MAXBLOCKSIZE (1024)
#define MAXSMEMBYTES (49152)
#define MAXCONSTMEMBYTES (65536)
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define H2H (cudaMemcpyHostToHost)
#define D2D (cudaMemcpyDeviceToDevice)

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

// cross platform classifiers
#ifdef __CUDACC__
    #define HOSTDEVICEQUALIFIER  __host__ __device__
#else
    #define HOSTDEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define INLINEQUALIFIER  __forceinline__
#else
    #define INLINEQUALIFIER inline
#endif

#ifdef __CUDACC__
    #define GLOBALQUALIFIER  __global__
#else
    #define GLOBALQUALIFIER
#endif

#ifdef __CUDACC__
    #define DEVICEQUALIFIER  __device__
#else
    #define DEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define HOSTQUALIFIER  __host__
#else
    #define HOSTQUALIFIER
#endif

#ifdef __CUDACC__
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }
#endif

#ifdef __CUDACC__
    // only valid for linear kernel i.e. y = z = 0
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t global_thread_id() noexcept
    {
        return
            std::uint64_t(blockDim.x) * std::uint64_t(blockIdx.x) +
            std::uint64_t(threadIdx.x);
    }
#endif

// redefinition of CUDA atomics for common cstdint types
#ifdef __CUDACC__
    // CAS
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicCAS(
        std::uint64_t* address,
        std::uint64_t compare,
        std::uint64_t val)
    {
        return atomicCAS(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(compare),
            static_cast<unsigned long long int>(val));
    }

    // Add
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicAdd(std::uint64_t* address, std::uint64_t val)
    {
        return atomicAdd(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // Exch
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicExch(std::uint64_t* address, std::uint64_t val)
    {
        return atomicExch(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // Min
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicMin(std::uint64_t* address, std::uint64_t val)
    {
        return atomicMin(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // Max
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicMax(std::uint64_t* address, std::uint64_t val)
    {
        return atomicMax(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // AND
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicAnd(std::uint64_t* address, std::uint64_t val)
    {
        return atomicAnd(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // OR
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicOr(std::uint64_t* address, std::uint64_t val)
    {
        return atomicOr(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    // XOR
    DEVICEQUALIFIER INLINEQUALIFIER
    std::uint64_t atomicXor(std::uint64_t* address, uint64_t val)
    {
        return atomicXor(
            reinterpret_cast<unsigned long long int*>(address),
            static_cast<unsigned long long int>(val));
    }

    #ifdef __CUDACC_EXTENDED_LAMBDA__
    template<class T>
    GLOBALQUALIFIER void lambda_kernel(T f)
    {
        f();
    }
    #endif

    DEVICEQUALIFIER INLINEQUALIFIER
    unsigned int lane_id()
    {
        unsigned int lane;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
        return lane;
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    int ffs(std::uint32_t x)
    {
        return __ffs(x);
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    int ffs(std::uint64_t x)
    {
        return __ffsll(x);
    }

    HOSTQUALIFIER INLINEQUALIFIER
    void init_cuda_context()
    {
        cudaFree(0);
    }

    #if CUDART_VERSION >= 9000
        #include <cooperative_groups.h>

        template<typename index_t>
        DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggInc(index_t * ctr)
        {
            using namespace cooperative_groups;
            coalesced_group g = coalesced_threads();
            index_t prev;
            if (g.thread_rank() == 0) {
                prev = atomicAdd(ctr, g.size());
            }
            prev = g.thread_rank() + g.shfl(prev, 0);
            return prev;
        }
    #else
        template<typename index_t>
        DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggInc(index_t * ctr)
        {
            int lane = lane_id();
            //check if thread is active
            int mask = __ballot(1);
            //determine first active lane for atomic add
            int leader = __ffs(mask) - 1;
            index_t res;
            if (lane == leader) res = atomicAdd(ctr, __popc(mask));
            //broadcast to warp
            res = __shfl(res, leader);
            //compute index for each thread
            return res + __popc(mask & ((1 << lane) -1));
        }
    #endif
#endif

#endif /*CUDA_HELPERS_CUH*/
