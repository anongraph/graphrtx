#ifndef DEVICE_BUFFER_HPP
#define DEVICE_BUFFER_HPP
#include <cuda_runtime.h>
#include "../common.hpp"

template <typename T>
struct DeviceBuffer {
    T* ptr {nullptr};
    size_t count {0};
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { allocate(n); }
    void allocate(size_t n) {
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        count = n;
    }
    void uploadAsync(const T* host, size_t n, cudaStream_t s) {
        if (!ptr || !host || n == 0) return;
        CUDA_CHECK(cudaMemcpyAsync(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice, s));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept { ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; }
        return *this;
    }
};

#endif