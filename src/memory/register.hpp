#ifndef DEVICE_REGISTER_HPP
#define DEVICE_REGISTER_HPP
#include <cuda_runtime.h>
#include "../common.hpp"

struct PinnedRegister { 
    void* p {nullptr};
    size_t bytes {0};
    bool active {false};
    PinnedRegister() = default;
    PinnedRegister(void* ptr, size_t nbytes) { registerRange(ptr, nbytes); }
    void registerRange(void* ptr, size_t nbytes) {
        if (ptr && nbytes) {
            CUDA_CHECK(cudaHostRegister(ptr, nbytes, cudaHostRegisterPortable));
            p = ptr; bytes = nbytes; active = true;
        }
    }
    ~PinnedRegister() { if (active) cudaHostUnregister(p); }
    PinnedRegister(const PinnedRegister&) = delete;
    PinnedRegister& operator=(const PinnedRegister&) = delete;
};

#endif