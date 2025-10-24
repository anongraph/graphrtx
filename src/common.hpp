#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <future>
#include <vector>

#ifndef DEVICE_PTX_PATH
#define DEVICE_PTX_PATH "device.ptx"
#endif

// ------------------- error helpers -------------------
#define CUDA_CHECK(x) do { cudaError_t e = (x); if(e != cudaSuccess){ \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) \
              << " @ " << __FILE__ << ":" << __LINE__ << "\n"; std::exit(1);} } while(0)
  #define OPTIX_CHECK(x) do { OptixResult r = (x); if(r != OPTIX_SUCCESS){ \
    std::cerr << "OptiX error " << (int)r \
              << " @ " << __FILE__ << ":" << __LINE__ << "\n"; std::exit(2);} } while(0)
  
  static void optixLogCb(unsigned l,const char*t,const char*m,void*) {
    std::cerr << "[OptiX]["<<l<<"]["<<(t?t:"")<<"] "<<(m?m:"")<<"\n";
  }

// ------------------- timing helper -------------------
static float elapsedMs(cudaEvent_t a,cudaEvent_t b){float ms=0;cudaEventElapsedTime(&ms,a,b);return ms;}

// ------------------- thread helper -------------------
template <typename F>
void parallel_for(int start, int end, int num_threads, F&& fn) {
    std::vector<std::future<void>> futures;
    int chunk = (end - start + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int s = start + t * chunk;
        int e = std::min(end, s + chunk);
        if (s >= e) break;

        futures.emplace_back(std::async(std::launch::async, [=, &fn]() {
            for (int i = s; i < e; ++i)
                fn(i);
        }));
    }

    for (auto& f : futures) f.get();
}

static inline double toMB(size_t bytes) { return double(bytes) / (1024.0 * 1024.0); }
static inline double toGB(size_t bytes) { return double(bytes) / (1024.0 * 1024.0 * 1024.0); }

struct ScopedStream {
  cudaStream_t s{nullptr};
  explicit ScopedStream(unsigned flags = cudaStreamNonBlocking) { CUDA_CHECK(cudaStreamCreateWithFlags(&s, flags)); }
  ~ScopedStream() { if (s) cudaStreamDestroy(s); }
  operator cudaStream_t() const { return s; }
};

struct ScopedEvent {
  cudaEvent_t evt{nullptr};
  explicit ScopedEvent(unsigned flags = cudaEventDisableTiming) { CUDA_CHECK(cudaEventCreateWithFlags(&evt, flags)); }
  ~ScopedEvent() { if (evt) cudaEventDestroy(evt); }
  operator cudaEvent_t() const { return evt; }
};

struct ScopedTimer {
  std::string name;
  bool quiet;
  std::chrono::high_resolution_clock::time_point t0;
  explicit ScopedTimer(std::string n, bool q = false) : name(std::move(n)), quiet(q), t0(std::chrono::high_resolution_clock::now()) {}
  ~ScopedTimer() {
      if (!quiet) {
          auto t1 = std::chrono::high_resolution_clock::now();
          std::cout << name << ": " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
      }
  }
};

  #endif