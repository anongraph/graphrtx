#ifndef GPU_MANAGER_HPP
#define GPU_MANAGER_HPP

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <list>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>

#include "../shared.h"
#include "../common.hpp"

struct Partition {
  uint32_t id;
  uint32_t seg_first;
  uint32_t seg_count;
  size_t   gas_bytes_est{0};
  bool     resident{false};
  OptixTraversableHandle gas{};
  CUdeviceptr d_gas{};

  // === DUMMY SUPPORT ===
  CUdeviceptr d_bounds{0};   
  size_t      bounds_bytes{0};
};

struct PartitionInfo {
  CUdeviceptr device_aabb_ptr{0};
  size_t      num_bytes{0};
  uint32_t seg_first{0};
  uint32_t seg_count{0};
  size_t   gas_bytes_est{0};
  bool     resident{false};
};


class GPUMemoryManager {
public:
  float bvh_build_ms{0.0f};

  // === Added optional dummy mask ===
  GPUMemoryManager(OptixDeviceContext ctx,
                   const std::vector<UASP>& h_uasps,
                   const std::vector<float>& h_aabbs,
                   uint32_t total_segments,
                   float memFrac = 0.7f,
                   uint32_t num_partitions = 0,
                   const std::vector<uint8_t>* aabb_mask_opt = nullptr)
  : ctx_(ctx), uasps_(h_uasps), aabbs_(h_aabbs), P_(total_segments),
    aabb_mask_opt_(aabb_mask_opt) {

    size_t freeB=0, totB=0;
    CUDA_CHECK(cudaMemGetInfo(&freeB,&totB));
    const size_t floorBytes = 64ull*1024*1024;
    limitBytes_ = (size_t)(memFrac * freeB);
    if (limitBytes_ < floorBytes) limitBytes_ = floorBytes;

    uint32_t tryParts;
    if (num_partitions > 0) {
      tryParts = num_partitions;
    } else {
      uint32_t baseParts = std::max<uint32_t>(1, (uint32_t)std::sqrt((double)P_));
      tryParts  = baseParts;

      createPartitions(tryParts);
      estimatePartitionBytes();

      while (maxPartBytes_ * 3 > limitBytes_ && tryParts < std::max<uint32_t>(baseParts*8, 2048u)) {
        tryParts *= 2;
        createPartitions(tryParts);
        estimatePartitionBytes();
      }
      std::cout << "[MM] Using auto-calculated partition count: " << tryParts << "\n";
    }

    createPartitions(tryParts);
    estimatePartitionBytes();

    // === DUMMY SUPPORT: detect tail of dummy AABBs ===
    if (aabb_mask_opt_ && !aabb_mask_opt_->empty()) {
      const auto& mask = *aabb_mask_opt_;
      const uint32_t total_prims = (uint32_t)(aabbs_.size() / 6);
      uint32_t tail = total_prims;
      while (tail > 0 && mask[tail - 1] == 1u) --tail;
      if (tail < total_prims) {
        dummy_first_ = tail;
        dummy_count_ = total_prims - tail;
      }
    }

    if (dummy_count_ > 0) {
      Partition p{};
      p.id = (uint32_t)partitions_.size();
      p.seg_first = dummy_first_;
      p.seg_count = dummy_count_;
      p.gas_bytes_est = std::max<size_t>(p.seg_count * sizeof(float) * 6 * 2, 1ull<<20);
      p.resident = false;
      partitions_.push_back(p);
      dummy_pid_ = p.id;
    }

    std::cout << "[MM] Free=" << (freeB/1e9) << " GB, Limit=" << (limitBytes_/1e9)
              << " GB, Segments=" << P_ << ", Partitions=" << partitions_.size()
              << ", MaxPartBytes≈" << (maxPartBytes_/1e6) << " MB\n";

    size_t totalEst = maxPartBytes_ * partitions_.size();
    if (totalEst < limitBytes_) {
      useSingleTLAS_ = true;
      CUstream stream; CUDA_CHECK(cudaStreamCreate(&stream));
      using clk = std::chrono::high_resolution_clock;
      const auto wall_t0 = clk::now();

      for (auto& p : partitions_) loadPartitionIfNeeded(p.id, stream);
      
      buildTLASDynamic(&global_tlas_, &global_tlas_mem_, &global_instance_bases_,
                       &global_num_instances_, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const auto wall_t1 = clk::now();
      const double elapsed_ms_bvh = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();  
      std::cout << "BVH:" << elapsed_ms_bvh << " ms" << std::endl;
      std::cout << "BVH size: " << getTotalBVHBytes() / (1024.0 * 1024.0) << " MB" << std::endl;
      CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
      useSingleTLAS_ = false;
      std::cout << "[MM] Using streaming mode (multi-TLAS).\n";

      // Force-load dummy partition so it always stays resident
      if (dummy_pid_ != UINT32_MAX) {
        CUstream s; CUDA_CHECK(cudaStreamCreate(&s));
        loadPartitionIfNeeded(dummy_pid_, s);
        touch(dummy_pid_);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaStreamDestroy(s));
      }
    }

    // === Initialize dummy pool ===
    if (dummy_count_ > 0) {
      initDummyPool();
      std::cout << "[MM] Dummy pool: first=" << dummy_first_
                << " count=" << dummy_count_ << "\n";
    }
  }

  size_t getTotalBVHBytes() const;

  bool hasSingleTLAS() const { return useSingleTLAS_; }

  void getSingleTLAS(OptixTraversableHandle* tlas,
                     CUdeviceptr* mem,
                     CUdeviceptr* bases,
                     uint32_t* num_instances) const;

  void ensureResident(const std::vector<uint32_t>& need_ids, CUstream stream);
  

  void buildTLASDynamic(OptixTraversableHandle* outTLAS,
                        CUdeviceptr* outTLASMem,
                        CUdeviceptr* d_instance_bases,
                        uint32_t* out_num_instances,
                        CUstream stream);

  uint32_t vertexToPartition(uint32_t, uint32_t uasp_first_v) const;

  PartitionInfo get_partition_info(uint32_t pid) const;

  // === DUMMY SUPPORT API ===
  void initDummyPool() {
    if (dummy_count_ == 0) return;
    dummy_free_.clear();
    dummy_used_.clear();
    dummy_free_.reserve(dummy_count_);
    for (uint32_t i = 0; i < dummy_count_; ++i)
      dummy_free_.push_back(dummy_first_ + i);
  }

  uint32_t allocateDummy() {
    if (dummy_free_.empty()) return UINT32_MAX;
    uint32_t id = dummy_free_.back();
    dummy_free_.pop_back();
    dummy_used_.push_back(id);
    return id;
  }

  // Preferred: try to allocate; if none left -> trigger full rebuild.
  uint32_t allocateDummyOrRebuild(CUstream stream);

  void freeDummy(uint32_t primIndex) {
    auto it = std::find(dummy_used_.begin(), dummy_used_.end(), primIndex);
    if (it != dummy_used_.end()) {
      dummy_free_.push_back(*it);
      dummy_used_.erase(it);
    }
  }

  void moveDummyAndRefit(uint32_t primIndex, const float new6[6], CUstream stream);


  // Expose how many full rebuilds were triggered by pool exhaustion
  uint32_t numFullRebuildsTriggered() const { return num_full_rebuilds_; }


  public:
  double benchmarkDummyUpdates(uint32_t num_updates, CUstream stream);

private:
  OptixDeviceContext ctx_;
  const std::vector<UASP>& uasps_;
  const std::vector<float>& aabbs_;
  const uint32_t P_;
  std::vector<Partition> partitions_;
  std::list<uint32_t> lru_;
  std::unordered_map<uint32_t, std::list<uint32_t>::iterator> where_;
  size_t usedBytes_{0}, limitBytes_{0}, maxPartBytes_{0};
  std::vector<uint32_t> instanceBases_;
  bool useSingleTLAS_{false};
  OptixTraversableHandle global_tlas_{0};
  CUdeviceptr global_tlas_mem_{0};
  CUdeviceptr global_instance_bases_{0};
  uint32_t global_num_instances_{0};
  size_t global_tlas_bytes_{0};

  // === DUMMY SUPPORT ===
  const std::vector<uint8_t>* aabb_mask_opt_{nullptr};
  uint32_t dummy_first_{0};
  uint32_t dummy_count_{0};
  uint32_t dummy_pid_{UINT32_MAX};
  std::vector<uint32_t> dummy_free_;
  std::vector<uint32_t> dummy_used_;
  uint32_t num_full_rebuilds_{0};

  void createPartitions(uint32_t parts);
  void estimatePartitionBytes();
  void loadPartitionIfNeeded(uint32_t pid, CUstream stream);
  void evictOne(CUstream);
  void touch(uint32_t id);

  // === COMPLETE REBUILD PATH ===
  void rebuildPartitionGAS(uint32_t pid, CUstream stream);


  // Rebuild *all resident* partition GAS and the TLAS.
  void fullRebuild(CUstream stream);
};

#endif