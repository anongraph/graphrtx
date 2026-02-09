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
#include <cstdint>
#include <iostream>

#include "../shared.h"
#include "../common.hpp"


enum class ShardMode : uint32_t { Contiguous = 0, RoundRobin = 1 };

struct ShardConfig {
  bool      enabled   = false;
  uint32_t  num_gpus  = 1;  
  uint32_t  this_gpu  = 0;   
  ShardMode mode      = ShardMode::Contiguous;

  bool owns(uint32_t part_id, uint32_t Pparts) const {
    if (!enabled || num_gpus <= 1) return true;
    if (mode == ShardMode::RoundRobin) {
      return (part_id % num_gpus) == this_gpu;
    }

    uint32_t chunk = (Pparts + num_gpus - 1u) / num_gpus;
    uint32_t begin = this_gpu * chunk;
    uint32_t end   = std::min(Pparts, begin + chunk);
    return (part_id >= begin && part_id < end);
  }
};

struct Partition {
  uint32_t id;
  uint32_t seg_first;
  uint32_t seg_count;
  size_t   gas_bytes_est{0};

  bool     resident{false};
  OptixTraversableHandle gas{};
  CUdeviceptr d_gas{};

  CUdeviceptr d_bounds{0};
  size_t      bounds_bytes{0};

  bool owned{true}; 
};

struct PartitionInfo {
  CUdeviceptr device_aabb_ptr{0};
  size_t      num_bytes{0};
  uint32_t seg_first{0};
  uint32_t seg_count{0};
  size_t   gas_bytes_est{0};
  bool     resident{false};
  bool     owned{true};
};

class GPUMemoryManager {
public:
  float bvh_build_ms{0.0f};

  GPUMemoryManager(OptixDeviceContext ctx,
                   const std::vector<UASP>& h_uasps,
                   const std::vector<float>& h_aabbs,
                   uint32_t total_segments,
                   float memFrac = 0.7f,
                   uint32_t num_partitions = 0,
                   const std::vector<uint8_t>* aabb_mask_opt = nullptr)
  : GPUMemoryManager(ctx, h_uasps, h_aabbs, total_segments, memFrac,
                     num_partitions, aabb_mask_opt,
                     false, 1, 0,
                     ShardMode::Contiguous) {}

  GPUMemoryManager(OptixDeviceContext ctx,
                   const std::vector<UASP>& h_uasps,
                   const std::vector<float>& h_aabbs,
                   uint32_t total_segments,
                   float memFrac,
                   uint32_t num_partitions,
                   const std::vector<uint8_t>* aabb_mask_opt,
                   bool shard_enabled,
                   uint32_t num_gpus,
                   uint32_t this_gpu,
                   ShardMode shard_mode);

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

  uint32_t vertexToPartition(uint32_t v, uint32_t uasp_first_v) const;

  PartitionInfo get_partition_info(uint32_t pid) const;

  bool shardingEnabled() const { return shard_.enabled; }
  uint32_t numGpus() const { return shard_.num_gpus; }
  uint32_t thisGpu() const { return shard_.this_gpu; }
  bool ownsPartition(uint32_t pid) const { return shard_.owns(pid, (uint32_t)partitions_.size()); }

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

  uint32_t allocateDummyOrRebuild(CUstream stream);

  void freeDummy(uint32_t primIndex) {
    auto it = std::find(dummy_used_.begin(), dummy_used_.end(), primIndex);
    if (it != dummy_used_.end()) {
      dummy_free_.push_back(*it);
      dummy_used_.erase(it);
    }
  }

  void moveDummyAndRefit(uint32_t primIndex, const float new6[6], CUstream stream);

  uint32_t numFullRebuildsTriggered() const { return num_full_rebuilds_; }

public:
  double benchmarkDummyUpdates(uint32_t num_updates, CUstream stream);

private:
  OptixDeviceContext ctx_;
  const std::vector<UASP>& uasps_;
  const std::vector<float>& aabbs_;
  const uint32_t P_;

  ShardConfig shard_;

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

  const std::vector<uint8_t>* aabb_mask_opt_{nullptr};
  uint32_t dummy_first_{0};
  uint32_t dummy_count_{0};
  uint32_t dummy_pid_{UINT32_MAX};
  std::vector<uint32_t> dummy_free_;
  std::vector<uint32_t> dummy_used_;
  uint32_t num_full_rebuilds_{0};

  void createPartitions(uint32_t parts);
  void estimatePartitionBytes();

  void markOwnership();

  void loadPartitionIfNeeded(uint32_t pid, CUstream stream);
  void evictOne(CUstream);
  void touch(uint32_t id);

  void rebuildPartitionGAS(uint32_t pid, CUstream stream);
  void fullRebuild(CUstream stream);
};

#endif
