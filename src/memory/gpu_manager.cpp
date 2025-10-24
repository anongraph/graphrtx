#include "gpu_manager.hpp"

void GPUMemoryManager::createPartitions(uint32_t parts) {
    partitions_.clear();
    if (parts == 0) parts = 1;
    uint32_t base = 0;
    for (uint32_t i=0;i<parts;i++){
      const uint32_t remain = P_ - base;
      const uint32_t take = (i+1==parts) ? remain : (remain / (parts - i));
      Partition p{};
      p.id = i;
      p.seg_first = base;
      p.seg_count = take;
      partitions_.push_back(p);
      base += take;
    }
  }
  
  void GPUMemoryManager::estimatePartitionBytes() {
    maxPartBytes_ = 0;
    for (auto& p : partitions_) {
      OptixBuildInput in{};
      in.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
      CUdeviceptr dummy=0;
      in.customPrimitiveArray.aabbBuffers       = &dummy;
      in.customPrimitiveArray.numPrimitives     = p.seg_count;
      in.customPrimitiveArray.strideInBytes     = sizeof(OptixAabb);
      static unsigned f = OPTIX_GEOMETRY_FLAG_NONE;
      in.customPrimitiveArray.flags             = &f;
      in.customPrimitiveArray.numSbtRecords     = 1;
      in.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
      in.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
      in.customPrimitiveArray.sbtIndexOffsetStrideInBytes=0;
      in.customPrimitiveArray.primitiveIndexOffset = 0;
  
      OptixAccelBuildOptions o{};
      o.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
      o.operation  = OPTIX_BUILD_OPERATION_BUILD;
      OptixAccelBufferSizes s{};
      OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx_, &o, &in, 1, &s));
      p.gas_bytes_est = (size_t)std::ceil(s.outputSizeInBytes * 1.1);
      if (p.gas_bytes_est > maxPartBytes_) maxPartBytes_ = p.gas_bytes_est;
    }
  }
  
  void GPUMemoryManager::loadPartitionIfNeeded(uint32_t pid, CUstream stream) {
    auto& p = partitions_.at(pid);
    if (p.resident) return;
  
    // --- 1. Compute Required Memory ---
    // AABB buffers will be freed quickly, but we need the GAS and TEMP buffer size.
    // We compute sizes early to get the actual need (s.outputSizeInBytes and s.tempSizeInBytes).
    
    OptixAccelBuildOptions o{};
    o.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    o.operation  = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixBuildInput in{};
    in.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    in.customPrimitiveArray.numPrimitives = p.seg_count;
    in.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    
    // Allocate AABB data first (small and essential)
    const uint64_t start6 = uint64_t(p.seg_first) * 6ull;
    const uint64_t count6 = uint64_t(p.seg_count) * 6ull;
    CUdeviceptr d_aabb{};
    CUDA_CHECK(cudaMalloc((void**)&d_aabb, count6*sizeof(float))); // Allocate AABB buffer
    CUDA_CHECK(cudaMemcpyAsync((void*)d_aabb,
                               aabbs_.data() + start6,
                               count6*sizeof(float),
                               cudaMemcpyHostToDevice, stream));
  
    // Finalize OptiX Build Input
    in.customPrimitiveArray.aabbBuffers       = &d_aabb;
    static unsigned f = OPTIX_GEOMETRY_FLAG_NONE;
    in.customPrimitiveArray.flags             = &f;
    in.customPrimitiveArray.numSbtRecords     = 1;
    in.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    in.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    in.customPrimitiveArray.sbtIndexOffsetStrideInBytes=0;
    in.customPrimitiveArray.primitiveIndexOffset = 0;
  
    // Get actual memory usage
    OptixAccelBufferSizes s{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx_, &o, &in, 1, &s));
  
    const size_t needed_actual_output = s.outputSizeInBytes;
    const size_t needed_actual_temp = s.tempSizeInBytes;
  
    const size_t need_estimate_for_eviction = p.gas_bytes_est;
    
    const size_t FIXED_NON_RESIDENT_OVERHEAD = 64ull << 20; // 64MB buffer
    
    const size_t target_used = limitBytes_ - need_estimate_for_eviction;
  
    while (usedBytes_ > target_used && !lru_.empty()) {
        evictOne(stream);
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream)); 
    
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    
    const size_t needed_for_build_peak = needed_actual_output + needed_actual_temp + FIXED_NON_RESIDENT_OVERHEAD;
    
    if (needed_for_build_peak > freeB) {
       std::cerr << "[MM] CRITICAL ERROR: Insufficient free memory (" << (freeB/1048576.0) << " MB) to allocate GAS (" << (needed_actual_output/1048576.0) << " MB) + TEMP (" << (needed_actual_temp/1048576.0) << " MB) + OVERHEAD, even after eviction!\n";
    }
  
  
    CUdeviceptr dgas{};
    CUdeviceptr d_temp{};
    
    CUDA_CHECK(cudaMalloc((void**)&d_temp, needed_actual_temp)); 
    
    CUDA_CHECK(cudaMalloc((void**)&dgas,  needed_actual_output)); 
  
    OptixTraversableHandle gas{};
    OPTIX_CHECK(optixAccelBuild(ctx_, stream, &o, &in, 1,
                                d_temp, needed_actual_temp,
                                dgas, needed_actual_output,
                                &gas, nullptr, 0));
    
    CUDA_CHECK(cudaStreamSynchronize(stream)); 
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree((void*)d_temp));
    CUDA_CHECK(cudaFree((void*)d_aabb)); 
  
    p.gas = gas;
    p.d_gas = dgas;
    p.resident = true;
    const size_t actual = (size_t)std::ceil(s.outputSizeInBytes * 1.05);
    p.gas_bytes_est = std::max(p.gas_bytes_est, actual);
    
    usedBytes_ += p.gas_bytes_est;
  
    touch(pid);
  }
  
  void GPUMemoryManager::evictOne(CUstream) {
    while (!lru_.empty()) {
      uint32_t id = lru_.back();
      lru_.pop_back();
      where_.erase(id);
      auto& p = partitions_[id];
      if (!p.resident) continue;
      if (p.d_gas) CUDA_CHECK(cudaFree((void*)p.d_gas));
      p.d_gas = 0;
      p.gas = 0;
      p.resident = false;
      if (usedBytes_ >= p.gas_bytes_est) usedBytes_ -= p.gas_bytes_est;
      else usedBytes_ = 0;
      //std::cout << "[MM] Evict P" << id << " | used="<<(usedBytes_/1e6)<<"MB\n";
      return;
    }
  }
  
  void GPUMemoryManager::touch(uint32_t id) {
    auto it = where_.find(id);
    if (it != where_.end()) {
      lru_.erase(it->second);
    }
    lru_.push_front(id);
    where_[id] = lru_.begin();
  }

  size_t GPUMemoryManager::getTotalBVHBytes() const {
    size_t total = global_tlas_bytes_;
    for (const auto& p : partitions_)
        total += p.gas_bytes_est;
    return total;
  }

  void GPUMemoryManager::getSingleTLAS(OptixTraversableHandle* tlas,
                     CUdeviceptr* mem,
                     CUdeviceptr* bases,
                     uint32_t* num_instances) const {
    *tlas = global_tlas_;
    *mem = global_tlas_mem_;
    *bases = global_instance_bases_;
    *num_instances = global_num_instances_;
  }

  void GPUMemoryManager::ensureResident(const std::vector<uint32_t>& need_ids, CUstream stream) {
    if (useSingleTLAS_) return;
    for (uint32_t pid : need_ids) {
      loadPartitionIfNeeded(pid, stream);
      touch(pid);
    }
    for (uint32_t pid : need_ids) {
      uint32_t nid = pid + 1;
      if (nid < partitions_.size()) {
        if (!partitions_[nid].resident
            && usedBytes_ + partitions_[nid].gas_bytes_est < limitBytes_) {
          loadPartitionIfNeeded(nid, stream);
        }
        touch(nid);
      }
    }
    while (usedBytes_ > limitBytes_ && !lru_.empty()) {
      evictOne(stream);
    }
  }

  void GPUMemoryManager::buildTLASDynamic(OptixTraversableHandle* outTLAS,
                        CUdeviceptr* outTLASMem,
                        CUdeviceptr* d_instance_bases,
                        uint32_t* out_num_instances,
                        CUstream stream)
  {
    std::vector<OptixInstance> instances;
    instanceBases_.clear();

    for (const auto& p : partitions_) {
      if (!p.resident) continue;
      OptixInstance inst{};
      const float T[12] = {1,0,0,0,  0,1,0,0,  0,0,1,0};
      std::memcpy(inst.transform, T, sizeof(T));
      inst.instanceId        = (unsigned)instances.size();
      inst.sbtOffset         = 0;
      inst.visibilityMask    = 255;
      inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
      inst.traversableHandle = p.gas;
      instances.push_back(inst);
      instanceBases_.push_back(p.seg_first);
    }

    *out_num_instances = (uint32_t)instances.size();
    if (instances.empty()) {
      *outTLAS = 0; *outTLASMem = 0; *d_instance_bases = 0;
      return;
    }

    CUdeviceptr d_instances{};
    CUDA_CHECK(cudaMalloc((void**)&d_instances, instances.size() * sizeof(OptixInstance)));
    CUDA_CHECK(cudaMemcpyAsync((void*)d_instances, instances.data(),
                               instances.size()*sizeof(OptixInstance),
                               cudaMemcpyHostToDevice, stream));

    OptixBuildInput in{};
    in.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    in.instanceArray.instances    = d_instances;
    in.instanceArray.numInstances = (unsigned)instances.size();

    OptixAccelBuildOptions o{};
    o.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    o.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes s{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx_, &o, &in, 1, &s));

    size_t freeB = 0, totB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totB));
    const size_t LAUNCH_OVERHEAD_SAFETY = 128ull << 20; 
    const size_t needed = s.tempSizeInBytes + s.outputSizeInBytes + LAUNCH_OVERHEAD_SAFETY;
    
    bool use_compaction = true;
    if (needed > freeB) {
      o.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx_, &o, &in, 1, &s));
      use_compaction = false;
    }

    CUdeviceptr d_temp{};
    CUDA_CHECK(cudaMalloc((void**)&d_temp, s.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)outTLASMem,  s.outputSizeInBytes));

    CUdeviceptr d_compacted_size{};
    OptixTraversableHandle tlas{};
    OptixAccelEmitDesc emit{};
    OptixAccelEmitDesc* emit_ptr = nullptr;
    if (use_compaction) {
      CUDA_CHECK(cudaMalloc((void**)&d_compacted_size, sizeof(size_t)));
      emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emit.result = d_compacted_size;
      emit_ptr = &emit;
    }

    OPTIX_CHECK(optixAccelBuild(ctx_, stream, &o, &in, 1,
      d_temp, s.tempSizeInBytes, *outTLASMem, s.outputSizeInBytes,
      &tlas, emit_ptr, emit_ptr ? 1 : 0));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree((void*)d_temp));
    CUDA_CHECK(cudaFree((void*)d_instances));

    size_t compactedSize = 0;
    if (use_compaction) {
      CUDA_CHECK(cudaMemcpy(&compactedSize, (void*)d_compacted_size, sizeof(size_t),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree((void*)d_compacted_size));

      if (compactedSize > 0 && compactedSize < s.outputSizeInBytes) {
        CUdeviceptr d_comp{};
        CUDA_CHECK(cudaMalloc((void**)&d_comp, compactedSize));
        OPTIX_CHECK(optixAccelCompact(ctx_, stream, tlas, d_comp, compactedSize, &tlas));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree((void*)*outTLASMem));
        *outTLASMem = d_comp;
      }
    }

    *outTLAS = tlas;
    global_tlas_bytes_ = (use_compaction && compactedSize > 0)
                         ? compactedSize
                         : s.outputSizeInBytes;

    if (!instanceBases_.empty()) {
      CUDA_CHECK(cudaMalloc((void**)d_instance_bases, instanceBases_.size()*sizeof(uint32_t)));
      CUDA_CHECK(cudaMemcpyAsync((void*)*d_instance_bases, instanceBases_.data(),
                                 instanceBases_.size()*sizeof(uint32_t),
                                 cudaMemcpyHostToDevice, stream));
    } else {
      *d_instance_bases = 0;
    }
  }

  uint32_t GPUMemoryManager::vertexToPartition(uint32_t, uint32_t uasp_first_v) const {
    const uint32_t segIdx = uasp_first_v;
    uint32_t lo=0, hi=(uint32_t)partitions_.size();
    while (lo<hi) {
      uint32_t mid=(lo+hi)>>1;
      const auto& p=partitions_[mid];
      if (segIdx < p.seg_first) hi=mid;
      else if (segIdx >= p.seg_first + p.seg_count) lo=mid+1;
      else return mid;
    }
    return 0u;
  }

  PartitionInfo GPUMemoryManager::get_partition_info(uint32_t pid) const {
    PartitionInfo out{};
    if (pid >= partitions_.size()) return out;
    const auto& p = partitions_[pid];
    out.seg_first     = p.seg_first;
    out.seg_count     = p.seg_count;
    out.gas_bytes_est = p.gas_bytes_est;
    out.resident      = p.resident;
    out.device_aabb_ptr = p.d_bounds;
    out.num_bytes = p.bounds_bytes;
    return out;
  }

  double GPUMemoryManager::benchmarkDummyUpdates(uint32_t num_updates, CUstream stream) {
    if (dummy_count_ == 0) {
      std::cout << "[MM][Bench] No dummy AABBs defined — skipping.\n";
      return 0.0;
    }

    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();

    std::vector<uint32_t> allocated;
    allocated.reserve(num_updates);

    for (uint32_t i = 0; i < num_updates; ++i) {
      uint32_t id = allocateDummyOrRebuild(stream);
      if (id == UINT32_MAX) {
        std::cout << "[MM][Bench] No dummy left; rebuild triggered at update " << i << "\n";
        continue;
      }
      allocated.push_back(id);

      float x = -1e6f + 10.0f * (float)i;  // move gradually further out
      float y = 0.0f;
      float z = 0.0f;
      float new6[6] = {x, y, z, x + 1.0f, y + 1.0f, z + 1.0f};

      moveDummyAndRefit(id, new6, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto t1 = clk::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[MM][Bench] Performed " << num_updates << " dummy updates in "
              << ms << " ms (" << (ms / std::max(1u, num_updates)) << " ms/update)"
              << ", rebuilds=" << num_full_rebuilds_ << std::endl;

    for (uint32_t id : allocated)
      freeDummy(id);

    return ms;
  }

  void GPUMemoryManager::rebuildPartitionGAS(uint32_t pid, CUstream stream) {
        if (pid >= partitions_.size()) return;
        auto& p = partitions_[pid];
        if (!p.resident) return;          
        if (p.seg_count == 0) return;   

        // Ensure d_bounds exists and is populated from host aabbs_
        if (p.d_bounds == 0) {
            const size_t bytes = size_t(p.seg_count) * 6 * sizeof(float);
            CUDA_CHECK(cudaMalloc((void**)&p.d_bounds, bytes));
            const float* src = aabbs_.data() + size_t(p.seg_first) * 6;
            CUDA_CHECK(cudaMemcpyAsync((void*)p.d_bounds, src, bytes,
                                    cudaMemcpyHostToDevice, stream));
            p.bounds_bytes = bytes;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

        OptixBuildInputCustomPrimitiveArray& aabbArray = buildInput.customPrimitiveArray;

        const CUdeviceptr aabbBuf = p.d_bounds;
        aabbArray.aabbBuffers    = &aabbBuf;
        aabbArray.numPrimitives  = p.seg_count;
        aabbArray.strideInBytes  = sizeof(float) * 6;

        static const unsigned int geomFlags = OPTIX_GEOMETRY_FLAG_NONE;
        aabbArray.flags                       = const_cast<unsigned int*>(&geomFlags);
        aabbArray.numSbtRecords               = 1;
        aabbArray.sbtIndexOffsetBuffer        = 0;
        aabbArray.sbtIndexOffsetSizeInBytes   = 0;
        aabbArray.sbtIndexOffsetStrideInBytes = 0;
        aabbArray.primitiveIndexOffset        = p.seg_first;

        OptixAccelBuildOptions opts{};
        opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD
                        | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                        | OPTIX_BUILD_FLAG_ALLOW_UPDATE;   
        opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes sizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx_, &opts, &buildInput, 1, &sizes));

        size_t freeB=0, totB=0;
        CUDA_CHECK(cudaMemGetInfo(&freeB,&totB));
        const size_t LAUNCH_OVERHEAD_SAFETY = 128ull << 20;
        const size_t needed = sizes.tempSizeInBytes + sizes.outputSizeInBytes + LAUNCH_OVERHEAD_SAFETY;

        bool use_compaction = true;
        if (needed > freeB) {
        opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx_, &opts, &buildInput, 1, &sizes));
        use_compaction = false;
        }

        if (p.d_gas) {
        usedBytes_ -= p.gas_bytes_est;
        CUDA_CHECK(cudaFree((void*)p.d_gas));
        p.d_gas = 0;
        p.gas_bytes_est = 0;
        }

        CUdeviceptr d_temp{};
        CUDA_CHECK(cudaMalloc((void**)&d_temp, sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc((void**)&p.d_gas, sizes.outputSizeInBytes));

        CUdeviceptr d_compacted_size{};
        OptixAccelEmitDesc emit{};
        OptixAccelEmitDesc* emit_ptr = nullptr;
        if (use_compaction) {
        CUDA_CHECK(cudaMalloc((void**)&d_compacted_size, sizeof(size_t)));
        emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = d_compacted_size;
        emit_ptr = &emit;
        }

        OptixTraversableHandle gas{};
        OPTIX_CHECK(optixAccelBuild(ctx_, stream, &opts,
                                    &buildInput, 1,
                                    d_temp, sizes.tempSizeInBytes,
                                    p.d_gas, sizes.outputSizeInBytes,
                                    &gas, emit_ptr, emit_ptr ? 1 : 0));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree((void*)d_temp));

        size_t compactedSize = 0;
        if (use_compaction) {
        CUDA_CHECK(cudaMemcpy(&compactedSize, (void*)d_compacted_size, sizeof(size_t),
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree((void*)d_compacted_size));
        if (compactedSize > 0 && compactedSize < sizes.outputSizeInBytes) {
            CUdeviceptr d_comp{};
            CUDA_CHECK(cudaMalloc((void**)&d_comp, compactedSize));
            OPTIX_CHECK(optixAccelCompact(ctx_, stream, gas, d_comp, compactedSize, &gas));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree((void*)p.d_gas));
            p.d_gas = d_comp;
            sizes.outputSizeInBytes = compactedSize;
        }
        }

        p.gas = gas;
        p.gas_bytes_est = sizes.outputSizeInBytes;
        usedBytes_ += p.gas_bytes_est; 
    }

    void GPUMemoryManager::fullRebuild(CUstream stream) {
        for (auto& p : partitions_) {
        if (!p.resident) continue;
            rebuildPartitionGAS(p.id, stream);
        }
        buildTLASDynamic(&global_tlas_, &global_tlas_mem_,
                     &global_instance_bases_, &global_num_instances_, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << "[MM] COMPLETE REBUILD finished.\n";
    }

    void GPUMemoryManager::moveDummyAndRefit(uint32_t primIndex, const float new6[6], CUstream stream) {
        if (dummy_pid_ == UINT32_MAX) return;
        auto& p = partitions_[dummy_pid_];
        if (!p.resident) loadPartitionIfNeeded(dummy_pid_, stream);
        if (p.d_bounds == 0) {
            std::cerr << "[MM][ERR] moveDummyAndRefit: dummy partition has no d_bounds allocated!\n";
            return;
        }
        if (primIndex < dummy_first_ || primIndex >= dummy_first_ + dummy_count_) return;
    
        const size_t localIdx = primIndex - dummy_first_;
        const size_t byteOffset = localIdx * 6 * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync((uint8_t*)p.d_bounds + byteOffset, new6,
                                   6*sizeof(float), cudaMemcpyHostToDevice, stream));
    
        buildTLASDynamic(&global_tlas_, &global_tlas_mem_,
                         &global_instance_bases_, &global_num_instances_, stream);
    }
    
    uint32_t GPUMemoryManager::allocateDummyOrRebuild(CUstream stream) {
        uint32_t id = allocateDummy();
        if (id != UINT32_MAX) return id;
    
        using clk = std::chrono::high_resolution_clock;
        const auto t0 = clk::now();
    
        std::cout << "[MM] Dummy pool exhausted — triggering COMPLETE REBUILD...\n";
    
        fullRebuild(stream);
    
        const auto t1 = clk::now();
        const double rebuild_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();;
    
        std::cout << "[MM] COMPLETE REBUILD finished in "
                  << rebuild_ms << " ms\n";
    
        return UINT32_MAX;
    }
