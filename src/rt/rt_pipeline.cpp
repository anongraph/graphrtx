#include "rt_pipeline.hpp"
#include <fstream>
#include <cstring>
#include <vector>

void rt_pipeline::read_device_ptx(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("Cannot open PTX: ") + path);

    const auto size = f.tellg();
    f.seekg(0, std::ios::beg);

    ptx_.resize(size);
    f.read(ptx_.data(), size);
}

void rt_pipeline::create_context() {
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(0));

    CUdevice cuDev = 0;
    CUcontext cuCtx = nullptr;
    CUresult res;

    res = cuInit(0);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("cuInit failed");

    res = cuDeviceGet(&cuDev, 0);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("cuDeviceGet failed");

    res = cuDevicePrimaryCtxRetain(&cuCtx, cuDev);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("cuDevicePrimaryCtxRetain failed");

    res = cuCtxSetCurrent(cuCtx);
    if (res != CUDA_SUCCESS)
        throw std::runtime_error("cuCtxSetCurrent failed");

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions o{};
    o.logCallbackFunction = optixLogCb;
    o.logCallbackLevel = 2;

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &o, &ctx_));
}

void rt_pipeline::create_module() {
    OptixModuleCompileOptions mc{};
    mc.maxRegisterCount = 8;//OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mc.optLevel  = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    mc.debugLevel= OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
    char log[8192]; size_t logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(ctx_,&mc,&pco_,ptx_.c_str(),strlen(ptx_.c_str()),log,&logSize,&module_));
    if(logSize>1) std::cerr<<log<<"\n";
}

void rt_pipeline::build_pipeline(bool useTriangles) {
    OptixProgramGroupOptions po{};
    char log[8192]; size_t logSize;
  
    // Raygen
    OptixProgramGroupDesc rgd{};
    rgd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgd.raygen.module = module_;
    rgd.raygen.entryFunctionName = "__raygen__graph";
    logSize=sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx_,&rgd,1,&po,log,&logSize,&raygenPG_));
  
    // Miss
    OptixProgramGroupDesc msd{};
    msd.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msd.miss.module = module_;
    msd.miss.entryFunctionName = "__miss__noop";
    logSize=sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx_,&msd,1,&po,log,&logSize,&missPG_));
  
    // Hitgroup
    OptixProgramGroupDesc hgd{};
    hgd.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  
    if (useTriangles) {
      hgd.hitgroup.moduleIS = nullptr;
      hgd.hitgroup.entryFunctionNameIS = nullptr;
      hgd.hitgroup.moduleCH = module_;
      hgd.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
      hgd.hitgroup.moduleAH = nullptr;
      hgd.hitgroup.entryFunctionNameAH = nullptr;
    } else {
      hgd.hitgroup.moduleIS = module_;
      hgd.hitgroup.entryFunctionNameIS = "__intersection__uasp";
      hgd.hitgroup.moduleCH = nullptr;
      hgd.hitgroup.entryFunctionNameCH = nullptr;
      hgd.hitgroup.moduleAH = module_;
      hgd.hitgroup.entryFunctionNameAH = "__anyhit__uasp";
    }
  
    logSize=sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx_,&hgd,1,&po,log,&logSize,&hitPG_));
  
    std::vector<OptixProgramGroup> gs{raygenPG_,missPG_,hitPG_};
    OptixPipelineLinkOptions l{}; l.maxTraceDepth = 1;
    logSize=sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(ctx_,&pco_,&l,gs.data(),(unsigned)gs.size(),log,&logSize,&pipeline_));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, 2*1024, 2*1024, 2*1024, 1));
  
    // SBT
    SbtRec rgR{},msR{},hgR{};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG_,&rgR));
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG_,&msR));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG_,&hgR));
  
    CUdeviceptr d_rg,d_ms,d_hg;
    CUDA_CHECK(cudaMalloc((void**)&d_rg,sizeof(rgR)));
    CUDA_CHECK(cudaMalloc((void**)&d_ms,sizeof(msR)));
    CUDA_CHECK(cudaMalloc((void**)&d_hg,sizeof(hgR)));
  
    CUDA_CHECK(cudaMemcpy((void*)d_rg,&rgR,sizeof(rgR),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)d_ms,&msR,sizeof(msR),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)d_hg,&hgR,sizeof(hgR),cudaMemcpyHostToDevice));
  
    sbt_.raygenRecord=d_rg;
    sbt_.missRecordBase=d_ms;
    sbt_.missRecordStrideInBytes=sizeof(SbtRec);
    sbt_.missRecordCount=1;
    sbt_.hitgroupRecordBase=d_hg;
    sbt_.hitgroupRecordStrideInBytes=sizeof(SbtRec);
    sbt_.hitgroupRecordCount=1;
    sbt_.exceptionRecord = 0;
}