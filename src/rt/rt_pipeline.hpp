#ifndef RT_PIPELINE_HPP
#define RT_PIPELINE_HPP

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "../common.hpp"
#include "sbt.hpp"

class rt_pipeline {
public:
    rt_pipeline(bool useTriangles = false) {
        create_context();

        pco_.usesMotionBlur                         = 0;
        pco_.traversableGraphFlags                  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pco_.numPayloadValues                       = 2;
        pco_.numAttributeValues                     = 2;
        pco_.exceptionFlags                         = OPTIX_EXCEPTION_FLAG_NONE;
        pco_.pipelineLaunchParamsVariableName       = "params";

        read_device_ptx(DEVICE_PTX_PATH);

        create_module();

        build_pipeline(useTriangles);
    }

    OptixDeviceContext get_context() { return ctx_; }
    OptixPipeline get_pipeline() { return pipeline_; }
    OptixShaderBindingTable get_sbt() { return sbt_; }
    OptixModule get_module() { return module_; }
private:
    void create_context();
    void create_module();

    void read_device_ptx(const char* path);
    void build_pipeline(bool useTriangles = false);

    OptixDeviceContext ctx_;
    OptixModule module_{};

    OptixPipelineCompileOptions pco_{};
    std::string ptx_;

    OptixPipeline pipeline_{};
    OptixProgramGroup raygenPG_{};
    OptixProgramGroup missPG_{}; 
    OptixProgramGroup hitPG_{};

    OptixShaderBindingTable sbt_{};
};

#endif