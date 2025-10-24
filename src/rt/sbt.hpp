#ifndef SBT_HPP
#define SBT_HPP
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

struct SbtRec {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char h[OPTIX_SBT_RECORD_HEADER_SIZE];
  };
  

#endif