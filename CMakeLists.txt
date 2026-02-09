cmake_minimum_required(VERSION 3.13)
project(HelloOptix LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# OptiX SDK root
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
include_directories(${OPTIX_ROOT}/include)

# Locate OptiX driver lib
find_library(OPTIX_DRIVER_LIB
  NAMES nvoptix
  PATHS ${OPTIX_ROOT}/lib64 ${OPTIX_ROOT}/lib /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/nvidia
)

# GPU arch for PTX
if(NOT DEFINED SM_ARCH)
  set(SM_ARCH 75)
endif()

# Generate PTX
set(PTX_OUT ${CMAKE_BINARY_DIR}/device.ptx)
add_custom_command(
  OUTPUT ${PTX_OUT}
  COMMAND ${CMAKE_CUDA_COMPILER} --ptx -arch=sm_${SM_ARCH} -O3 -lineinfo -use_fast_math -std=c++17 -Xptxas -maxrregcount=64
          -I ${OPTIX_ROOT}/include
          -I ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/optix/device.cu
          -o ${PTX_OUT}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/optix/device.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/shared.h
  COMMENT "NVCC: generating PTX -> ${PTX_OUT}"
  VERBATIM
)
add_custom_target(device_ptx DEPENDS ${PTX_OUT})

# Host executable
add_executable(bench_all
    src/memory/gpu_manager.cpp
    bench/bench_all.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/algorithms/wcc.cpp
    src/algorithms/cdlp.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
    src/kernels/cuda/wcc.cu
    src/kernels/cuda/cdlp.cu
    src/shared.h
)
add_dependencies(bench_all device_ptx)

# Pass PTX path
target_compile_definitions(bench_all PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

# Includes & libs
target_include_directories(bench_all PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

target_link_libraries(bench_all PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(bench_all PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# RPATH for runtime libnvoptix
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(bench_all PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(bench_all PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()


###################################################### BFS TEST

add_executable(bfs_test
    test/bfs_test.cpp
    src/memory/gpu_manager.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
    src/shared.h
)
add_dependencies(bfs_test device_ptx)

# Pass PTX path
target_compile_definitions(bfs_test PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

# Includes & libs
target_include_directories(bfs_test PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

target_link_libraries(bfs_test PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(bfs_test PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# RPATH for runtime libnvoptix
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(bfs_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(bfs_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()


###################################################### PR TEST

add_executable(pr_test
    test/pr_test.cpp
    src/memory/gpu_manager.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
    src/shared.h
)
add_dependencies(pr_test device_ptx)

# Pass PTX path
target_compile_definitions(pr_test PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

# Includes & libs
target_include_directories(pr_test PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

target_link_libraries(pr_test PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(pr_test PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# RPATH for runtime libnvoptix
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(pr_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(pr_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()

###################################################### SSSP TEST

add_executable(sssp_test
    test/sssp_test.cpp
    src/memory/gpu_manager.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
    src/shared.h
)
add_dependencies(sssp_test device_ptx)

# Pass PTX path
target_compile_definitions(sssp_test PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

# Includes & libs
target_include_directories(sssp_test PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

target_link_libraries(sssp_test PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(sssp_test PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# RPATH for runtime libnvoptix
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(sssp_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(sssp_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()

###################################################### BC TEST

add_executable(bc_test
    test/bc_test.cpp
    src/memory/gpu_manager.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
    src/shared.h
)
add_dependencies(bc_test device_ptx)

# Pass PTX path
target_compile_definitions(bc_test PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

# Includes & libs
target_include_directories(bc_test PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

target_link_libraries(bc_test PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(bc_test PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# RPATH for runtime libnvoptix
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(bc_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(bc_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()

###################################################### TC TEST

add_executable(tc_test
    test/tc_test.cpp
    src/memory/gpu_manager.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
    src/shared.h
)
add_dependencies(tc_test device_ptx)

# Pass PTX path
target_compile_definitions(tc_test PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

# Includes & libs
target_include_directories(tc_test PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

target_link_libraries(tc_test PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(tc_test PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# RPATH for runtime libnvoptix
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(tc_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(tc_test PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()

###################################################### PYBIND11 MODULE (pygraph_rtx)

# ---- Find or fetch pybind11 ----
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found, fetching...")
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.5
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ---- Python binding source ----
set(PYGRAPH_SRC
    python/py_graphrtx.cpp 
    src/memory/gpu_manager.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
)

# ---- Create module ----
pybind11_add_module(pygraph_rtx MODULE ${PYGRAPH_SRC})
add_dependencies(pygraph_rtx device_ptx)
set_target_properties(pygraph_rtx PROPERTIES SUFFIX ".so")

# ---- Definitions and includes ----
target_compile_definitions(pygraph_rtx PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

target_include_directories(pygraph_rtx PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

# ---- Link CUDA, OptiX, etc ----
target_link_libraries(pygraph_rtx PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(pygraph_rtx PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# ---- RPATH for OptiX ----
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(pygraph_rtx PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(pygraph_rtx PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()

# Host executable for update benchmark
add_executable(bench_updates
    src/memory/gpu_manager.cpp
    bench/bench_updates.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/algorithms/wcc.cpp
    src/algorithms/cdlp.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
    src/kernels/cuda/wcc.cu
    src/kernels/cuda/cdlp.cu
    src/shared.h
)
add_dependencies(bench_updates device_ptx)

# Pass PTX path
target_compile_definitions(bench_updates PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

# Includes & libs
target_include_directories(bench_updates PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

target_link_libraries(bench_updates PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(bench_updates PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# RPATH for runtime libnvoptix
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(bench_updates PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(bench_updates PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()

##################################


# Host executable for update benchmark
add_executable(bench_multi_gpu
    src/memory/gpu_manager.cpp
    bench/bench_multi_gpu.cpp
    src/rt/rt_pipeline.cpp
    src/graph/graph.cpp
    src/graph/algorithms.cpp
    src/algorithms/partition.cpp
    src/algorithms/bfs.cpp
    src/algorithms/pr.cpp
    src/algorithms/sssp.cpp
    src/algorithms/bc.cpp
    src/algorithms/tc.cpp
    src/algorithms/wcc.cpp
    src/algorithms/cdlp.cpp
    src/kernels/cuda/bfs.cu
    src/kernels/cuda/tc.cu
    src/kernels/cuda/pr.cu
    src/kernels/cuda/bc.cu
    src/kernels/cuda/sssp.cu
    src/kernels/cuda/wcc.cu
    src/kernels/cuda/cdlp.cu
    src/shared.h
)
add_dependencies(bench_multi_gpu device_ptx)

# Pass PTX path
target_compile_definitions(bench_multi_gpu PRIVATE DEVICE_PTX_PATH="${PTX_OUT}")

# Includes & libs
target_include_directories(bench_multi_gpu PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPTIX_ROOT}/include
    src
)

target_link_libraries(bench_multi_gpu PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cuda
    dl
)

if(OPTIX_DRIVER_LIB)
    target_link_libraries(bench_multi_gpu PRIVATE ${OPTIX_DRIVER_LIB})
endif()

# RPATH for runtime libnvoptix
if(UNIX)
  if(EXISTS "${OPTIX_ROOT}/lib64")
    set_target_properties(bench_multi_gpu PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib64"
      INSTALL_RPATH "${OPTIX_ROOT}/lib64")
  elseif(EXISTS "${OPTIX_ROOT}/lib")
    set_target_properties(bench_multi_gpu PROPERTIES
      BUILD_RPATH "${OPTIX_ROOT}/lib"
      INSTALL_RPATH "${OPTIX_ROOT}/lib")
  endif()
endif()