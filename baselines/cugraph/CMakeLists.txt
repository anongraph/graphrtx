cmake_minimum_required(VERSION 3.23)
project(cugraph_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find RAPIDS dependencies installed by your cuGraph build
find_package(CUDAToolkit REQUIRED)
find_package(rmm REQUIRED)
find_package(raft REQUIRED)
find_package(cugraph REQUIRED)

add_executable(cugraph_app main.cpp)
target_compile_options(cugraph_app PRIVATE "-O3")
target_link_libraries(cugraph_app PRIVATE cugraph::cugraph rmm::rmm raft::raft)
target_compile_definitions(cugraph_app PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)

