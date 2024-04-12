
cmake_minimum_required(VERSION 3.16)
project(diffusion_kernel LANGUAGES CUDA CXX)
find_package(CUDA 11.8 REQUIRED)

# CPP 17
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 11.8
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} libstdc++;
                                        -O3;
                                        -G;
                                        -g;
                                        --std=C++17;
                                        --lineinfo;
                                        --use_fast_math;
                                        -gencode arch=compute_86,code=sm_86)
set(CUDA_NVCC_FLAGS CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)


# Specify include directories
include_directories(
    include
    )

cuda_add_executable(
    diffusion_kernel
    diffusion.cu
)

target_include_directories(diffusion_kernel
        PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(diffusion_kernel
        ${CUDA_LIBRARIES}
        stdc++fs)
