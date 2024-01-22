# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.18)
project(nsight_vlog_memory_workload CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

FetchContent_Declare(lodepng
        GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
)
FetchContent_MakeAvailable(lodepng)

function(set_target_options target)
    target_include_directories(${target} PRIVATE
            ${lodepng_SOURCE_DIR}
            src/
    )

    target_link_libraries(${target}
            CUDA::cudart
    )

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    else ()
        # needed to get SASS correlation in NCU
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif ()


    set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
endfunction()

# use this for profiling in nsight compute
add_executable(nsight_vlog_memory_workload
        src/main.cu
        ${lodepng_SOURCE_DIR}/lodepng.cpp
)
set_target_options(nsight_vlog_memory_workload)

# use this for simply running the kernels in a benchmark mode
add_executable(memory_workload_benchmark
        src/main.cu
        ${lodepng_SOURCE_DIR}/lodepng.cpp
)
set_target_options(memory_workload_benchmark)
target_compile_definitions(memory_workload_benchmark PRIVATE BENCHMARK=1)