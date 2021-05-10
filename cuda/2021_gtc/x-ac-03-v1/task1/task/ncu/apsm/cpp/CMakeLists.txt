# Copyright (c) 2019-2021
# Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
# All rights reserved.
# 
# Licensed by NVIDIA CORPORATION with permission. 
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# 
# NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, including without 
# limitation the patents of Fraunhofer, ARE GRANTED BY THIS SOFTWARE LICENSE. 
# Fraunhofer provides no warranty of patent non-infringement with respect to 
# this software. 

cmake_minimum_required( VERSION 3.10 FATAL_ERROR )
cmake_policy( VERSION 3.10.0 )

project ( APSM
    VERSION 0.1.0
    LANGUAGES C CXX CUDA
)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )
set( CMAKE_INSTALL_PREFIX install )

##################################################################
# Dependencies
##################################################################

set( CUDA_MIN_VERSION 10.0 )

###

# Add CUDA and Thrust dependencies to this project
find_package( CUDA ${CUDA_MIN_VERSION} REQUIRED )
find_package( Thrust REQUIRED )

##################################################################
# Configure Compiler
##################################################################

# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set( CMAKE_CXX_STANDARD 17 ) # 11, 14 or 17
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

#include(CheckCXXCompilerFlag)
# Check for standard to use
#check_cxx_compiler_flag( -std=c++17 HAVE_FLAG_STD_CXX17 )
#if( HAVE_FLAG_STD_CXX17 )
#    # Have -std=c++17, use it
#    message( STATUS "Use c++ 17" )
#endif()

##################################################################
# Configure CUDA
##################################################################

#https://devblogs.nvidia.com/building-cuda-applications-cmake/

#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-std=" )
#    list ( APPEND CUDA_NVCC_FLAGS "-std=c++14" )
#endif ()

#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-expt-extended-lambda" )
#    list ( APPEND CUDA_NVCC_FLAGS "-expt-extended-lambda" )
#endif ()

# create relocatable device code
#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-rdc=" )
#    list ( APPEND CUDA_NVCC_FLAGS "-rdc=true" )
#endif ()

# fast math
#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-use_fast_math" )
#    list ( APPEND CUDA_NVCC_FLAGS "-use_fast_math" )
#endif ()

# Lineinfo (--generate-line-info/-lineinfo)
#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-lineinfo" )
#    list ( APPEND CUDA_NVCC_FLAGS "-lineinfo" )
#endif ()

# Host debug (--debug/-g)
#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-g" )
#    list ( APPEND CUDA_NVCC_FLAGS "-g" )
#endif ()

# Device debug (--device-debug/-G)
#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-G" )
#    list ( APPEND CUDA_NVCC_FLAGS "-G" )
#endif ()

#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "--cudart" )
#    list ( APPEND CMAKE_CUDA_FLAGS "--cudart shared" )
#endif ()

# Sample flags for generation on CUDA 9 for maximum compatibility with Volta cards.
# https://docs.nvidia.com/cuda/volta-compatibility-guide/index.html
# TESLA V100
#set ( GPU_COMPUTE_CAPABILITIES 50 52 60 61 70 )
#set ( GPU_COMPUTE_CAPABILITY 70 )

# Sample flags for generation on CUDA 10 for maximum compatibility with Turing cards.
# https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html
# RTX 2080 TI
#set ( GPU_COMPUTE_CAPABILITIES 50 52 60 61 70 75 )
#set ( GPU_COMPUTE_CAPABILITY 75 )

# Sample flags for generation on CUDA 10 for maximum compatibility with Turing cards.
# https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html
# RTX 3080 TI
#set ( GPU_COMPUTE_CAPABILITIES 50 52 60 61 70 75 80 86 )
#set ( GPU_COMPUTE_CAPABILITY 86 )

# cooperative groups functionality is not provided by 50 and 52
set ( GPU_COMPUTE_CAPABILITIES 60 61 70 75 )
set ( GPU_COMPUTE_CAPABILITY 75 )
if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-gencode" )
    list ( APPEND CUDA_NVCC_FLAGS "-arch=sm_60" )
    # GPU Compute Capability: real architecture (code=sm_X)
    foreach( GPU_CAP IN LISTS GPU_COMPUTE_CAPABILITIES )
        list ( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${GPU_CAP},code=sm_${GPU_CAP}" )
    endforeach()
    # GPU Compute Capability: virtual architecture (code=compute_X)
    list ( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${GPU_COMPUTE_CAPABILITY},code=compute_${GPU_COMPUTE_CAPABILITY}" )
endif ()

#if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-O" )
#    #list ( APPEND CUDA_NVCC_FLAGS "-O2" )
#    list ( APPEND CUDA_NVCC_FLAGS "-O3" )
#endif ()

# add param list to CMAKE CUDA FLAGS
message(STATUS "CMAKE CUDA flags for nvcc: ${CMAKE_CUDA_FLAGS}")
string( REPLACE ";" " " CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
string( CONCAT CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} " " ${CUDA_NVCC_FLAGS} )
message(STATUS "CMAKE CUDA flags for nvcc: ${CMAKE_CUDA_FLAGS}")

##################################################################
# Subprojects
##################################################################

if ( CUDA_FOUND AND THRUST_FOUND )

    message( STATUS "Building APSM lib" )
    add_subdirectory( lib )

    # when we build the APSM lib we also build the tool
    # no dependencies only pure C++ and the lib
    message( STATUS "Building APSM tool" )
    add_subdirectory( cli )

endif( CUDA_FOUND AND THRUST_FOUND )

##################################################################
# Subprojects Optional
##################################################################

option( PACKAGE_DOCS "Build documentation" OFF )

if( PACKAGE_DOCS )
    add_subdirectory( docs )
endif()
