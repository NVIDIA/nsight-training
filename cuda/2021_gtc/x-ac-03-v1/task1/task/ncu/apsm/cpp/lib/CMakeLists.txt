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

#https://devblogs.nvidia.com/building-cuda-applications-cmake/

set(SOURCES_APSM_LIBRARY
    apsm/apsm_detect.cu
    apsm/apsm_detect.cuh
    apsm/apsm_harddecoder.cu
    apsm/apsm_harddecoder.cuh
    apsm/apsm_kernel.cu
    apsm/apsm_kernel.cuh
    apsm/apsm_matrix.cu
    apsm/apsm_matrix.cuh
    apsm/apsm_parameters.cu
    apsm/apsm_parameters.cuh
    apsm/apsm_train.cu
    apsm/apsm_train.cuh
    apsm/apsm_types.cuh
    apsm/apsm_versions.h
    apsm/apsm_wrapper.cu
    apsm/apsm_wrapper.cuh
    apsm/cuda_device.cu
    apsm/cuda_device.cuh
    apsm/cuda_errorhandling.cuh
    apsm/cuda_eventtimer.cuh
    apsm/cuda_indexing.cu
    apsm/cuda_indexing.cuh
    apsm/cuda_types.cuh
)

### Make the library ######################################################

# create an library in which the tests will be stored
add_library( apsm_lib ${SOURCES_APSM_LIBRARY} )

# Create a static library to link the code to the tests
target_include_directories( apsm_lib PUBLIC ${CUDA_INCLUDE_DIRS} )

# we using separate cu files in lib
set_target_properties( apsm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# compiler more strict
target_compile_options( apsm_lib PRIVATE --Werror all-warnings,cross-execution-space-call )
