/*** 
 * Copyright (c) 2019-2021
 * Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
 * All rights reserved.
 * 
 * Licensed by NVIDIA CORPORATION with permission. 
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 * 
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, including without 
 * limitation the patents of Fraunhofer, ARE GRANTED BY THIS SOFTWARE LICENSE. 
 * Fraunhofer provides no warranty of patent non-infringement with respect to 
 * this software. 
 */
 
/**
 * @file cuda_indexing.cuh
 * @brief CUDA kernel indexing header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.12.11   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

#pragma once

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

// CUDA LIB
#include <cuda_runtime.h>

/**
 * @brief Defines number of blocks (SM's) in a APSM kernel
 * 
 * @note Volta architecture, V100, 80 SMs
 * @note Turing architecture, RTX 2080 TI, 68 SMs
 */
#define APSM_BLOCK_BATCH_SIZE 32

__device__ unsigned int getBlockIdx_1D();

__device__ unsigned int getBlockIdx_2D();

__device__ unsigned int getBlockIdx_3D();

__device__ unsigned int getBlockThreadIdx_1D();

__device__ unsigned int getBlockThreadIdx_2D();

__device__ unsigned int getBlockThreadIdx_3D();

__device__ unsigned int getThreadIdx_1D_1D();

__device__ unsigned int getThreadIdx_2D_1D();

__device__ unsigned int getThreadIdx_3D_1D();

__device__ unsigned int getThreadIdx_1D_2D();

__device__ unsigned int getThreadIdx_2D_2D();

__device__ unsigned int getThreadIdx_3D_2D();

__device__ unsigned int getThreadIdx_1D_3D();

__device__ unsigned int getThreadIdx_2D_3D();

__device__ unsigned int getThreadIdx_3D_3D();

__device__ unsigned int getBlockDim_1D();

__device__ unsigned int getBlockDim_2D();

__device__ unsigned int getBlockDim_3D();

__host__ void apsm_kernel_dims( dim3*, dim3*, unsigned int );

/**
 * @}
 */
