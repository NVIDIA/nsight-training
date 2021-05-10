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
 * @file cuda_indexing.cu
 * @brief CUDA kernel indexing
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.12.11   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 *
 * @note CUDA Thread Indexing
 *       https://blog.usejournal.com/cuda-thread-indexing-fb9910cba084
 *
 * @note CUDA Thread Indexing Cheatsheet
 *       https://www.trilliwon.com/ABL/2018-06-11/cuda-index
 */

// CUDA helper
#include "cuda_indexing.cuh"

/**
 * @brief Get block index.
 * @details Returning an unique block index.
 *          Blocks are 1D.
 *
 * @return block index
 */
__device__ unsigned int getBlockIdx_1D()
{
    return blockIdx.x;
}

/**
 * @brief Get block index.
 * @details Returning an unique block index.
 *          Blocks are 2D.
 *
 * @return block index
 */
__device__ unsigned int getBlockIdx_2D()
{
    return blockIdx.y * ( gridDim.x )
        + blockIdx.x;
}

/**
 * @brief Get block index.
 * @details Returning an unique block index.
 *          Blocks are 3D.
 *
 * @return block index
 */
__device__ unsigned int getBlockIdx_3D()
{
    return blockIdx.z * ( gridDim.y * gridDim.x )
        + blockIdx.y * ( gridDim.x )
        + blockIdx.x;
}

/**
 * @brief Get local thread index.
 * @details Returning an unique thread index within each block.
 *          Threads are 1D:
 *
 * @return block index
 */
__device__ unsigned int getBlockThreadIdx_1D()
{
    return threadIdx.x;
}

/**
 * @brief Get local thread index.
 * @details Returning an unique thread index within each block.
 *          Threads are 2D.
 *
 * @return block index
 */
__device__ unsigned int getBlockThreadIdx_2D()
{
    return threadIdx.y * getBlockDim_1D()
        + threadIdx.x;
}

/**
 * @brief Get local thread index.
 * @details Returning an unique thread index within each block.
 *          Threads are 3D.
 *
 * @return block index
 */
__device__ unsigned int getBlockThreadIdx_3D()
{
    return threadIdx.z * getBlockDim_2D()
        + threadIdx.y * getBlockDim_1D()
        + threadIdx.x;
}

/**
 * @brief Get global thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 1D and threads are 1D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_1D_1D()
{
    return getBlockIdx_1D() * getBlockDim_1D()
        + getBlockThreadIdx_1D();
}

/**
 * @brief Get global thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 2D and threads are 1D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_2D_1D()
{
    return getBlockIdx_2D() * getBlockDim_1D()
        + getBlockThreadIdx_1D();
}

/**
 * @brief Get global thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 3D and threads are 1D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_3D_1D()
{
    return getBlockIdx_3D() * getBlockDim_1D()
        + getBlockThreadIdx_1D();
}

/**
 * @brief Get global thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 1D and threads are 2D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_1D_2D()
{
    return getBlockIdx_1D() * getBlockDim_2D()
        + getBlockThreadIdx_2D();
}

/**
 * @brief Get global thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 2D and threads are 2D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_2D_2D()
{
    return getBlockIdx_2D() * getBlockDim_2D()
        + getBlockThreadIdx_2D();
}

/**
 * @brief Get global thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 3D and threads are 2D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_3D_2D()
{
    return getBlockIdx_3D() * getBlockDim_2D()
        + getBlockThreadIdx_2D();
}

/**
 * @brief Get global thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 1D and threads are 3D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_1D_3D()
{
    return getBlockIdx_1D() * getBlockDim_3D()
        + getBlockThreadIdx_3D();
}

/**
 * @brief Get gloabl thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 2D and threads are 3D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_2D_3D()
{
    return getBlockIdx_2D() * getBlockDim_3D()
        + getBlockThreadIdx_3D();
}

/**
 * @brief Get global thread index.
 * @details Returning an unique thread index over all blocks.
 *          Blocks are 3D and threads are 3D.
 *
 * @return block index
 */
__device__ unsigned int getThreadIdx_3D_3D()
{
    return getBlockIdx_3D() * getBlockDim_3D()
        + getBlockThreadIdx_3D();
}

/**
 * @brief Get block dimension.
 * 
 * @return block dimension
 */
__device__ unsigned int getBlockDim_1D()
{
    return blockDim.x;
}

/**
 * @brief Get block dimension.
 * 
 * @return block dimension
 */
__device__ unsigned int getBlockDim_2D()
{
    return blockDim.x * blockDim.y;
}

/**
 * @brief Get block dimension.
 * 
 * @return block dimension
 */
__device__ unsigned int getBlockDim_3D()
{
    return blockDim.x * blockDim.y * blockDim.z;
}

/**
 * @brief Kernel call grid and block dimension calculation
 * @details This function is used in train and detection.
 *
 * @param[out] grid_dim One dimensional grid with input_vectors_size blocks 
 * @param[out] block_dim One dimensional block with APSM_BLOCK_BATCH_SIZE threads
 * @param[in] input_vector_size 
 * 
 * @return void 
 */
__host__ void apsm_kernel_dims( dim3* block_dim, dim3* grid_dim, unsigned int input_vector_size )
{

    // Number of blocks in a grid
    grid_dim->x = input_vector_size;
    grid_dim->y = 1;
    grid_dim->z = 1;

    // Number of threads per block
    block_dim->x = APSM_BLOCK_BATCH_SIZE;
    block_dim->y = 1;
    block_dim->z = 1;
}
