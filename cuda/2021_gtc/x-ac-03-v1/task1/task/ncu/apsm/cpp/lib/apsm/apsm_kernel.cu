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
 * @file apsm_kernel.cu
 * @brief APSM RKHS kernel
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 * @author Lukas Buse,        HHI, lukas.buse@hhi.fraunhofer.de
 *
 * @date 2019.12.10   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

// CUDA LIB
#include <cuda_runtime.h>

// APSM helper
#include "apsm_kernel.cuh"

// reproducing kernel hilbert space (RKHS) kernel functions

// Gaussian kernel
/**
 * @brief CUDA shared device function of an gaussian APSM RKHS kernel
 * 
 * @param[in] length vector length
 * @param[in] basis basis vector
 * @param[in] data data vector
 * @param[in] variance variance value
 *
 * @return gaussian_inner_product 
 */
__device__
    apsm_fp
    gaussian_kernel( unsigned int length, const CudaDeviceDedupRingBuffer& basis, unsigned int basisIdx, const apsm_fp* data, apsm_fp variance )
{
    // TODO (mm) This for loop can be calculated in parallel,
    //           but the addition needs attention

    // calculate weight
    apsm_fp exp_weight = apsm_fp( -0.5 ) / variance;

    // calculate argument
    apsm_fp exp_argument = apsm_fp( 0.0 );
#pragma unroll
    for ( unsigned int dim = 0; dim < length; dim++ )
    {

        apsm_fp dist_element = basis( basisIdx, dim ) - data[ dim ];
        exp_argument += dist_element * dist_element; // alternative: pow( dist_element, apsm_fp( 2.0 ) );
    }

    // return gaussian kernel value
    return exp( exp_weight * exp_argument );
}

/**
 * @brief CUDA shared device function of an linear APSM RKHS kernel
 * 
 * @param[in] length vector length
 * @param[in] basis basis vector
 * @param[in] data data vector
 *
 * @return linear_inner_product 
 */
__device__
    apsm_fp
    linear_kernel( unsigned int length, const apsm_fp* basis, const apsm_fp* data )
{
    // TODO (mm) This for loop can be calculated in parallel,
    //           but the addition needs attention

    // calculate inner product
    apsm_fp inner_product = apsm_fp( 0.0 );
#pragma unroll
    for ( unsigned int dim = 0; dim < length; dim++ )
    {

        inner_product += basis[ dim ] * data[ dim ];
    }

    // return linear kernel value
    return inner_product;
}
