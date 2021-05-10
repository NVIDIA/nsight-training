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
 * @file cuda_errorhandling.cuh
 * @brief CUDA error handling header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.06.17   0.01    initial version
 *
 * @note How to do error checking in CUDA
 *       https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
 */

#pragma once

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

// STD
#include <stdio.h>
#include <stdlib.h> /* exit, EXIT_FAILURE */

// CUDA LIB
#include <cuda_runtime.h>

// ERROR checks on or off
#if 1
/**
 * @brief Enable CUDA Error check
 */
#define CUDA_ERROR_CHECK
#endif // !CUDA_ERROR_CHECK

/**
 * @brief Helper function to print file and line information, gets error code as input.
 */
#define CUDA_CHECK( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

/**
 * @brief Helper function to print file and line information, read last error.
 */
#define CUDA_CHECK_ERROR() __cudaCheckError( __FILE__, __LINE__ )

/**
 * @brief CUDA safe call inline function.
 */
inline void __cudaSafeCall( cudaError err, const char* file, const int line )
{

#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA_CHECK( ... ) failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( EXIT_FAILURE );
    }
#endif

    return;
}

/**
 * @brief CUDA error check inline function.
 */
inline void __cudaCheckError( const char* file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA_CHECK_ERROR() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( EXIT_FAILURE );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "CUDA_CHECK_ERROR() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( EXIT_FAILURE );
    }
#endif

    return;
}

/**
 * @}
 */
