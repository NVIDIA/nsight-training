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
 * @file apsm_detect.cu
 * @brief APSM detect kernel
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 * @author Lukas Buse,        HHI, lukas.buse@hhi.fraunhofer.de
 *
 * @date 2019.11.25   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

// CUDA helper
#include "cuda_errorhandling.cuh"
#include "cuda_eventtimer.cuh"
#include "cuda_indexing.cuh"

#include "apsm_wrapper.cuh"
#include "apsm_versions.h"

// APSM detect include
#include "apsm_detect.cuh"

// Cooperative Groups
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// the warp size is very rarely changed, and it complicates reading the code.
// so we fix it with a define instead of setting it as a tune parameter in TuneKernel.
#define WARP_SIZE 32

/**
 * @brief Group tune parameters per function version
 * @details Different implementations of APSM Detect will have different operating points.
            This struct abstracts the tuning parameters so each version can have its own
            parameter set.
 */

template<int version_id> struct TuneKernel
{};


/**************************************************************************************************
 * APSM_DETECT_ORIGINAL - Original
 *************************************************************************************************/

template<> struct TuneKernel<APSM_DETECT_ORIGINAL>
{
    static const int BLOCK_BATCH_SIZE = 32;
};
 
/**
 * @brief CUDA shared detection device function
 * @details This is the original version called during the training phase.
 *
 * @param detectedSymbol 
 * @param linearLength 
 * @param basisLength 
 * @param[in] rxdata_input pointer to intermediate memory 
 * @param threadIdOffset 
 * @param[in] basis basis matrix (dictionary learned in train)
 * @param[in] coeff coefficient vector (dictionary learned in train)
 * @param[in] rx_data received rx constallation vector
 * @param[in] par APSM parameter set
 *
 * @return void 
 */
__device__ void kernel_apsm_detection( apsm_fp* detectedSymbol, unsigned int linearLength, unsigned int gaussianLength, apsm_fp* rxdata_input, unsigned int threadIdOffset, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{

    // register for unique indexing
    //const unsigned int threadId	    = getThreadIdx_1D_1D();
    const unsigned int blockId = getBlockIdx_1D(); // gives sample id
    const unsigned int blockThreadId = getBlockThreadIdx_1D(); // gives basis id

    const unsigned int batch_size = getBlockDim_1D();

    const unsigned int basisLength = max( linearLength, gaussianLength );

    // ---------------------------------------------------------------------------

    // set register to zero
    if ( blockThreadId < batch_size )
        detectedSymbol[ blockThreadId ] = 0;

    // Iterate through the rx data vector
    // this for loop can be done in parallel and is computed in batches
    for ( unsigned int batch_idx = blockThreadId; batch_idx < linearLength; batch_idx += batch_size )
    {
        rxdata_input[ batch_idx ] = rx_data( batch_idx, threadIdOffset );
    }

    // we have to be sure that all threads finished, before we can use rxdata_input
    __syncthreads();

    // ---------------------------------------------------------------------------

    // Iterate through the basis matrix
    // this foor loop can be done in parallel and is computed in batches
    for ( unsigned int basis_idx = blockThreadId; basis_idx < basisLength; basis_idx += batch_size )
    {
        // linear kernel and linear weight
        if ( basis_idx < linearLength )
        {
            // linear basis is identity matrix so kernel call is not necessary
            // furthermore basis does not contain linear part any more
            apsm_fp kernel_eval = par.linearKernelWeight * rxdata_input[ basis_idx ];
            kernel_eval *= trainingState.linearCoeffs( basis_idx );
            atomicAdd( &detectedSymbol[ blockThreadId % batch_size ], kernel_eval ); // because different threads writing to the same memory address
        }
        // Gaussian kernel and Gaussian weight
        if ( basis_idx < gaussianLength )
        {
            apsm_fp kernel_eval = par.gaussianKernelWeight * gaussian_kernel( linearLength, trainingState.basis, basis_idx, rxdata_input, par.gaussianKernelVariance );
            kernel_eval *= trainingState.gaussianCoeffs( basis_idx );
            atomicAdd( &detectedSymbol[ blockThreadId % batch_size ], kernel_eval ); // because different threads writing to the same memory address
        }
    }
}

/**
 * @brief CUDA detect kernel (original)
 * @details 
 * 
 * @param[in] basis basis matrix (dictionary learned in train)
 * @param[in] coeff coefficient vector (dictionary learned in train)
 * @param[in] rx_data received rx constallation vector
 * @param[out] det_out equalized or decoded constallation vector 
 * @param[in] par APSM parameter set
 *
 * @param[in] d_rxdata_input intermediate memory 
 *
 * @return void 
 */
 template<>
 __global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_ORIGINAL>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
 {
     // register for unique indexing
     const unsigned int blockId = getBlockIdx_1D();             // gives sample id
     const unsigned int blockThreadId = getBlockThreadIdx_1D(); // gives basis id
 
     const unsigned int batch_size = getBlockDim_1D();
 
     const unsigned int linearLength = rx_data.getHeight();
 
     const unsigned int gaussianLength = trainingState.basis.getUsedHeight();
 
     extern __shared__ apsm_fp shared[];
 
     // ---------------------------------------------------------------------------
 
     // Check that blockid is not outside of the range
     if ( blockId >= rx_data.getWidth() ) // data length
         return;
 
     // ---------------------------------------------------------------------------
 
     // Shared memory for detected symbol vector
     apsm_fp* detectedSymbol = &shared[ 0 ]; // detectedSymbol is manually set at the beginning of shared mem
     apsm_fp* rxdata_input = &shared[ batch_size ];
 
     // call shared detection function
     kernel_apsm_detection( detectedSymbol, linearLength, gaussianLength, rxdata_input,
                            blockId, trainingState, rx_data, par );
 
     // we have to be sure that all threads finished, and than write it back
     __syncthreads();
 
     // only one thread per block writes output to GPU memory
     if ( blockThreadId == 0 )
     {
         for ( unsigned int idx = 1; idx < batch_size; idx++ )
             detectedSymbol[ 0 ] += detectedSymbol[ idx ];
 
         det_out( 0, blockId ) = detectedSymbol[ 0 ];
     }
 }

// wrapper specialization
template<>
float ApsmWrapper::wrapperDetect<apsm_versions::APSM_DETECT_ORIGINAL>( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_ORIGINAL> TK;

    // Initialise timer
    CUDA_EventTimer timer;

    // get vector sizes
    unsigned int vector_size = d_apsm_rxd2r.getWidth(); // sample length

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, vector_size );

    // calculate shared memory size parameter
    unsigned int sharedMemorySize = 0;
    sharedMemorySize += TK::BLOCK_BATCH_SIZE * sizeof( apsm_fp );
    sharedMemorySize += d_apsm_rxd2r.getHeight() * sizeof( apsm_fp );

    // run kernel and measure time
    timer.start( stream );
    kernel_apsm_detect<apsm_versions::APSM_DETECT_ORIGINAL> <<<grid_dim, block_dim, sharedMemorySize, stream>>>( trainingState.toDevice(), d_apsm_rxd2r.toDevice(), d_apsm_esd2r.toDevice(), par );
    timer.stop( stream );

     // sync after kernel call
     CUDA_CHECK( cudaStreamSynchronize( stream ) );
 
     // give back the measured kernel processing time
     return timer.elapsed();
}


/**************************************************************************************************
 * APSM_DETECT_CG - Cooperative Groups
 *************************************************************************************************/

template<> struct TuneKernel<APSM_DETECT_CG>
{};
 
// gaussian kernel function modified to address rx_data directly 
 __device__ apsm_fp gaussian_kernel_cg( unsigned int length, const CudaDeviceDedupRingBuffer& basis, unsigned int basisIdx, const CudaDeviceDedupMatrix& data, const apsm_fp sample_idx, apsm_fp variance )
{
 // calculate weight
 apsm_fp exp_weight = apsm_fp( -0.5 ) / variance;

 // calculate argument
 apsm_fp exp_argument = apsm_fp( 0.0 );
#pragma unroll
 for ( unsigned int dim = 0; dim < length; dim++ )
 {
     apsm_fp dist_element = basis( basisIdx, dim ) - data( dim , sample_idx );
     exp_argument += dist_element * dist_element; // alternative: pow( dist_element, apsm_fp( 2.0 ) );
 }

 // return gaussian kernel value
 return exp( exp_weight * exp_argument );
}

__device__ void kernel_apsm_detection_cg( cg::thread_block& tg, const unsigned int sample_idx, apsm_fp& detSymbol, const unsigned int linearLength, const unsigned int gaussianLength, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_CG> TKP;

    // work over a tile, that does work for a single sample.

    __shared__ apsm_fp detectedSymbol[ WARP_SIZE ];
    
    const int tid = tg.thread_rank();
    const int basisLength = max( linearLength, gaussianLength );

    // ---------------------------------------------------------------------------

    // set register to zero
    detectedSymbol[ tg.thread_rank() ] = 0.0;

    // ---------------------------------------------------------------------------

    // Iterate through the basis matrix
    // this foor loop can be done in parallel and is computed in batches
    for ( unsigned int basis_idx = tg.thread_rank(); basis_idx < basisLength; basis_idx += tg.size() )
    {
        // linear kernel and linear weight
        if ( basis_idx < linearLength )
        {
            // linear basis is identity matrix so kernel call is not necessary
            // furthermore basis does not contain linear part any more
            apsm_fp kernel_eval = par.linearKernelWeight * rx_data( basis_idx, sample_idx );
            kernel_eval *= trainingState.linearCoeffs( basis_idx );
            atomicAdd( &detectedSymbol[ tid ], kernel_eval ); // because different threads writing to the same memory address
        }

        // Gaussian kernel and Gaussian weight
        if ( basis_idx < gaussianLength )
        {
            apsm_fp kernel_eval = par.gaussianKernelWeight * gaussian_kernel_cg( linearLength, trainingState.basis, basis_idx, rx_data, sample_idx, par.gaussianKernelVariance );
            kernel_eval *= trainingState.gaussianCoeffs( basis_idx );
            atomicAdd( &detectedSymbol[ tid ], kernel_eval ); // because different threads writing to the same memory address
        }
    }

    for ( int i = tg.size() / 2; i > 0; i /= 2 )
    {
        if ( tg.thread_rank() < i)
            detectedSymbol[ tid ] += detectedSymbol[ tid + i ];
        tg.sync();
    }
    if ( tg.thread_rank() == 0 )
    {
        detSymbol = detectedSymbol[0];
    }
}

template<>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_CG>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
     cg::thread_block block = cg::this_thread_block(); // the block running in this SM
     const unsigned int sample_idx = block.group_index().x; // the sample index is the block index in the grid
  
     // Check that the sample idx is not outside of the range
     if ( sample_idx >= rx_data.getWidth() ) // data length
         return;
 
     const unsigned int linearLength = rx_data.getHeight();
     const unsigned int gaussianLength = trainingState.gaussianCoeffs.getUsedHeight();
 
     // every tile will work in a sample and return the detected symbol for that sample
     apsm_fp detected_symbol = apsm_fp( 0.0 );
 
     kernel_apsm_detection_cg(   block,
                                 sample_idx,
                                 detected_symbol,
                                 linearLength,
                                 gaussianLength,
                                 trainingState,
                                 rx_data,
                                 par );
 
     // write detected symbol to global memory
     if ( block.thread_rank() == 0 )
         det_out( 0, sample_idx ) = detected_symbol;
 
     return;
}

// wrapper specialization
template<>
float ApsmWrapper::wrapperDetect<apsm_versions::APSM_DETECT_CG>( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_CG> TKP;

    // Initialise timer
    CUDA_EventTimer timer;

    // get vector sizes
    unsigned int vector_size = d_apsm_rxd2r.getWidth(); // sample length

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, vector_size );

    // calculate dynamic shared memory size parameter
    unsigned int sharedMemorySize = 0;

    // fix launch dimensions
    grid_dim.x = vector_size;
    block_dim.x = WARP_SIZE;

    // run kernel and measure time
    timer.start( stream );
    kernel_apsm_detect<apsm_versions::APSM_DETECT_CG> <<<grid_dim, block_dim, sharedMemorySize, stream>>>( trainingState.toDevice(), d_apsm_rxd2r.toDevice(), d_apsm_esd2r.toDevice(), par );
    timer.stop( stream );

    // sync after kernel call
    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    // give back the measured kernel processing time
    return timer.elapsed();
}


/**************************************************************************************************
 * APSM_DETECT_SPB - Multiple Samples per Block
 *************************************************************************************************/

template<> struct TuneKernel<APSM_DETECT_SPB>
{
    static const int SAMPLES_PER_BLOCK            = 4;
    static const int SHMEM_DETECTEDSYMBOL_SIZE    = SAMPLES_PER_BLOCK * WARP_SIZE;
};

__device__ void kernel_apsm_detection_spb( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, const unsigned int sample_idx, apsm_fp& detSymbol, const unsigned int linearLength, const unsigned int gaussianLength, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SPB> TKP;

    // work over a tile, that does work for a single sample.

    __shared__ apsm_fp shmem_detectedSymbol[ TKP::SHMEM_DETECTEDSYMBOL_SIZE ];
    apsm_fp* detectedSymbol = &shmem_detectedSymbol[ tg.meta_group_rank() * WARP_SIZE ];
    
    const int tid = tg.thread_rank();
    const int basisLength = max( linearLength, gaussianLength );

    // ---------------------------------------------------------------------------

    // set register to zero
    detectedSymbol[ tg.thread_rank() ] = 0.0;

    // ---------------------------------------------------------------------------

    // Iterate through the basis matrix
    // this foor loop can be done in parallel and is computed in batches
    for ( unsigned int basis_idx = tg.thread_rank(); basis_idx < basisLength; basis_idx += tg.size() )
    {
        // linear kernel and linear weight
        if ( basis_idx < linearLength )
        {
            // linear basis is identity matrix so kernel call is not necessary
            // furthermore basis does not contain linear part any more
            apsm_fp kernel_eval = par.linearKernelWeight * rx_data( basis_idx, sample_idx );
            kernel_eval *= trainingState.linearCoeffs( basis_idx );
            atomicAdd( &detectedSymbol[ tid ], kernel_eval ); // because different threads writing to the same memory address
        }

        // Gaussian kernel and Gaussian weight
        if ( basis_idx < gaussianLength )
        {
            apsm_fp kernel_eval = par.gaussianKernelWeight * gaussian_kernel_cg( linearLength, trainingState.basis, basis_idx, rx_data, sample_idx, par.gaussianKernelVariance );
            kernel_eval *= trainingState.gaussianCoeffs( basis_idx );
            atomicAdd( &detectedSymbol[ tid ], kernel_eval ); // because different threads writing to the same memory address
        }
    }

    for ( int i = tg.size() / 2; i > 0; i /= 2 )
    {
        if ( tg.thread_rank() < i)
            detectedSymbol[ tid ] += detectedSymbol[ tid + i ];
        tg.sync();
    }
    if ( tg.thread_rank() == 0 )
    {
        detSymbol = detectedSymbol[0];
    }
}

template<>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_SPB>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SPB> TKP;

    cg::thread_block block = cg::this_thread_block(); // the block running in this SM
    const unsigned int block_idx = block.group_index().x; // the block index in the grid

    cg::thread_block_tile<WARP_SIZE> sample_block = cg::tiled_partition<WARP_SIZE>( block ); // divide block in tiles of WARP_SIZE (eg 32) threads
    const unsigned int sample_idx = TKP::SAMPLES_PER_BLOCK * block_idx + sample_block.meta_group_rank(); // the sample index to be processed by a tile

    // Check that the sample idx is not outside of the range
    if ( sample_idx >= rx_data.getWidth() ) // data length
        return;

    const unsigned int linearLength = rx_data.getHeight();
    const unsigned int gaussianLength = trainingState.gaussianCoeffs.getUsedHeight();

    // every tile will work in a sample and return the detected symbol for that sample
    apsm_fp detected_symbol = apsm_fp( 0.0 );

    kernel_apsm_detection_spb(  block,
                                sample_block,
                                sample_idx,
                                detected_symbol,
                                linearLength,
                                gaussianLength,
                                trainingState,
                                rx_data,
                                par );

    // write detected symbol to global memory, one per sample
    if ( sample_block.thread_rank() == 0 )
        det_out( 0, sample_idx ) = detected_symbol;

    return;
}


/**************************************************************************************************
 * APSM_DETECT_SPLIT - Split linear and Gaussian loops
 *************************************************************************************************/

template<> struct TuneKernel<APSM_DETECT_SPLIT>
{
    static const int SAMPLES_PER_BLOCK          = 4;
    static const int SHMEM_DETECTEDSYMBOL_SIZE  = SAMPLES_PER_BLOCK * WARP_SIZE;
};
 
__device__ void kernel_apsm_detection_linear_split( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, apsm_fp* detectedSymbol, unsigned int linearLength, unsigned int sample_idx, const CudaDeviceMatrix& coeff, const CudaDeviceDedupMatrix& rx_data, const apsm_fp linearKernelWeight )
{
    for ( unsigned int idx = tg.thread_rank(); idx < linearLength; idx += tg.size() )
    {
        const unsigned int local_idx = tg.thread_rank();
        apsm_fp kernel_eval = linearKernelWeight * rx_data( idx, sample_idx );
        apsm_fp coeff_val = coeff( idx );

        atomicAdd( &detectedSymbol[ local_idx ], coeff_val * kernel_eval );
    }
}

__device__ void kernel_apsm_detection_gaussian_split( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, apsm_fp* detectedSymbol, unsigned int linearLength, unsigned int gaussianLength, unsigned int sample_idx, const CudaDeviceDedupRingBuffer& basis, const CudaDeviceRingBuffer& coeff, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    const int tid = tg.thread_rank();

    // Iterate through the basis matrix
    // this foor loop can be done in parallel and is computed in batches
    for ( unsigned int basis_idx = tg.thread_rank(); basis_idx < gaussianLength; basis_idx += tg.size() )
    {
        // Gaussian kernel and Gaussian weight
        apsm_fp kernel_eval = par.gaussianKernelWeight * gaussian_kernel_cg( linearLength, basis, basis_idx, rx_data, sample_idx, par.gaussianKernelVariance );
        kernel_eval *= coeff( basis_idx );
        atomicAdd( &detectedSymbol[ tid ], kernel_eval ); // because different threads writing to the same memory address
    }
}

__device__ void kernel_apsm_detection_split( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, const unsigned int sample_idx, apsm_fp& detSymbol, const unsigned int linearLength, const unsigned int gaussianLength, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SPLIT> TKP;

    // work over a tile, that does work for a single sample.

    __shared__ apsm_fp shmem_detectedSymbol[ TKP::SHMEM_DETECTEDSYMBOL_SIZE ];
    apsm_fp* detectedSymbol = &shmem_detectedSymbol[ tg.meta_group_rank() * WARP_SIZE ];
    
    const int tid = tg.thread_rank();

    // set register to zero
    detectedSymbol[ tg.thread_rank() ] = 0.0;

    // compute linear and gaussian contributions
    kernel_apsm_detection_linear_split( bg, tg, detectedSymbol, linearLength, sample_idx, trainingState.linearCoeffs, rx_data, par.linearKernelWeight );
    kernel_apsm_detection_gaussian_split( bg, tg, detectedSymbol, linearLength, gaussianLength, sample_idx, trainingState.basis, trainingState.gaussianCoeffs, rx_data, par );

    // reduce symbols
    apsm_fp acc = detectedSymbol[ tg.thread_rank() ];
    tg.sync();
    for ( int i = tg.size() / 2; i > 0; i /= 2 )
        acc += tg.shfl_down( acc, i );

    if ( tg.thread_rank() == 0 )
        detSymbol = acc;
}
 
template<>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_SPLIT>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SPLIT> TKP;

    cg::thread_block block = cg::this_thread_block(); // the block running in this SM
    const unsigned int block_idx = block.group_index().x; // the block index in the grid

    cg::thread_block_tile<WARP_SIZE> sample_block = cg::tiled_partition<WARP_SIZE>( block ); // divide block in tiles of WARP_SIZE (eg 32) threads
    const unsigned int sample_idx = TKP::SAMPLES_PER_BLOCK * block_idx + sample_block.meta_group_rank(); // the sample index to be processed by a tile

    // Check that the sample idx is not outside of the range
    if ( sample_idx >= rx_data.getWidth() ) // data length
        return;

    const unsigned int linearLength = rx_data.getHeight();
    const unsigned int gaussianLength = trainingState.gaussianCoeffs.getUsedHeight();

    // every tile will work in a sample and return the detected symbol for that sample
    apsm_fp detected_symbol = apsm_fp( 0.0 );

    kernel_apsm_detection_split( block,
                                sample_block,
                                sample_idx,
                                detected_symbol,
                                linearLength,
                                gaussianLength,
                                trainingState,
                                rx_data,
                                par );

    // write detected symbol to global memory, one per sample
    if ( sample_block.thread_rank() == 0 )
        det_out( 0, sample_idx ) = detected_symbol;

    return;
}


/**************************************************************************************************
 * APSM_DETECT_SHMEM - Store vectors in shared memory
 *************************************************************************************************/

template<> struct TuneKernel<APSM_DETECT_SHMEM>
{
    static const int SAMPLES_PER_BLOCK          = 4;
    static const int MAX_LINEAR_LENGTH          = 64;
    static const int PADDING                    = 1;
    static const int PADDED_LINEAR_LENGTH       = MAX_LINEAR_LENGTH + PADDING;
    static const int BASIS_PER_BATCH            = WARP_SIZE;
    static const int SHMEM_DETECTEDSYMBOL_SIZE  = SAMPLES_PER_BLOCK * WARP_SIZE;
    static const int SHMEM_RXDATA_SIZE          = PADDED_LINEAR_LENGTH * SAMPLES_PER_BLOCK;
    static const int SHMEM_BASIS_SIZE           = PADDED_LINEAR_LENGTH * BASIS_PER_BATCH;
};
 
__device__ void kernel_apsm_detection_gaussian_shmem( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, apsm_fp* detectedSymbol, unsigned int linearLength, unsigned int basisLength, unsigned int sample_idx, const CudaDeviceDedupRingBuffer& basis, const CudaDeviceRingBuffer& coeff, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SHMEM> TKP;

    const int tid = tg.thread_rank();
    const apsm_fp exp_weight = apsm_fp( -0.5 ) / par.gaussianKernelVariance;

    // Padding is used in the following shared memory arrays to avoid bank conflicts.

    // each tile can cache the input data in share memory
    __shared__ apsm_fp shmem_rxdata[ TKP::SHMEM_RXDATA_SIZE ];
    apsm_fp* data = &shmem_rxdata[ tg.meta_group_rank() * TKP::PADDED_LINEAR_LENGTH ];

    // each thread can cache a basis vector, and share it across other samples in the block
    __shared__ apsm_fp shmem_basis[ TKP::SHMEM_BASIS_SIZE ];
    apsm_fp* basis_sh = &shmem_basis[ tg.thread_rank() * TKP::PADDED_LINEAR_LENGTH ];

    // read rxdata into shared memory
    for( int idx = tg.thread_rank(); idx < linearLength; idx += tg.size() )
        data[ idx ] = rx_data( idx , sample_idx );
    tg.sync();

    // Iterate through the basis matrix
    // this foor loop can be done in parallel and is computed in batches
    // tg.size() and BASIS_PER_BATCH must match, otherwise not all basis are processed. Here both are WARP_SIZE.
    for ( int basis_offset = 0; basis_offset < basisLength; basis_offset += TKP::BASIS_PER_BATCH )
    {
        // wait before reading new data, avoid overwrite of data being processed by previous iteration.
        bg.sync();

        const int basis_idx = basis_offset + tg.thread_rank();

        // let the first tile (sample) in the block issue the data read operations.
        if ((tg.meta_group_rank() == 0) && (basis_idx < basisLength))
        {
            for( int idx = 0; idx < linearLength; idx ++ )
                basis_sh[ idx ] = basis( basis_idx, idx );
        }
        // tell all tiles (samples) to wait for data to be available.
        bg.sync();

        // since we are advancing in batches, it is possible that a basis index
        // goes above the limit. In that case, skip the rest of the loop.
        // Note that we do this after the sync of the block.
        if (basis_idx >= basisLength)
            continue;

        // Gaussian kernel and Gaussian weight
        apsm_fp kernel_eval = 0.0;

        // Embed gaussian kernel in loop, as we will now process the argument in batches
        apsm_fp exp_argument = apsm_fp( 0.0 );
        for ( int dim = 0; dim < linearLength; dim++ )
        {
            apsm_fp dist_element = basis_sh[ dim ] - data[ dim ];
            exp_argument += dist_element * dist_element;
        }
        kernel_eval = coeff( basis_idx ) * par.gaussianKernelWeight * exp( exp_weight * exp_argument );
        atomicAdd( &detectedSymbol[ tid ], kernel_eval ); // because different threads writing to the same memory address
    }
}
 
__device__ void kernel_apsm_detection_shmem( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, const unsigned int sample_idx, apsm_fp& detSymbol, const unsigned int linearLength, const unsigned int gaussianLength, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SHMEM> TKP;

    // work over a tile, that does work for a single sample.
    __shared__ apsm_fp shmem_detectedSymbol[ TKP::SHMEM_DETECTEDSYMBOL_SIZE ];
    apsm_fp* detectedSymbol = &shmem_detectedSymbol[ tg.meta_group_rank() * WARP_SIZE ];
    
    const int tid = tg.thread_rank();

    // set register to zero
    detectedSymbol[ tg.thread_rank() ] = 0.0;

    // compute linear and gaussian contributions
    kernel_apsm_detection_linear_split( bg, tg, detectedSymbol, linearLength, sample_idx, trainingState.linearCoeffs, rx_data, par.linearKernelWeight );
    kernel_apsm_detection_gaussian_shmem( bg, tg, detectedSymbol, linearLength, gaussianLength, sample_idx, trainingState.basis, trainingState.gaussianCoeffs, rx_data, par );

    // reduce symbols
    apsm_fp acc = detectedSymbol[ tg.thread_rank() ];
    tg.sync();
    for ( int i = tg.size() / 2; i > 0; i /= 2 )
        acc += tg.shfl_down( acc, i );

    if ( tg.thread_rank() == 0 )
        detSymbol = acc;
}
  
template<>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_SHMEM>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_SHMEM> TKP;

    cg::thread_block block = cg::this_thread_block(); // the block running in this SM
    const unsigned int block_idx = block.group_index().x; // the block index in the grid

    cg::thread_block_tile<WARP_SIZE> sample_block = cg::tiled_partition<WARP_SIZE>( block ); // divide block in tiles of WARP_SIZE (eg 32) threads
    const unsigned int sample_idx = TKP::SAMPLES_PER_BLOCK * block_idx + sample_block.meta_group_rank(); // the sample index to be processed by a tile

    // Check that the sample idx is not outside of the range
    if ( sample_idx >= rx_data.getWidth() ) // data length
        return;

    const unsigned int linearLength = rx_data.getHeight();
    const unsigned int gaussianLength = trainingState.gaussianCoeffs.getUsedHeight();

    // every tile will work in a sample and return the detected symbol for that sample
    apsm_fp detected_symbol = apsm_fp( 0.0 );

    kernel_apsm_detection_shmem( block,
                                sample_block,
                                sample_idx,
                                detected_symbol,
                                linearLength,
                                gaussianLength,
                                trainingState,
                                rx_data,
                                par );

    // write detected symbol to global memory, one per sample
    if ( sample_block.thread_rank() == 0 )
        det_out( 0, sample_idx ) = detected_symbol;

    return;
}


/**************************************************************************************************
 * APSM_DETECT_BALANCED - Balance computation and memory accesses
 *************************************************************************************************/

template<> struct TuneKernel<APSM_DETECT_BALANCED>
{
    static const int SAMPLES_PER_BLOCK          = 32;
    static const int PADDING                    = 1;
    static const int BASIS_PER_BATCH            = 4*WARP_SIZE;
    static const int SHMEM_DETECTEDSYMBOL_SIZE  = SAMPLES_PER_BLOCK * WARP_SIZE;
    static const int SHMEM_RXDATA_SIZE          = (WARP_SIZE + PADDING) * SAMPLES_PER_BLOCK;
    static const int SHMEM_BASIS_SIZE           = (WARP_SIZE + PADDING) * BASIS_PER_BATCH;
    static const int SHMEM_EXPARGUMENT_SIZE     = SAMPLES_PER_BLOCK * (BASIS_PER_BATCH + PADDING);
};

__device__ void kernel_apsm_detection_gaussian_balanced( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, apsm_fp* detectedSymbol, unsigned int linearLength, unsigned int gaussianLength, unsigned int sample_idx, const CudaDeviceDedupRingBuffer& basis, const CudaDeviceRingBuffer& coeff, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_BALANCED> TKP;

    const int tid = tg.thread_rank();
    const apsm_fp exp_weight = apsm_fp( -0.5 ) / par.gaussianKernelVariance;

    // each tile can cache a segment (WARP_SIZE) of the input data in share memory
    __shared__ apsm_fp shmem_rxdata[ TKP::SHMEM_RXDATA_SIZE ];
    apsm_fp* data = &shmem_rxdata[ tg.meta_group_rank() * (WARP_SIZE + TKP::PADDING) ];

    // each block can cache a batch of segments (WARP_SIZE) of basis vectors, and share it across other samples in the block
    __shared__ apsm_fp shmem_basis[ TKP::SHMEM_BASIS_SIZE ];

    // each tile keeps intermediate values for the exp_arguments in shared memory (as many as in a BASIS_PER_BATCH)
    __shared__ apsm_fp shmem_exp_argument[ TKP::SHMEM_EXPARGUMENT_SIZE ];
    apsm_fp* exp_argument = &shmem_exp_argument[ tg.meta_group_rank() * (TKP::BASIS_PER_BATCH + TKP::PADDING) ];

    // Iterate through the dictionary, in batches
    for ( int basis_offset = 0; basis_offset < gaussianLength; basis_offset += TKP::BASIS_PER_BATCH )
    {
        tg.sync();
        // clear exp_argument for all elements in the batch
        for( int i=tid; i< TKP::BASIS_PER_BATCH; i+= WARP_SIZE )
            exp_argument[ i ] = apsm_fp( 0.0 );

        // traverse the vector length in segments of WARP_SIZE, makes the algorithm independent of linear length
        for( int dim_offset = 0; dim_offset < linearLength; dim_offset += WARP_SIZE )
        {
            const apsm_fp dim_oob = ( dim_offset+tid < linearLength) ? 1.0 : 0.0;

            // wait before reading new data, avoid overwrite of data being processed by previous iteration.
            bg.sync();

            // load the data input segment
            data[ tid ] = dim_oob * rx_data( dim_offset + tid, sample_idx );

            // load the basis vectors segments

#if 0 // v1: fetch from single tile
            // use a single tile in the block, similar to SHMEM
            if (tg.meta_group_rank() == 0)
            {
                for( int bid=tid; bid<TKP::BASIS_PER_BATCH; bid += WARP_SIZE )
                {
                    const int basis_idx = basis_offset + bid;
                    const apsm_fp basis_oob = (basis_idx < gaussianLength) ? 1.0 : 0.0;
                    apsm_fp* basis_sh = &shmem_basis[ bid * (WARP_SIZE + TKP::PADDING) ];

                    const apsm_fp* basis_input = basis.getRawPointer( basis_idx );
                    for( int idx = 0; idx < WARP_SIZE; idx ++ )
                    {
                        const apsm_fp dim_oob2 = (dim_offset + idx < linearLength ) ? 1.0 : 0.0;
                        basis_sh[ idx ] = basis_oob * dim_oob2 * basis( basis_idx, dim_offset + idx );
                    }
                }
            }
#endif
#if 1 // v2: fetch using all warps in the block
            #pragma unroll
            for( int bid=tid; bid<TKP::BASIS_PER_BATCH; bid += WARP_SIZE )
            {
                const int basis_idx = basis_offset + bid;
                const apsm_fp basis_oob = (basis_idx < gaussianLength) ? 1.0 : 0.0;
                apsm_fp* basis_sh = &shmem_basis[ bid * (WARP_SIZE + TKP::PADDING) ];

                #pragma unroll
                for( int idx=tg.meta_group_rank(); idx < WARP_SIZE; idx += tg.meta_group_size() )
                {
                    const apsm_fp dim_oob2 = (dim_offset + idx < linearLength ) ? 1.0 : 0.0;
                    basis_sh[ idx ] = basis_oob * dim_oob2 * basis( basis_idx, dim_offset + idx );
                }
        }
#endif
            // tell all tiles (samples) to wait for data to be available.
            bg.sync();

            // compute stage, part 1
            // Process the batch for this segment and save the intermediate result in shared memory
            #pragma unroll
            for( int bid=tid; bid<TKP::BASIS_PER_BATCH; bid += WARP_SIZE )
            {
                apsm_fp* basis_sh = &shmem_basis[ bid * (WARP_SIZE + TKP::PADDING) ];
                #pragma unroll
                for ( int dim = 0; dim < min(WARP_SIZE,linearLength); dim++ )
                {
                    apsm_fp dist_element = basis_sh[ dim ] - data[ dim ];
                    exp_argument[ bid ] += dist_element * dist_element;
                }
            }
        }
        tg.sync();

        // compute stage, part 2: now that all contributions in exp_argument are complete,
        // finalize the detected symbol computation, again going over the whole batch
        #pragma unroll
        for( int bid=tid; bid<TKP::BASIS_PER_BATCH; bid += WARP_SIZE )
        {
            const int basis_idx = basis_offset + bid;

            if (basis_idx >= gaussianLength)
                continue;

            apsm_fp kernel_eval = coeff( basis_idx ) * par.gaussianKernelWeight * exp( exp_weight * exp_argument[ bid ] );
            atomicAdd( &detectedSymbol[ tid ], kernel_eval ); // because different threads writing to the same memory address
        }
    }
}
  
__device__ void kernel_apsm_detection_balanced( cg::thread_block& bg, cg::thread_block_tile<WARP_SIZE>& tg, const unsigned int sample_idx, apsm_fp& detSymbol, const unsigned int linearLength, const unsigned int gaussianLength, const DeviceTrainingState& trainingState, const CudaDeviceDedupMatrix& rx_data, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_BALANCED> TKP;

    // work over a tile, that does work for a single sample.

    __shared__ apsm_fp shmem_detectedSymbol[ TKP::SHMEM_DETECTEDSYMBOL_SIZE ];
    apsm_fp* detectedSymbol = &shmem_detectedSymbol[ tg.meta_group_rank() * WARP_SIZE ];
    
    const int tid = tg.thread_rank();

    // set register to zero
    detectedSymbol[ tg.thread_rank() ] = 0.0;

    // compute linear and gaussian contributions
    kernel_apsm_detection_linear_split( bg, tg, detectedSymbol, linearLength, sample_idx, trainingState.linearCoeffs, rx_data, par.linearKernelWeight );
    kernel_apsm_detection_gaussian_balanced( bg, tg, detectedSymbol, linearLength, gaussianLength, sample_idx, trainingState.basis, trainingState.gaussianCoeffs, rx_data, par );

    // reduce symbols
    apsm_fp acc = detectedSymbol[ tg.thread_rank() ];
    tg.sync();
    for ( int i = tg.size() / 2; i > 0; i /= 2 )
        acc += tg.shfl_down( acc, i );

    if ( tg.thread_rank() == 0 )
        detSymbol = acc;
}
   
template<>
__global__ void kernel_apsm_detect<apsm_versions::APSM_DETECT_BALANCED>( const DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, CudaDeviceMatrix det_out, const apsm_parameters par )
{
    // rename type for easier reference
    typedef struct TuneKernel<apsm_versions::APSM_DETECT_BALANCED> TKP;

    cg::thread_block block = cg::this_thread_block(); // the block running in this SM
    const unsigned int block_idx = block.group_index().x; // the block index in the grid

    cg::thread_block_tile<WARP_SIZE> sample_block = cg::tiled_partition<WARP_SIZE>( block ); // divide block in tiles of WARP_SIZE (eg 32) threads
    const unsigned int sample_idx = TKP::SAMPLES_PER_BLOCK * block_idx + sample_block.meta_group_rank(); // the sample index to be processed by a tile

    // Check that the sample idx is not outside of the range
    if ( sample_idx >= rx_data.getWidth() ) // data length
        return;

    const unsigned int linearLength = rx_data.getHeight();
    const unsigned int gaussianLength = trainingState.gaussianCoeffs.getUsedHeight();

    // every tile will work in a sample and return the detected symbol for that sample
    apsm_fp detected_symbol = apsm_fp( 0.0 );

    kernel_apsm_detection_balanced( block,
                                sample_block,
                                sample_idx,
                                detected_symbol,
                                linearLength,
                                gaussianLength,
                                trainingState,
                                rx_data,
                                par );

    // write detected symbol to global memory, one per sample
    if ( sample_block.thread_rank() == 0 )
        det_out( 0, sample_idx ) = detected_symbol;

    return;
}


/**************************************************************************************************
 * APSM Wrapper Function - Entry point to the detect function from the wrapper code
 *************************************************************************************************/

/**
 * @brief C++ wrapper function for APSM detection part.
 * @details This function call the APSM CUDA detect kernel.
 *
 * @param[in] d_apsm_basis basis matrix (dictionary learned in train)
 * @param[in] d_apsm_coeff coefficient vector (dictionary learned in train)
 * @param[in] d_apsm_rxd2r received rx constallation vector
 * @param[out] d_apsm_esd2r equalized or decoded constallation vector 
 * @param[in] par APSM parameter set
 *
 * @return float measured kernel processing time
 */
template< int version_id >
float ApsmWrapper::wrapperDetect( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par )
{
    // rename type for easier reference
    typedef struct TuneKernel<version_id> TKP;

    // Initialise timer
    CUDA_EventTimer timer;

    // get vector sizes
    unsigned int vector_size = d_apsm_rxd2r.getWidth(); // sample length

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, vector_size );

    // calculate dynamic shared memory size parameter
    unsigned int sharedMemorySize = 0;

    // fix launch dimensions
    grid_dim.x = ( vector_size + TKP::SAMPLES_PER_BLOCK - 1 ) / TKP::SAMPLES_PER_BLOCK;
    block_dim.x = TKP::SAMPLES_PER_BLOCK * WARP_SIZE;

    // run kernel and measure time
    timer.start( stream );
    kernel_apsm_detect<version_id> <<<grid_dim, block_dim, sharedMemorySize, stream>>>( trainingState.toDevice(), d_apsm_rxd2r.toDevice(), d_apsm_esd2r.toDevice(), par );
    timer.stop( stream );

    // sync after kernel call
    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    // give back the measured kernel processing time
    return timer.elapsed();
}

// explicit instantiation of all known versions of the wrapper function, so they are present in the library
template float ApsmWrapper::wrapperDetect<apsm_versions::APSM_DETECT_ORIGINAL>( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par );
template float ApsmWrapper::wrapperDetect<apsm_versions::APSM_DETECT_CG>( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par );
template float ApsmWrapper::wrapperDetect<apsm_versions::APSM_DETECT_SPB>( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par );
template float ApsmWrapper::wrapperDetect<apsm_versions::APSM_DETECT_SPLIT>( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par );
template float ApsmWrapper::wrapperDetect<apsm_versions::APSM_DETECT_SHMEM>( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par );
template float ApsmWrapper::wrapperDetect<apsm_versions::APSM_DETECT_BALANCED>( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par );
 
