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
 * @file apsm_train.cu
 * @brief APSM train kernel
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.12.10   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

// CUDA LIB
#include <cooperative_groups.h>

// CUDA helper
#include "cuda_errorhandling.cuh"
#include "cuda_eventtimer.cuh"
#include "cuda_indexing.cuh"

#include "apsm_wrapper.cuh"

// APSM train include
#include "apsm_train.cuh"

//#define INPUT_SHAREDMEM

namespace cg = cooperative_groups;

/**
 * @brief C++ wrapper function for APSM train part.
 * @details This function call the APSM CUDA train kernel.
 * 
 * @param[out] d_apsm_basis basis matrix (learned dictionary)
 * @param[out] d_apsm_coeff coefficient vector (learned dictionary)
 * @param[in] d_apsm_rxd2r received rx constallation vector
 * @param[in] d_apsm_txd1r transmitted tx constallation vector (pilots)
 * @param[in] par APSM parameter set
 *
 * @return measured kernel processing time 
 */
float ApsmWrapper::wrapperTrain( HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, const CudaHostMatrix& d_apsm_txd1r, const apsm_parameters& par )
{
    assert( par.windowSize <= par.dictionarySize ); // window can't be larger than dictionary
    assert( par.trainPasses == 1 || par.dictionarySize * 2 == d_apsm_rxd2r.getWidth() ); // dictionary sparsification + multiple train passes doesn't make sense

    // Initialise timer
    CUDA_EventTimer timer;

    trainingState.basis.pushRowsFromMatrix( stream, d_apsm_rxd2r );

    unsigned int vector_size = 2 * par.windowSize; //max_gaussian_length;
    unsigned int max_linear_length = d_apsm_rxd2r.getHeight();
    assert( max_linear_length * vector_size <= bufferSize );

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, 2 );

    timer.start( stream );

    for ( int i = 0; i < par.trainPasses; i++ )
    {
        for ( unsigned int sample_idx = par.sampleStep; sample_idx <= d_apsm_rxd2r.getWidth(); sample_idx += par.sampleStep )
        {
            if ( i == 0 )
            {
                trainingState.basis.moveWindow();
                trainingState.gaussianCoeffs.moveWindow();
            }

            unsigned int windowSize = min( 2 * par.windowSize, sample_idx );
            grid_dim.x = windowSize;

            // calculate shared memory size parameter
            unsigned int sharedMemorySize = 0;
            sharedMemorySize += APSM_BLOCK_BATCH_SIZE * sizeof( apsm_fp );

            DeviceTrainingState trainingState_device = trainingState.toDevice();
            CudaDeviceDedupMatrix d_apsm_rxd2r_device = d_apsm_rxd2r.toDevice();
            CudaDeviceMatrix d_apsm_txd1r_device = d_apsm_txd1r.toDevice();

            void* kernelArgs[] = {
                (void*)&trainingState_device,
                (void*)&d_apsm_rxd2r_device,
                (void*)&d_apsm_txd1r_device,
                (void*)&par,
                (void*)&deviceBuffer,
                (void*)&sample_idx,
            };

            CUDA_CHECK( cudaLaunchCooperativeKernel( (void*)kernel_apsm_train,
                                                     grid_dim, block_dim, kernelArgs,
                                                     sharedMemorySize, stream ) );

            if ( par.normConstraint > 0 )
            {
                wrapperAdaptCoeffs( trainingState, par );
            }
        }
    }

    // stop timer
    timer.stop( stream );

    // sync after kernel call
    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    return timer.elapsed();
}

void ApsmWrapper::wrapperAdaptCoeffs( HostTrainingState& trainingState, const apsm_parameters& par )
{
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, trainingState.basis.getUsedHeight() );

    unsigned int sharedMemorySize = trainingState.basis.getUsedHeight() * sizeof( apsm_fp );

    apsm_fp* reduction_memory;
    CUDA_CHECK( cudaMalloc( &reduction_memory, sizeof( apsm_fp ) * trainingState.basis.getUsedHeight() ) );

    DeviceTrainingState trainingState_device = trainingState.toDevice();

    void* kernelArgs[] = {
        (void*)&trainingState_device, (void*)&par, (void*)&reduction_memory
    };

    CUDA_CHECK( cudaLaunchCooperativeKernel( (void*)kernel_adapt_coeffs,
                                             grid_dim, block_dim, kernelArgs,
                                             sharedMemorySize, stream ) );
}

__global__ void kernel_adapt_coeffs( DeviceTrainingState trainingState, const apsm_parameters par, apsm_fp* reduction_memory )
{
    cg::grid_group grid = cg::this_grid();

    const unsigned int blockId = getBlockIdx_1D();
    const unsigned int blockThreadId = getBlockThreadIdx_1D();

    const unsigned int batch_size = getBlockDim_1D();

    extern __shared__ apsm_fp shared[];
    shared[ blockThreadId ] = 0;

    for ( unsigned int basis_idx = blockThreadId; basis_idx < trainingState.basis.getUsedHeight(); basis_idx += batch_size )
    {
        apsm_fp* firstBasisVector = trainingState.basis.getRawPointer( blockId );
        apsm_fp* secondBasisVector = trainingState.basis.getRawPointer( basis_idx );
        apsm_fp firstCoeff = trainingState.gaussianCoeffs( blockId );
        apsm_fp secondCoeff = trainingState.gaussianCoeffs( basis_idx );

        shared[ blockThreadId ] += par.gaussianKernelWeight * gaussian_kernel( trainingState.basis.getWidth(), trainingState.basis, blockId, secondBasisVector, par.gaussianKernelVariance )
                                    * firstCoeff * secondCoeff;
    }

    __syncthreads();

    if ( blockThreadId == 0 )
    {
        apsm_fp result = 0;
        for ( unsigned int i = 0; i < min( trainingState.basis.getUsedHeight(), batch_size ); i++ )
        {
            result += shared[ i ];
        }
        reduction_memory[ blockId ] = result;
    }

    cg::sync( grid ); // Sync whole grid

    if ( blockId == 0 && blockThreadId == 0 )
    {
        apsm_fp result = 0;
        for ( unsigned int i = 0; i < trainingState.basis.getUsedHeight(); i++ )
        {
            result += reduction_memory[ i ];
        }
        if ( result > 0 )
            result = sqrt( result );
        reduction_memory[ 0 ] = result;
        // printf("Correction result = %f\n", reduction_memory[0]);
    }

    cg::sync( grid ); // Sync whole grid

    if ( blockThreadId == 0 )
    {
        if ( reduction_memory[ 0 ] > par.normConstraint )
        {
            apsm_fp correction_factor = par.normConstraint / reduction_memory[ 0 ];

            trainingState.gaussianCoeffs( blockId, 0 ) *= correction_factor;
        }
    }
}

/**
 * @brief CUDA train kernel
 * @details 
 * 
 * @param[out] basis 
 * @param[out] coeff 
 * @param[in] rx_data 
 * @param[in] train_data 
 * @param[in] par 
 * @param[in] d_rxdata_input 
 *
 * @return void 
 */
__global__ void kernel_apsm_train( DeviceTrainingState trainingState, const CudaDeviceDedupMatrix rx_data, const CudaDeviceMatrix train_data, const apsm_parameters par, apsm_fp* d_rxdata_input )
{

    // to sync across the whole grid
    //cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    // register for unique indexing
    //const unsigned int threadId	    = getThreadIdx_1D_1D();
    const unsigned int blockId = getBlockIdx_1D();
    const unsigned int blockThreadId = getBlockThreadIdx_1D();

    const unsigned int batch_size = getBlockDim_1D();

    const unsigned int linearLength = rx_data.getHeight();
    //const unsigned int numberOfSamples = rx_data.width;

    const unsigned int gaussianLength = trainingState.basis.getUsedHeight();

    extern __shared__ apsm_fp shared[];

    // Shared memory for detected symbol vector
    apsm_fp* detectedSymbol = &shared[ 0 ]; // detectedSymbol is manually set at the beginning of shared mem
    //	apsm_fp *rxdata_input = &shared[1];	// shared memory for rxinput_data

    // ---------------------------------------------------------------------------

    // If thread index is out of bounds, do nothing.
    if ( blockId >= 2 * par.windowSize )
        return;

        // copy input data to shared memory
#ifdef INPUT_SHAREDMEM
    apsm_fp* rxdata_input_global = d_rxdata_input + ( linearLength * blockId );
    apsm_fp* rxdata_input = &shared[ batch_size ];

    for ( int idx = 0; idx < linearLength; idx += batch_size )
        rxdata_input[ idx + blockThreadId ] = rxdata_input_global[ idx + blockThreadId ];
    __syncthreads();
#else
    apsm_fp* rxdata_input = d_rxdata_input + ( linearLength * blockId );
#endif

#if 0
	if ((blockThreadId == 0) && (blockId==0))
	{
		printf("rxdata_input [size=%d,blockId=%d]: ", linearLength,blockId );
		for(int idx=0;idx<linearLength;idx++)
			printf("%f ", rxdata_input[idx] );
		printf("\n");
	}
	__syncthreads();
#endif

    // ---------------------------------------------------------------------------

    // Next step is to loop over all input samples, regarding to the window size
    // only par.windowsSize threads are running in each round
    {

        // determine how many and which threads are involved to calculate
        // should be 0,2,4,...,40,40,40,....40 if par.windowsSize = 20;
        unsigned int windowSize = min( 2 * par.windowSize, trainingState.basis.getUsedHeight() );

        // thread wait barrier
        //__syncthreads();
        //__threadfence();

        unsigned int threadIdStart = trainingState.basis.getUsedHeight() - windowSize;

        // only let some threads working in this round (windowing)
        if ( blockId < windowSize )
        {
            // call shared detection function
            kernel_apsm_detection( detectedSymbol, linearLength, gaussianLength, rxdata_input,
                                   blockId + threadIdStart, trainingState, rx_data, par );

            // write output to GPU memory  // the last thread
            if ( blockThreadId == 0 )
            {
                apsm_fp symbol = detectedSymbol[ 0 ];
                for ( unsigned int idx = 1; idx < batch_size; idx++ )
                    symbol += detectedSymbol[ idx ];

                detectedSymbol[ 0 ] = symbol;
            }

// write rx_data_input back to global memory
#if 0 //def INPUT_SHAREDMEM
	for(int idx=0;idx<linearLength;idx+=batch_size)
		rxdata_input_global[ idx+blockThreadId ] = rxdata_input[ idx+blockThreadId ]; 
	__syncthreads();
#endif

            // we need it maybe in the future
            //__syncthreads();
            cg::sync( grid ); // Sync whole grid

            // Accumulations registers for projection
            __shared__ apsm_fp WContribConc;

            // compare with detected symbols with transmitted symbols
            //------------------------------------------------------------------------------------
            if ( blockThreadId == 0 )
            {
                // get tx symbol (known pilots during training phase)
                apsm_fp transmittedSymbol = train_data( par.machineLearningUser, blockId + threadIdStart );

                // compute distance between tx and est. rx symbol
                apsm_fp symbol_distance = transmittedSymbol - detectedSymbol[ 0 ];

                // initialize with zero
                WContribConc = 0;

                if ( symbol_distance > +par.eB )
                {
                    WContribConc = symbol_distance - par.eB;
                }
                else if ( symbol_distance < -par.eB )
                {
                    WContribConc = symbol_distance + par.eB;
                }

                // calculate linear and gaussian norms
                // Because distance is zero the gaussian kernel result is always 1.0, so we can use the weight directly -> par.gaussianKernelWeight * e^( 0 )
                apsm_fp linearNorm = apsm_fp( par.linearKernelWeight ) * linear_kernel( linearLength, rxdata_input, rxdata_input );
                apsm_fp gaussianNorm = apsm_fp( par.gaussianKernelWeight ) /* * gaussian_kernel( linearLength, rxdata_input, rxdata_input, par.gaussianKernelVariance ) */;

                // normalization
                WContribConc /= apsm_fp( windowSize );
                WContribConc /= apsm_fp( linearNorm + gaussianNorm );

                //  Extend coefficients
                //------------------------------------------------------------------------------------

                // gaussian part
                trainingState.gaussianCoeffs( blockId + threadIdStart ) += WContribConc;
            }

            // be sure that WContribConc is visible
            __syncthreads();

            // linear part
            {
                const unsigned int batch_size = blockDim.x;

                for ( unsigned int batch_idx = 0; batch_idx < linearLength; batch_idx += batch_size )
                {

                    // read RX vector ...
                    unsigned int dim = blockThreadId + batch_idx;
                    if ( dim < linearLength )
                    // CUDA: for ( unsigned int dim = 0; dim < linearLength; dim++ )
                    {
                        // *coeff.getRawPointer(0, dim) += WContribConc * rxdata_input[ dim ];
                        atomicAdd( trainingState.linearCoeffs.getRawPointer( dim ), WContribConc * rxdata_input[ dim ] ); // because different threads writing to the same memory address
                    }
                }
            }
        }

    } // loop to calculate coeffs
}
