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
 * @file apsm_wrapper.cu
 * @brief APSM chain host c++ wrapper code
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.05.27   0.01    initial version
 */

#include <bitset>

// APSM wrapper include
#include "apsm_wrapper.cuh"
#include "apsm_versions.h"

#include "apsm_matrix.cuh"
#include "cuda_errorhandling.cuh"

ComplexSampleMatrix selectAntennas( const ComplexSampleMatrix& input, const apsm_parameters& par )
{
    const size_t MAX_ANTENNAS = 16;
    std::bitset<MAX_ANTENNAS> antennaPattern( par.antennaPattern );
    const unsigned int numAntennas = min( input.size(), MAX_ANTENNAS );
    ComplexSampleMatrix output;
    output.reserve( antennaPattern.count() );
    for ( size_t i = 0; i < numAntennas; i++ )
    {
        if ( antennaPattern.test( i ) )
            output.push_back( input[ i ] );
    }
    return output;
}

const unsigned int ApsmWrapper::bufferSize = 32 * 1000; // 2 * 16 antennas * 2 * 500 windowSize

ApsmWrapper::ApsmWrapper()
{
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK( cudaMalloc( &deviceBuffer, sizeof( apsm_fp ) * bufferSize ) );
}

ApsmWrapper::~ApsmWrapper()
{
    CUDA_CHECK( cudaStreamDestroy( stream ) );
    CUDA_CHECK( cudaFree( deviceBuffer ) );
}

/**
 * @brief C++ wrapper for APSM train and detection chain kernel
 * 
 * @param[in] rxSigTraining received rx constallation vector
 * @param[in] txSigTraining transmitted tx constallation vector (pilots)
 * @param[in] rxSigData received rx constallation vector
 * @param[out] estSigData equalized or decoded constallation vector 
 * @param[in] par APSM parameter set
 *
 * @return void
 */
void ApsmWrapper::wrapperChain( const ComplexSampleMatrix& rxSigTraining, const ComplexSampleMatrix& txSigTraining, const ComplexSampleMatrix& rxSigData, ComplexSampleMatrix& estSigData, const apsm_parameters& par )
{
    CudaHostDedupMatrix d_apsm_rxd2r( stream, selectAntennas( rxSigTraining, par ) );
    CudaHostMatrix d_apsm_txd1r( stream, txSigTraining );
    CudaHostDedupMatrix d_apsm_ryd2r( stream, selectAntennas( rxSigData, par ) );

    // TEST DETECTION KERNEL HERE
    // INPUT   d_apsm_rxd2r		// rx data - two row vector
    // INPUT   d_apsm_txd1r		// tx data - one row vector
    // OUTPUT  d_apsm_basis		// Basis
    // OUTPUT  d_apsm_coeff		// Coefficents

    unsigned int max_linear_length = d_apsm_rxd2r.getHeight();
    unsigned int max_gaussian_length = d_apsm_rxd2r.getWidth();

    HostTrainingState trainingState( max_linear_length, 2 * par.dictionarySize );

    wrapperTrain( trainingState, d_apsm_rxd2r, d_apsm_txd1r, par );

    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // at this point all prepared data is in the device (time for detection)
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------

    // TEST DETECTION KERNEL HERE
    // INPUT  d_apsm_basis		// Basis
    // INPUT  d_apsm_coeff		// Coefficents
    // INPUT  d_apsm_ryd2r		// rx data  - two row vector
    // OUTPUT d_apsm_esd2r		// est data - two row vector

    CudaHostMatrix d_apsm_esd2r( 1, d_apsm_ryd2r.getWidth() );

    wrapperDetect<APSM_DETECT_VERSION>( trainingState, d_apsm_ryd2r, d_apsm_esd2r, par );

    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // at this point it is time to copy the results back to host
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------

    // prepare result buffer
    // demodulated data (place holder)
    ThrustComplexSampleDeviceVector d_esdat_vec = d_apsm_esd2r.toComplexVector( stream );

    // copy device vector back to std buffer
    estSigData.resize( 1 );
    estSigData[ 0 ].resize( d_esdat_vec.size() );
    thrust::copy( d_esdat_vec.begin(), d_esdat_vec.end(), estSigData[ 0 ].begin() );
}
