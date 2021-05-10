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
 * @file apsm_wrapper.cuh
 * @brief APSM chain host c++ wrapper code header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.05.27   0.01    initial version
 */

#pragma once

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

#include <cuda_runtime.h>

// APSM train and detect include
#include "apsm_parameters.cuh"
#include "apsm_types.cuh"

// Forward declare cuda matrix classes (cuda_matrix.cuh can't be included here, because when thrust headers are included from gcc you will get some completely illegible error messages)
class HostTrainingState;
class CudaHostDedupMatrix;
class CudaHostMatrix;

class ApsmWrapper
{
public:
    ApsmWrapper();
    ~ApsmWrapper();
    void wrapperChain( const ComplexSampleMatrix&, const ComplexSampleMatrix&, const ComplexSampleMatrix&, ComplexSampleMatrix&, const apsm_parameters& );
    float wrapperTrain( HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, const CudaHostMatrix& d_apsm_txd1r, const apsm_parameters& par );
    template< int version_id > float wrapperDetect( const HostTrainingState& trainingState, const CudaHostDedupMatrix& d_apsm_rxd2r, CudaHostMatrix& d_apsm_esd2r, const apsm_parameters& par );
    void wrapperAdaptCoeffs( HostTrainingState& trainingState, const apsm_parameters& par );

private:
    cudaStream_t stream;
    apsm_fp* deviceBuffer;

    static const unsigned int bufferSize;
};

/**
 * @}
 */
