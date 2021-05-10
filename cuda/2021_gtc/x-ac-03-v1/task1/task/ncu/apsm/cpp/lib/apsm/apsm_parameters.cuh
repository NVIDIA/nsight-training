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
 * @file apsm_parameters.cuh
 * @brief APSM parameters header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.10.16   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

#pragma once

#include <iostream>

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

// APSM harddecoder include
#include "apsm_harddecoder.cuh"

/**
 * @brief APSM parameter struct.
 */
typedef struct
{

    // APSM kernel parameters
    apsm_fp linearKernelWeight; ///< Linear Kernel Weight
    apsm_fp gaussianKernelWeight; ///< Gaussian Kernel Weight
    apsm_fp gaussianKernelVariance; ///< Gaussian Kernel Variance

    // training parameters
    unsigned int windowSize; ///< number of past samples to reuse
    unsigned int sampleStep; ///< sample to skip during training
    unsigned int trainPasses = 1; ///< number of training passes over the training data
    unsigned int machineLearningUser; ///< index of user to train
    unsigned int dictionarySize; ///< maximum number of (complex) basis vectors to be saved in the dictionary
    apsm_fp normConstraint; ///< norm constraint to use for dictionary sparsification (set to 0 to disable)
    apsm_fp eB; ///< hyperslab width

    // other parameters
    ApsmModulation modulation = ApsmModulation::off; ///< modulation
    uint32_t antennaPattern = 0xFFFF; ///< antennas that should be used for calculations

} apsm_parameters;

// for debugging of paramset use this (e.g., std::cout << "Parameters = " << par << std::endl;)
ostream& operator<<( ostream& os, const apsm_parameters& par );

/**
 * @}
 */
