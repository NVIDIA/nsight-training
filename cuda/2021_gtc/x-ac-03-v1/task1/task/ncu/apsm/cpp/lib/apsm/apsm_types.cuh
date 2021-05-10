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
 * @file apsm_types.cuh
 * @brief APSM data types header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.01.06   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 */

#pragma once

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

// STD LIB
#include <complex>
#include <vector>

using namespace std;

/**
 * @brief Precision type
 * @note This can be used to switch the precision from single to double
 */
#if 0
typedef double apsm_fp;   ///< double precision floating point calculation
#else
typedef float apsm_fp; ///< single precision floating point calculation
#endif

/** Real sample */
typedef apsm_fp RealSample;

/** Vector for real samples or complex to real mapped samples */
typedef vector<RealSample> RealSampleVector;
/** Matrix for real samples or complex to real mapped samples */
typedef vector<RealSampleVector> RealSampleMatrix;

/** Complex sample */
typedef complex<RealSample> ComplexSample;

/** Vector for complex samples */
typedef vector<ComplexSample> ComplexSampleVector;
/** Matrix for complex samples */
typedef vector<ComplexSampleVector> ComplexSampleMatrix;

/**
 * @}
 */
