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
 * @file cuda_types.cuh
 * @brief CUDA data types header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.08.05   0.01    initial version
 */

#pragma once

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

// THRUST LIB
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// APSM helper
#include "apsm_types.cuh"

/** Real sample */
typedef apsm_fp ThrustRealSample;

/** Host Vector for real samples or complex to real mapped samples */
typedef thrust::host_vector<ThrustRealSample> ThrustRealSampleHostVector;
/** Host Matrix for real samples or complex to real mapped samples */
typedef thrust::host_vector<ThrustRealSampleHostVector> ThrustRealSampleHostMatrix;

/** Device Vector for real samples or complex to real mapped samples */
typedef thrust::device_vector<ThrustRealSample> ThrustRealSampleDeviceVector;
/** Device Matrix for real samples or complex to real mapped samples */
typedef thrust::device_vector<ThrustRealSampleDeviceVector> ThrustRealSampleDeviceMatrix;

/** Complex sample */
typedef thrust::complex<ThrustRealSample> ThrustComplexSample;

/** Host Vector for complex samples */
typedef thrust::host_vector<ThrustComplexSample> ThrustComplexSampleHostVector;
/** Host Matrix for complex samples */
typedef thrust::host_vector<ThrustComplexSampleHostVector> ThrustComplexSampleHostMatrix;

/** Device Vector for complex samples */
typedef thrust::device_vector<ThrustComplexSample> ThrustComplexSampleDeviceVector;
/** Device Matrix for complex samples */
typedef thrust::device_vector<ThrustComplexSampleDeviceVector> ThrustComplexSampleDeviceMatrix;

/**
 * @}
 */
