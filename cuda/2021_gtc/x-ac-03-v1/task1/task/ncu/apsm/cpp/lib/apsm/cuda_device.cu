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
 * @file cuda_device.cu
 * @brief CUDA device capabilities
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.02.05   0.01    initial version
 *
 * @note detecting NVIDIA GPUs without CUDA
 *       https://stackoverflow.com/questions/12828468/detecting-nvidia-gpus-without-cuda
 */

// CUDA LIB
#include <cuda_runtime.h>

// CUDA helper
#include "cuda_device.cuh"

/**
 * @brief CUDA device capabilities detection
 * 
 * @return cuda_device 
 */
cuda_device cuda_device_getCapabilities()
{
    cuda_device gpu;
    gpu.QueryFailed = false;
    gpu.StrongestDeviceId = -1;
    gpu.ComputeCapabilityMajor = -1;
    gpu.ComputeCapabilityMinor = -1;

    cudaError_t error_id = cudaGetDeviceCount( &gpu.DeviceCount );
    if ( error_id != cudaSuccess )
    {
        gpu.QueryFailed = true;
        gpu.DeviceCount = 0;
        return gpu;
    }

    if ( gpu.DeviceCount == 0 )
        return gpu; // "There are no available device(s) that support CUDA

    // Find best device
    for ( int dev = 0; dev < gpu.DeviceCount; ++dev )
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties( &deviceProp, dev );
        if ( deviceProp.major > gpu.ComputeCapabilityMajor )
        {
            gpu.ComputeCapabilityMajor = dev;
            gpu.ComputeCapabilityMajor = deviceProp.major;
            gpu.ComputeCapabilityMinor = 0;
        }
        if ( deviceProp.minor > gpu.ComputeCapabilityMinor )
        {
            gpu.ComputeCapabilityMajor = dev;
            gpu.ComputeCapabilityMinor = deviceProp.minor;
        }
    }
    return gpu;
}