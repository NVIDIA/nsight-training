/*
* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: MIT
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <cstdio>
#include <cuda_runtime.h>

// CUDA alignment is set to 16 to support the largest load type that we have =>
// uint4 (4*4 bytes) => 128 bit
#define CUDA_ALIGNMENT 16

inline unsigned divUp(unsigned x, unsigned y) { return (x + y - 1) / y; }

#ifndef NDEBUG
#define CudaCheckError()
#else
#define CudaCheckError()                                                       \
  do {                                                                         \
    cudaError err_ = cudaGetLastError();                                       \
    if (err_ != cudaSuccess) {                                                 \
      printf("CudaCheckError() failed at: %s:%d\n", __FILE__, __LINE__);       \
      printf("code: %d ; description: %s\n", err_, cudaGetErrorString(err_));  \
      exit(1);                                                                 \
    }                                                                          \
                                                                               \
    err_ = cudaDeviceSynchronize();                                            \
    if (cudaSuccess != err_) {                                                 \
      printf("CudaCheckError() failed after sync at: %s:%d;\n", __FILE__,      \
             __LINE__);                                                        \
      printf("code: %d; description: %s\n", err_, cudaGetErrorString(err_));   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

#define CudaCheck(CALL)                                                        \
  do {                                                                         \
    if (cudaSuccess != CALL) {                                                 \
      printf("Cuda error at  %s:%d with error %s(%i)\n", __FILE__, __LINE__,   \
             cudaGetErrorString(CALL), CALL);                                  \
      exit(CALL);                                                              \
    }                                                                          \
  } while (0);
