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


#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>

#include "cuda_helper.cuh"
#include "lodepng.h"

__constant__ __device__ float rgb_coeff[4] = {0.2126f, 0.7152f, 0.0722f, 0.0f};

__global__ void rgba2grayscale(const uint8_t *__restrict__ input,
                               uint8_t *__restrict__ output, unsigned h, unsigned w,
                               unsigned in_pitch, unsigned out_pitch) {
    const unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_y < h && idx_x < w) {
        constexpr int channel = 4;
        unsigned idx = idx_y * in_pitch + idx_x * channel;
        float gray = rgb_coeff[0] * input[idx] + rgb_coeff[1] * input[idx + 1] +
                     rgb_coeff[2] * input[idx + 2] + rgb_coeff[3] * input[idx + 3];
        output[idx_y * out_pitch + idx_x] = static_cast<uint8_t>(gray);
    }
}

template<int values_per_thread>
__global__ void rgba2grayscale_multiple_values(const uint8_t *__restrict__ input,
                                               uint8_t *__restrict__ output,
                                               unsigned h, [[maybe_unused]] unsigned w,
                                               unsigned in_pitch,
                                               unsigned out_pitch) {
    const unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * values_per_thread;

    constexpr unsigned channel = 4;
    unsigned idx = idx_y * in_pitch + idx_x * channel;
    unsigned idx_out = idx_y * out_pitch + idx_x;
    if (idx_y < h) {
        for (unsigned i = 0; i < values_per_thread; idx += channel, ++i) {
            if ((idx_x + i) < out_pitch) {
                float gray = rgb_coeff[0] * input[idx] +
                             rgb_coeff[1] * input[idx + 1] +
                             rgb_coeff[2] * input[idx + 2] +
                             rgb_coeff[3] * input[idx + 3];

                output[idx_out + i] = static_cast<uint8_t>(gray);
            }
        }
    }
}

template<typename T, int values_per_thread, typename T_out>
__global__ void rgba2grayscale_wide(const T *__restrict__ input,
                                    T_out *__restrict__ output, unsigned h, [[maybe_unused]] unsigned w,
                                    unsigned in_pitch, unsigned out_pitch) {
    static_assert(values_per_thread % sizeof(T_out) == 0);
    constexpr unsigned channel = 4;
    static_assert((sizeof(T_out) * channel) == sizeof(T));

    const unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned idx_x_T = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr unsigned values_per_store = sizeof(T_out);
    constexpr unsigned values_per_load = sizeof(T);
    constexpr unsigned total_memory_ops = values_per_thread / values_per_store;
    const unsigned stride_T = in_pitch / values_per_load;

    if (idx_y < h) {
        for (int mem_op = 0; mem_op < total_memory_ops; ++mem_op) {
            if ((idx_x_T + mem_op) < stride_T) {
                union {
                    T in_load;
                    uint8_t in_u8[values_per_load];
                };
                union {
                    T_out out_store;
                    uint8_t out_u8[values_per_store];
                };

                // we load at least one pixel (uchar4) or even more in a single load
                // operation therefore the index calculation

                in_load = input[idx_y * stride_T + idx_x_T + mem_op];

                for (int i = 0; i < values_per_store; ++i) {
                    float gray =
                            rgb_coeff[0] * in_u8[i * 4] + rgb_coeff[1] * in_u8[i * 4 + 1] +
                            rgb_coeff[2] * in_u8[i * 4 + 2] + rgb_coeff[3] * in_u8[i * 4 + 3];
                    out_u8[i] = static_cast<uint8_t>(gray);
                }
                const int stride_T_out = out_pitch / values_per_store;
                const int idx_out = idx_y * stride_T_out + idx_x_T + mem_op;
                output[idx_out] = out_store;
            }
        }
    }
}

enum Optimization {
    UNALIGNED, ALIGNED, MULTIPLE_LOADS, WIDE_LOADS,
};

std::string optimization_to_string(Optimization o) {
    switch (o) {
        case Optimization::UNALIGNED:
            return "UNALIGNED";
        case Optimization::ALIGNED:
            return "ALIGNED";
        case Optimization::WIDE_LOADS:
            return "WIDE_LOADS";
        case Optimization::MULTIPLE_LOADS:
            return "MULTIPLE_LOADS";
        default:
            throw std::invalid_argument("Unimplemented item");
    }
}

int main(int argc, char **argv) {
    std::string filename = "./images/input.png";
    // load png
    unsigned h, w;
    constexpr unsigned channel = 4;
    std::vector<uint8_t> image;
    unsigned error =
            lodepng::decode(image, w, h, filename, LodePNGColorType::LCT_RGBA);
    if (error) {
        std::cout << "[lodepng] decoder error " << error << ": "
                  << lodepng_error_text(error) << std::endl;
        return -1;
    }

    unsigned pitch_device_rgb =
            divUp(w * channel, CUDA_ALIGNMENT) * CUDA_ALIGNMENT;
    unsigned pitch_device_gray = divUp(w, CUDA_ALIGNMENT) * CUDA_ALIGNMENT;
    uint8_t *input_device, *output_device;
    // prepare cuda buffers
    CudaCheck(
            cudaMalloc(&input_device, h * pitch_device_rgb)) // 3840*2160*4 ~ 32MB
    CudaCheck(cudaMemcpy2D(input_device, pitch_device_rgb, image.data(),
                           w * channel, w * channel, h, cudaMemcpyHostToDevice))
    CudaCheck(
            cudaMalloc(&output_device, h * pitch_device_gray)) // 3840*2160*1 ~ 8MB

    cudaEvent_t start, stop;
    cudaStream_t stream;
    size_t shm = 0; // required shared memory
    CudaCheck(cudaEventCreate(&start))
    CudaCheck(cudaStreamCreate(&stream))
    CudaCheck(cudaEventCreate(&stop))
    // execute kernel with different optimization "levels"

#ifdef BENCHMARK
    for (auto o: {UNALIGNED, ALIGNED, MULTIPLE_LOADS, WIDE_LOADS})
#else
    if (argc < 2)
    {
        std::cerr << "Please specify one of the optimization modes: \n";
        for (auto o: {UNALIGNED, ALIGNED, MULTIPLE_LOADS, WIDE_LOADS})
        {
            std::cerr << "\tMode "<< (int)o << ":" << optimization_to_string(o) << "\n";
        }
        std::cerr << std::endl;
        return -1;
    }
    const Optimization o = Optimization(std::stoi(argv[1]));
#endif
    {
        CudaCheck(cudaEventRecord(start, stream))
#ifdef BENCHMARK
        constexpr int number_of_benchmark_iteraions = 100;
        for (int i = 0; i < number_of_benchmark_iteraions; ++i)
#endif
        {
            switch (o) {
                case UNALIGNED: {
                    // this is exactly one warp per scheduler, one SM has 4 schedulers with
                    // each 32 threads see:
                    // https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
                    // -> Figure 5
                    dim3 block(1, 256);
                    dim3 grid(divUp(w, block.x), divUp(h, block.y));
                    rgba2grayscale<<<grid, block, shm, stream>>>(
                            input_device, output_device, h, w, pitch_device_rgb,
                            pitch_device_gray);
                    break;
                }
                case ALIGNED: {
                    dim3 block(256);
                    dim3 grid(divUp(w, block.x), divUp(h, block.y));
                    rgba2grayscale<<<grid, block, shm, stream>>>(
                            input_device, output_device, h, w, pitch_device_rgb,
                            pitch_device_gray);
                    break;
                }
                case MULTIPLE_LOADS: {
                    // How many grayscale pixels are produced per thread
                    constexpr int values_per_thread = 4;
#ifndef BENCHMARK
                    std::cout << "processing " << values_per_thread << " values per thread"
                              << std::endl;
#endif
                    // 32 * 4 bytes are exactly one cache line
                    dim3 block(32, 8);
                    dim3 grid(divUp(w, block.x * values_per_thread), divUp(h, block.y));
                    rgba2grayscale_multiple_values<values_per_thread>
                    <<<grid, block, shm, stream>>>(input_device, output_device, h, w,
                                                   pitch_device_rgb, pitch_device_gray);
                    break;
                }
                case WIDE_LOADS: {
                    // going from uchar4 to ushort4/uint4 here will lead to a good
                    // performance improvement
                    using load_type = uint4;
                    // we have to respect our load data type it can lead to processing more
                    // than 1 value ideally we would know this value per thread and use it
                    // as template arg to unroll loops inside the kernel
                    constexpr unsigned values_per_thread = (sizeof(load_type) / channel);
                    // TODO: feel free to play around with the values per thread and see if the kernel gets even faster !
                    //    constexpr int values_per_thread = (sizeof(load_type) / channel) * 4;
#ifndef BENCHMARK
                    std::cout << "processing " << values_per_thread << " values per thread"
                              << std::endl;
#endif
                    using store_type = uchar4;
                    static_assert(sizeof(store_type) == sizeof(load_type) / channel);
                    static_assert((sizeof(load_type) * channel) % values_per_thread == 0);

                    // this is block has exactly two warps per scheduler to hide memory
                    // latency within each warp see:
                    // https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
                    // -> Figure 4 we still want to have enough blocks to fill all SMs
                    // otherwise we will not achieve full bandwidth
                    dim3 block(32, 8);
                    dim3 grid(divUp(w, block.x * values_per_thread), divUp(h, block.y));
                    rgba2grayscale_wide<load_type, values_per_thread, store_type>
                    <<<grid, block, shm, stream>>>(
                            reinterpret_cast<load_type *>(input_device),
                            reinterpret_cast<store_type *>(output_device), h, w,
                            pitch_device_rgb, pitch_device_gray);
                    break;
                }
                default:
                    std::cerr << "Not implemented" << std::endl;
                    return -1;
            }
        }

        CudaCheck(cudaEventRecord(stop, stream))
        CudaCheck(cudaEventSynchronize(stop))
        CudaCheckError();
        float elapsedTime;
        CudaCheck(cudaEventElapsedTime(&elapsedTime, start, stop))

        // This timing is relatively unreliable if we only run the kernel once
        // in Nsight compute we will get stable timing as it records the kernel and
        // reruns it to sample another alternative is to use the benchmark executable
        printf("Elapsed time for optimization level %s: %f ms\n", optimization_to_string(o).c_str(), elapsedTime);
    }

    // download cuda buffers
    std::vector<uint8_t> out_image;
    out_image.resize(h * w);
    CudaCheck(cudaMemcpy2D(out_image.data(), w, output_device, pitch_device_gray,
                           w, h, cudaMemcpyDeviceToHost))

    // Encode the image
    error = lodepng::encode("./output.png", out_image, w, h,
                            LodePNGColorType::LCT_GREY);
    if (error)
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
    return 0;
}
