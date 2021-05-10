/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
 */

#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

#include <libpng/png.h>

using pixel_type = char;

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPU error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
        if (abort)
        {
            exit(code);
        }
    }
}

// Simple kernel where each thread takes car of a single pixel
__global__ void greyscale(int width, int height, pixel_type* buffer)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned x = idx % width;
    unsigned y = idx / width;
    if (x < width && y < height)
    {
        pixel_type* px = &buffer[(y * width * 4) + (x * 4)];
        pixel_type grey = .299f * (float)px[0] + .587f * (float)px[1] + .114f * (float)px[2];
        px[0] = grey; // Red
        px[1] = grey; // Green
        px[2] = grey; // Blue
        // px[3] Alpha
    }
}

int main()
{
    // Force CUDA initialization
    cudaFree(nullptr);

    // Load image metadata from disk
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_file(&image, "/home/nvidia/eclipse-workspace/cuda-greyscale/colors.png"))
    {
        std::cerr << "Failed to open image\n";
        return 1;
    }
    std::cout << "Image size: " << image.width << " * " << image.height << "\n";

    // Load image data from disk
    image.format = PNG_FORMAT_RGBA;
    pixel_type* buffer = (pixel_type*) malloc(PNG_IMAGE_SIZE(image));
    if (!png_image_finish_read(&image, NULL/*background*/, buffer, 0/*row_stride*/, NULL/*colormap*/))
    {
        std::cerr << "Failed to read image\n";
        return 1;
    }

    // Init GPU buffer and copy image
    pixel_type* d_image;
    gpuErrchk(cudaMalloc(&d_image, PNG_IMAGE_SIZE(image)));
    gpuErrchk(cudaMemcpy(d_image, buffer, PNG_IMAGE_SIZE(image), cudaMemcpyHostToDevice));

    // Compute grid and blocks sizes
    unsigned maxThreadsPerBlock = 1024;
    dim3 blockSize(std::min(maxThreadsPerBlock, image.width * image.height), 1, 1);
    dim3 gridSize((image.width * image.height + (blockSize.x - 1)) / blockSize.x, 1, 1);
    std::cout << "Grid size: " << gridSize.x << " * " << gridSize.y << " * " << gridSize.z << "\n";
    std::cout << "Block size: " << blockSize.x << " * " << blockSize.y << " * " << blockSize.z << "\n";

    // Run the greyscale kernel
    greyscale<<<gridSize, blockSize>>>(image.width, image.height, buffer);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Copy back the image
    gpuErrchk(cudaMemcpy(buffer, d_image, PNG_IMAGE_SIZE(image), cudaMemcpyDeviceToHost));

    // Save the image on disk
    if (!png_image_write_to_file(&image, "/home/nvidia/eclipse-workspace/cuda-greyscale/colors-greyscale.png", 0/*convert_to_8bit*/, buffer, 0/*row_stride*/, NULL/*colormap*/))
    {
        std::cerr << "Failed to save image\n";
        return 1;
    }

    return 0;
}
