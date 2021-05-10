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
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <utility>

#include <cuda_runtime.h>

#include <libpng/png.h>

struct pixel
{
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
};

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPU error: " << cudaGetErrorString(code) << " at " << file << ":" << line << "\n";
        if (abort)
        {
            std::exit(code);
        }
    }
}

constexpr unsigned BlockLength = 31;

__global__ void convolution(const pixel* image, const int width, const int height, const float* matrix, const int matrixLength, pixel* out)
{
    extern __shared__ float sMatrix[];
    pixel* subImage = (pixel*)(sMatrix + (matrixLength * matrixLength));
    const int subImageLength = BlockLength + (matrixLength / 2) * 2;

    const int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    // Copy convolution matrix to shared memory
    if (threadIdx.x < matrixLength && threadIdx.y < matrixLength)
    {
        sMatrix[threadIdx.x + threadIdx.y * matrixLength] = matrix[threadIdx.x + threadIdx.y * matrixLength];
    }

    const int apronSize = matrixLength / 2;

    const int shmemXIdx = matrixLength / 2 + threadIdx.x;
    const int shmemYIdx = matrixLength / 2 + threadIdx.y;

    // Clamping the accessed coordinates to the limits of the image, as we still
    // want threads that would be over that to participate in copying the image
    // chunk to shared memory
    const int cXIdx = min(width - 1, xIdx);
    const int cYIdx = min(height - 1, yIdx);

    // Copy center part of the image chunk in shared memory
    subImage[shmemXIdx + shmemYIdx * subImageLength] = image[cXIdx - cYIdx * width];

    // Copy apron part of the image chunk in shared memory
    if (threadIdx.x < apronSize)
    {
        subImage[shmemXIdx - apronSize + shmemYIdx * subImageLength] = image[max(0, cXIdx - apronSize) + cYIdx * width];
        // Copy a corner of the apron
        if (threadIdx.y < apronSize)
        {
            subImage[shmemXIdx - apronSize + (shmemYIdx - apronSize) * subImageLength] =
                image[max(0, cXIdx - apronSize) + max(0, cYIdx - apronSize) * width];
        }
    }

    if (threadIdx.y < apronSize)
    {
        subImage[shmemXIdx + (shmemYIdx - apronSize) * subImageLength] = image[cXIdx + max(0, cYIdx - apronSize) * width];
        if (threadIdx.x >= blockDim.x - apronSize)
        {
            subImage[shmemXIdx + apronSize + (shmemYIdx - apronSize) * subImageLength] =
                image[min(cXIdx + apronSize, width) + max(0, cYIdx - apronSize) * width];
        }
    }

    if (threadIdx.x >= (blockDim.x - apronSize))
    {
        subImage[shmemXIdx + apronSize + shmemYIdx * subImageLength] = image[min(cXIdx + apronSize, width - 1) + cYIdx * width];
        if (threadIdx.y >= blockDim.y - apronSize)
        {
            subImage[shmemXIdx + apronSize + (shmemYIdx + apronSize) * subImageLength] =
                image[min(cXIdx + apronSize, width - 1) + min(height - 1, cYIdx + apronSize) * width];
        }
    }

    if (threadIdx.y >= (blockDim.y - apronSize))
    {
        subImage[shmemXIdx + (shmemYIdx + apronSize) * subImageLength] = image[cXIdx + min(height - 1, cYIdx + apronSize) * width];
        if (threadIdx.x < apronSize)
        {
            subImage[shmemXIdx - apronSize + (shmemYIdx + apronSize) * subImageLength] =
                image[max(0, cXIdx - apronSize) + min(height - 1, cYIdx + apronSize) * width];
        }
    }

    // At this point we are done copying the sub-image to the shared memory

    // Discard threads that do not participate in the final image
    if (xIdx > width || yIdx >= height)
    {
        return;
    }

    float accR = 0.0;
    float accG = 0.0;
    float accB = 0.0;

    // Apply the convolution
    for (int i = 0; i < matrixLength; i++)
    {
        for (int j = 0; j < matrixLength; j++)
        {
            int baseX = shmemXIdx - apronSize;
            int baseY = shmemYIdx - apronSize;
            accR += (sMatrix[i * matrixLength + j] * subImage[baseX + j + (baseY + i) * subImageLength].red);
            accG += (sMatrix[i * matrixLength + j] * subImage[baseX + j + (baseY + i) * subImageLength].green);
            accB += (sMatrix[i * matrixLength + j] * subImage[baseX + j + (baseY + i) * subImageLength].blue);
        }
    }

    // Clamp values to avoid overflows
    accR = max(0., min(accR, 255.));
    accG = max(0., min(accG, 255.));
    accB = max(0., min(accB, 255.));

    const pixel res = {(uint8_t)accR, (uint8_t)accG, (uint8_t)accB, 255};
    out[xIdx + yIdx * width] = res;
}

static void loadImage(const std::string& filename, png_image& header, std::vector<uint8_t>& image)
{
    std::memset(&header, 0, sizeof(header));
    header.version = PNG_IMAGE_VERSION;

    // Load image metadata from fs
    if (!png_image_begin_read_from_file(&header, filename.c_str()))
    {
        std::cerr << "Failed to open image" << filename << "\n";
        std::exit(1);
    }
    std::cout << "Image "<< filename << " size: " << header.width << " * " << header.height << "\n";

    // Load image data from fs
    header.format = PNG_FORMAT_RGBA;
    image.resize(PNG_IMAGE_SIZE(header));
    if (!png_image_finish_read(&header, NULL/*background*/, image.data(), 0/*row_stride*/, NULL/*colormap*/))
    {
        std::cerr << "Failed to read image " << filename << "\n";
        std::exit(1);
    }
}

static uint8_t* copyImageToDevice(const png_image& header, const std::vector<uint8_t>& image)
{
    uint8_t* dImage;
    gpuErrchk(cudaMalloc(&dImage, PNG_IMAGE_SIZE(header)));
    gpuErrchk(cudaMemcpy(dImage, image.data(), PNG_IMAGE_SIZE(header), cudaMemcpyHostToDevice));
    return dImage;
}

static void runConvolutionKernel(const png_image& header, std::vector<uint8_t>& hImage, uint8_t* dImage, const float* dMatrix, const int matrixLength, uint8_t* dOutput)
{
    dim3 blockSize(BlockLength, BlockLength);
    dim3 gridSize(header.width / blockSize.x + ((header.width % blockSize.x) != 0),
                  header.height / blockSize.y + ((header.height % blockSize.y) != 0));

    int subImageLength = BlockLength + (matrixLength / 2) * 2;
    int sharedMemorySize = (matrixLength * matrixLength * sizeof(float)) + (subImageLength * subImageLength * sizeof(pixel));

    std::cout << "Grid size: " << gridSize.x << " * " << gridSize.y << " * " << gridSize.z << "\n";
    std::cout << "Block size: " << blockSize.x << " * " << blockSize.y << " * " << blockSize.z << "\n";

    convolution<<<gridSize, blockSize, sharedMemorySize>>>((pixel*)dImage, header.width, header.height, dMatrix, matrixLength, (pixel*)dOutput);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(hImage.data(), dOutput, PNG_IMAGE_SIZE(header), cudaMemcpyDeviceToHost));
}


static void saveImage(const std::string& filename, png_image& header, std::vector<uint8_t>& image)
{
    if (!png_image_write_to_file(&header, filename.c_str(), 0/*convert_to_8bit*/, image.data(), 0/*row_stride*/, NULL/*colormap*/))
    {
        std::cerr << "Failed to save image " << filename << "\n";
        std::exit(1);
    }
}

int main(int argc, char* argv[])
{

    std::string imagePath = ".";

    if (argc > 1)
    {
        imagePath = argv[1];
    }

    png_image headerChecker;
    std::vector<uint8_t> imageChecker;
    loadImage(imagePath + "/checkerboard.png", headerChecker, imageChecker);

    uint8_t* dImageChecker = copyImageToDevice(headerChecker, imageChecker);
    uint8_t* dOutputChecker;
    gpuErrchk(cudaMalloc(&dOutputChecker, PNG_IMAGE_SIZE(headerChecker)));

    png_image headerIcon;
    std::vector<uint8_t> imageIcon;
    loadImage(imagePath + "/icon.png", headerIcon, imageIcon);

    uint8_t* dImageIcon = copyImageToDevice(headerIcon, imageIcon);
    uint8_t* dOutputIcon;
    gpuErrchk(cudaMalloc(&dOutputIcon, PNG_IMAGE_SIZE(headerIcon)));

    // Allocate convolution matrix buffer (in this example, we use at most a 5x5 matrix)
    float* dMatrix;
    gpuErrchk(cudaMalloc(&dMatrix, 5 * 5 * sizeof(double)));

    const float edgeDetectionMatrix[9] = {-1.f, -1.f, -1.f,
                                          -1.f, 8.f,  -1.f,
                                          -1.f, -1.f, -1.f};

    gpuErrchk(cudaMemcpy(dMatrix, edgeDetectionMatrix, sizeof(edgeDetectionMatrix), cudaMemcpyHostToDevice));

    runConvolutionKernel(headerChecker, imageChecker, dImageChecker, dMatrix, 3, dOutputChecker);

    gpuErrchk(cudaFree(dImageChecker));

    saveImage(imagePath + "/checkerboard-out.png", headerChecker, imageChecker);

    float identityMatrix[9] = {0.f, };
    identityMatrix[4] = 1.f;
    gpuErrchk(cudaMemcpy(dMatrix, identityMatrix, sizeof(identityMatrix), cudaMemcpyHostToDevice));

    runConvolutionKernel(headerIcon, imageIcon, dImageIcon, dMatrix, 3, dOutputIcon);

    gpuErrchk(cudaFree(dImageIcon));

    saveImage(imagePath + "/icon-out.png", headerIcon, imageIcon);

    // TODO: Uncomment me when reaching step 06
/*
    png_image headerCoffee;
    std::vector<uint8_t> imageCoffee;
    loadImage(imagePath + "/coffee.png", headerCoffee, imageCoffee);
    uint8_t* dImageCoffee = copyImageToDevice(headerCoffee, imageCoffee);

    uint8_t* dOutputCoffee;
    gpuErrchk(cudaMalloc(&dOutputCoffee, PNG_IMAGE_SIZE(headerCoffee)));

    const float gaussianBlurMatrix[25] = {1.f / 256.f,  4.f / 256.f,  6.f / 256.f,  4.f / 256.f, 1.f / 256.f,
                                          4.f / 256.f, 16.f / 256.f, 24.f / 256.f, 16.f / 256.f, 4.f / 256.f,
                                          6.f / 256.f, 24.f / 256.f, 36.f / 256.f, 24.f / 256.f, 6.f / 256.f,
                                          4.f / 256.f, 16.f / 256.f, 24.f / 256.f, 16.f / 256.f, 4.f / 256.f,
                                          1.f / 256.f,  4.f / 256.f,  6.f / 256.f,  4.f / 256.f, 1.f / 256.f};

    gpuErrchk(cudaMemcpy(dMatrix, gaussianBlurMatrix, 25, cudaMemcpyHostToDevice));

    runConvolutionKernel(headerCoffee, imageCoffee, dImageCoffee, dMatrix, 5, dOutputCoffee);

    gpuErrchk(cudaFree(dImageCoffee));

    saveImage(imagePath + "/coffee-out.png", headerCoffee, imageCoffee);
*/

    cudaDeviceReset();

    return 0;
}
