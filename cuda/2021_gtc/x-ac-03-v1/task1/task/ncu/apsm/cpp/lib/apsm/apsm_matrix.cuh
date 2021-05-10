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
 * @file apsm_matrix.cuh
 * @brief APSM matrix multiplication header
 *
 * @author Daniel Schäufele, HHI, 
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.10.29   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 * @date 2020.07.17   0.03    refactoring
 */

#pragma once

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

// STD
#include <iostream>

// CUDA helper
#include "cuda_types.cuh"

// forward declaration for friend access
class CudaHostRingBuffer;

class CudaDeviceMatrix
{
public:
    CudaDeviceMatrix( const unsigned int _height, const unsigned int _width, const apsm_fp* _elements )
        : height( _height )
        , width( _width )
        , elements( _elements )
    {
    }

    /**
     * @brief Get the height (number of rows)
     * 
     * @return height
     */
    __device__ unsigned int getHeight() const
    {
        return height;
    }

    /**
     * @brief Get the width (number of columns)
     * 
     * @return width
     */
    __device__ unsigned int getWidth() const
    {
        return width;
    }

    /**
     * @brief Get a raw pointer to an element.
     * 
     * @param row row
     * @param col column
     * @return constant pointer to device memory
     */
    __device__ const apsm_fp* getRawPointer( unsigned int row = 0, unsigned int col = 0 ) const
    {
#ifdef __CUDACC_DEBUG__
        assert( row < height );
        assert( col < width );
#endif
        return elements + row * width + col;
    }

    /**
     * @brief Get a raw pointer to an element.
     * 
     * @param row row
     * @param col column
     * @return pointer to device memory
     */
    __device__
        apsm_fp*
        getRawPointer( unsigned int row = 0, unsigned int col = 0 )
    {
#ifdef __CUDACC_DEBUG__
        assert( row < height );
        assert( col < width );
#endif
        return const_cast<apsm_fp*>( elements + row * width + col );
    }

    /**
     * @brief Return a reference to a single element.
     * 
     * This function exists in different versions ([device, host] x [const, non-const]), that do exactly the same thing.
     * 
     * @param row 
     * @param col 
     * @return reference to element 
     */
    __device__
        apsm_fp&
        operator()( unsigned int row, unsigned int col = 0 )
    {
#ifdef __CUDACC_DEBUG__
        assert( row < height );
        assert( col < width );
#endif
        return const_cast<apsm_fp&>( elements[ row * width + col ] );
    }

    __device__ const apsm_fp& operator()( unsigned int row, unsigned int col = 0 ) const
    {
#ifdef __CUDACC_DEBUG__
        assert( row < height );
        assert( col < width );
#endif
        return elements[ row * width + col ];
    }

protected:
    const unsigned int height;
    const unsigned int width;

    const apsm_fp* const elements;
};

/**
 * @brief Class that saves a matrix in a thrust::device_vector and handles copy-operations to and from the device and complex-real conversions.
 * 
 * Elements can be accessed by using the ()-operator, like this: `mat(i, j) = x;`. The []-operator unfortunately only supports a single argument in C++.
 * 
 * All data is saved on the device.
 */
class CudaHostMatrix
{
public:
    CudaHostMatrix( unsigned int _height, unsigned int _width = 1, apsm_fp fill_value = 0 );
    CudaHostMatrix( cudaStream_t& stream, const ComplexSampleMatrix& data );
    CudaHostMatrix( const RealSampleMatrix& data );

    /**
	 * @brief Get the height (number of rows)
	 * 
	 * @return height
	 */
    unsigned int getHeight() const
    {
        return height;
    }

    /**
	 * @brief Get the width (number of columns)
	 * 
	 * @return width
	 */
    unsigned int getWidth() const
    {
        return width;
    }

    ThrustComplexSampleDeviceVector toComplexVector( cudaStream_t& stream );

    bool operator==( const CudaHostMatrix& other ) const;

    /**
	 * @brief Return a reference to a single element.
	 * 
	 * This function exists in different versions ([device, host] x [const, non-const]), that do exactly the same thing.
	 * 
	 * @param row 
	 * @param col 
	 * @return reference to element 
	 */
    ThrustRealSampleDeviceVector::reference operator()( unsigned int row, unsigned int col = 0 )
    {
#ifdef __CUDACC_DEBUG__
        assert( row < height );
        assert( col < width );
#endif
        return elements[ row * width + col ];
    }

    ThrustRealSampleDeviceVector::const_reference operator()( unsigned int row, unsigned int col = 0 ) const
    {
#ifdef __CUDACC_DEBUG__
        assert( row < height );
        assert( col < width );
#endif
        return elements[ row * width + col ];
    }

    CudaDeviceMatrix toDevice()
    {
        return CudaDeviceMatrix( height, width, thrust::raw_pointer_cast( elements.data() ) );
    }

    const CudaDeviceMatrix toDevice() const
    {
        return CudaDeviceMatrix( height, width, thrust::raw_pointer_cast( elements.data() ) );
    }

    friend ostream& operator<<( ostream& os, const CudaHostMatrix& m );
    friend class CudaHostRingBuffer;

protected:
    const unsigned int height;
    const unsigned int width;

    ThrustRealSampleDeviceVector elements;
};

ostream& operator<<( ostream& os, const CudaHostMatrix& m );

class CudaDeviceDedupMatrix : public CudaDeviceMatrix
{
public:
    using CudaDeviceMatrix::CudaDeviceMatrix; // inherit constructors

    __device__
        apsm_fp
        operator()( unsigned int row, unsigned int col = 0 ) const
    {
        if ( row < height )
        {
            return CudaDeviceMatrix::operator()( row, col );
        }
        else
        {
            if ( col % 2 == 0 )
                return CudaDeviceMatrix::operator()( row - height, col + 1 );
            else
                return -1 * CudaDeviceMatrix::operator()( row - height, col - 1 );
        }
    }

    __device__ unsigned int getHeight() const
    {
        return height * 2;
    }
};

class CudaHostDedupMatrix : public CudaHostMatrix
{
public:
    using CudaHostMatrix::CudaHostMatrix; // inherit constructors

    apsm_fp operator()( unsigned int row, unsigned int col = 0 ) const
    {
        if ( row < height )
        {
            return CudaHostMatrix::operator()( row, col );
        }
        else
        {
            if ( col % 2 == 0 )
                return CudaHostMatrix::operator()( row - height, col + 1 );
            else
                return -1 * CudaHostMatrix::operator()( row - height, col - 1 );
        }
    }

    unsigned int getHeight() const
    {
        return height * 2;
    }

    const CudaDeviceDedupMatrix toDevice() const
    {
        return CudaDeviceDedupMatrix( height, width, thrust::raw_pointer_cast( elements.data() ) );
    }
};

class CudaDeviceRingBuffer : public CudaDeviceMatrix
{
public:
    CudaDeviceRingBuffer( const unsigned int _height, const unsigned int _width, const apsm_fp* _elements, const unsigned int _offset, const unsigned int _usedHeight, const unsigned int _windowHeight )
        : CudaDeviceMatrix( _height, _width, _elements )
        , offset( _offset )
        , usedHeight( _usedHeight )
        , windowHeight( _windowHeight )
    {
    }

    /**
	 * @brief Get the height (number of rows)
	 * 
	 * @return height
	 */
    __device__ unsigned int getUsedHeight() const
    {
        return usedHeight > windowHeight ? windowHeight : usedHeight;
    }

    /**
	 * @brief Get a raw pointer to an element.
	 * 
	 * @param row row
	 * @param col column
	 * @return constant pointer to device memory
	 */
    __device__ const apsm_fp* getRawPointer( unsigned int row = 0, unsigned int col = 0 ) const
    {
#ifdef __CUDACC_DEBUG__
        assert( row < usedHeight );
        assert( col < width );
#endif
        return CudaDeviceMatrix::getRawPointer( ( row + offset ) % height, col );
    }

    /**
	 * @brief Get a raw pointer to an element.
	 * 
	 * @param row row
	 * @param col column
	 * @return pointer to device memory
	 */
    __device__
        apsm_fp*
        getRawPointer( unsigned int row = 0, unsigned int col = 0 )
    {
#ifdef __CUDACC_DEBUG__
        assert( row < usedHeight );
        assert( col < width );
#endif
        return CudaDeviceMatrix::getRawPointer( ( row + offset ) % height, col );
    }

    /**
	 * @brief Return a reference to a single element.
	 * 
	 * This function exists in different versions ([device, host] x [const, non-const]), that do exactly the same thing.
	 * 
	 * @param row 
	 * @param col 
	 * @return reference to element 
	 */
    __device__
        apsm_fp&
        operator()( unsigned int row, unsigned int col = 0 )
    {
#ifdef __CUDACC_DEBUG__
        assert( row < usedHeight );
        assert( col < width );
#endif
        return CudaDeviceMatrix::operator()( ( row + offset ) % height, col );
    }

    __device__ const apsm_fp& operator()( unsigned int row, unsigned int col = 0 ) const
    {
#ifdef __CUDACC_DEBUG__
        assert( row < usedHeight );
        assert( col < width );
#endif
        return CudaDeviceMatrix::operator()( ( row + offset ) % height, col );
    }

protected:
    const unsigned int offset;
    const unsigned int usedHeight;
    const unsigned int windowHeight;
};

class CudaHostRingBuffer : public CudaHostMatrix
{
public:
    CudaHostRingBuffer( unsigned int windowHeight, unsigned int width = 1, unsigned int heightCapacity = 0 )
        : CudaHostMatrix( heightCapacity == 0 ? windowHeight : heightCapacity, width )
        , offset( 0 )
        , usedHeight( 0 )
        , windowHeight( windowHeight )
    {
    }

    CudaHostRingBuffer( const CudaHostMatrix& other, unsigned int heightCapacity = 0 ); // TODO: change to windowHeight

    /**
	 * @brief Get the height (number of rows)
	 * 
	 * @return height
	 */
    unsigned int getUsedHeight() const
    {
        return usedHeight > windowHeight ? windowHeight : usedHeight;
    }

    bool operator==( const CudaHostRingBuffer& other ) const
    {
        return offset == other.offset && usedHeight == other.usedHeight && CudaHostMatrix::operator==( other );
        // TODO: detect shifted versions of the same buffer
    }

    void pushRowFromMatrix( cudaStream_t& stream, const CudaHostMatrix& m, unsigned int col );

    void pushRowsFromMatrix( cudaStream_t& stream, const CudaHostMatrix& m, unsigned int startOffset = 0 );

    void moveWindow( unsigned int count = 2 );

    void pushEmptyRow( cudaStream_t& stream );

    /**
	 * @brief Return a reference to a single element.
	 * 
	 * This function exists in different versions ([device, host] x [const, non-const]), that do exactly the same thing.
	 * 
	 * @param row 
	 * @param col 
	 * @return reference to element 
	 */
    ThrustRealSampleDeviceVector::reference operator()( unsigned int row, unsigned int col = 0 )
    {
#ifdef __CUDACC_DEBUG__
        assert( row < usedHeight );
        assert( row < windowHeight );
        assert( col < width );
#endif
        return CudaHostMatrix::operator()( ( row + offset ) % height, col );
    }

    ThrustRealSampleDeviceVector::const_reference operator()( unsigned int row, unsigned int col = 0 ) const
    {
#ifdef __CUDACC_DEBUG__
        assert( row < usedHeight );
        assert( row < windowHeight );
        assert( col < width );
#endif
        return CudaHostMatrix::operator()( ( row + offset ) % height, col );
    }

    CudaDeviceRingBuffer toDevice()
    {
        return CudaDeviceRingBuffer( height, width, thrust::raw_pointer_cast( elements.data() ), offset, usedHeight, windowHeight );
    }

    const CudaDeviceRingBuffer toDevice() const
    {
        return CudaDeviceRingBuffer( height, width, thrust::raw_pointer_cast( elements.data() ), offset, usedHeight, windowHeight );
    }

    friend ostream& operator<<( ostream& os, const CudaHostRingBuffer& m );

protected:
    unsigned int offset;
    unsigned int usedHeight;
    unsigned int windowHeight;
};

ostream& operator<<( ostream& os, const CudaHostRingBuffer& m );

class CudaDeviceDedupRingBuffer : public CudaDeviceRingBuffer
{
public:
    using CudaDeviceRingBuffer::CudaDeviceRingBuffer; // inherit constructors

    __device__ apsm_fp
    operator()( unsigned int row, unsigned int col = 0 ) const
    {
        if ( col < width )
        {
            return CudaDeviceRingBuffer::operator()( row, col );
        }
        else
        {
            if ( row % 2 == 0 )
                return CudaDeviceRingBuffer::operator()( row + 1, col - width );
            else
                return -1 * CudaDeviceRingBuffer::operator()( row - 1, col - width );
        }
    }

    __device__ unsigned int getWidth() const
    {
        return width * 2;
    }
};

class CudaHostDedupRingBuffer : public CudaHostRingBuffer
{
public:
    using CudaHostRingBuffer::CudaHostRingBuffer; // inherit constructors

    apsm_fp operator()( unsigned int row, unsigned int col = 0 ) const
    {
        if ( col < width )
        {
            return CudaHostRingBuffer::operator()( row, col );
        }
        else
        {
            if ( row % 2 == 0 )
                return CudaHostRingBuffer::operator()( row + 1, col - width );
            else
                return -1 * CudaHostRingBuffer::operator()( row - 1, col - width );
        }
    }

    unsigned int getWidth() const
    {
        return width * 2;
    }

    const CudaDeviceDedupRingBuffer toDevice() const
    {
        return CudaDeviceDedupRingBuffer( height, width, thrust::raw_pointer_cast( elements.data() ), offset, usedHeight, windowHeight );
    }
};

class DeviceTrainingState
{
public:
    DeviceTrainingState( const CudaDeviceDedupRingBuffer& _basis, const CudaDeviceRingBuffer& _gaussianCoeffs, const CudaDeviceMatrix& _linearCoeffs )
        : basis( _basis )
        , gaussianCoeffs( _gaussianCoeffs )
        , linearCoeffs( _linearCoeffs )
    {
    }

    CudaDeviceDedupRingBuffer basis;
    CudaDeviceRingBuffer gaussianCoeffs;
    CudaDeviceMatrix linearCoeffs;
};

class HostTrainingState
{
public:
    HostTrainingState( unsigned int linearLength, unsigned int dictWindowHeight, unsigned int dictCapacity = 0 )
        : basis( dictWindowHeight, linearLength / 2, dictCapacity )
        , gaussianCoeffs( dictWindowHeight, 1, dictCapacity )
        , linearCoeffs( linearLength )
    {
    }

    HostTrainingState( CudaHostDedupRingBuffer& _basis, CudaHostRingBuffer& _gaussianCoeffs, CudaHostMatrix& _linearCoeffs )
        : basis( _basis )
        , gaussianCoeffs( _gaussianCoeffs )
        , linearCoeffs( _linearCoeffs )
    {
    }

    DeviceTrainingState toDevice()
    {
        return DeviceTrainingState( basis.toDevice(), gaussianCoeffs.toDevice(), linearCoeffs.toDevice() );
    }

    const DeviceTrainingState toDevice() const
    {
        return DeviceTrainingState( basis.toDevice(), gaussianCoeffs.toDevice(), linearCoeffs.toDevice() );
    }

    CudaHostDedupRingBuffer basis;
    CudaHostRingBuffer gaussianCoeffs;
    CudaHostMatrix linearCoeffs;
};

/**
 * @}
 */
