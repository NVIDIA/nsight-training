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
 * @file apsm_matrix.cu
 * @brief APSM matrix multiplication
 *
 * @author Daniel Schäufele, HHI, 
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 * 
 * @date 2019.10.29   0.01    initial version
 * @date 2020.01.17   0.02    APSM (dictionary no sparsification)
 * @date 2020.07.17   0.03    refactoring
 *
 * @note GITHUB THRUST - A Parallel Algorithms Library 
 *       https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
 */

// THRUST LIB
#include <thrust/copy.h>

// APSM helper
#include "apsm_matrix.cuh"

// CUDA helper
#include "cuda_indexing.cuh"

/**
 * @brief helper class that iterates over every n-th element
 * 
 * @tparam Iterator 
 */
template <typename Iterator>
class strided_range
{
public:
    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type, difference_type>
    {
        difference_type stride;

        stride_functor( difference_type stride )
            : stride( stride )
        {
        }

        __host__ __device__
            difference_type
            operator()( const difference_type& i ) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type> CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range( Iterator first, Iterator last, difference_type stride )
        : first( first )
        , last( last )
        , stride( stride )
    {
    }

    iterator begin( void ) const
    {
        return PermutationIterator( first, TransformIterator( CountingIterator( 0 ), stride_functor( stride ) ) );
    }

    iterator end( void ) const
    {
        return begin() + ( ( last - first ) + ( stride - 1 ) ) / stride;
    }

protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

/**
 * @brief Functor that extracts the real part of a complex input
 */
struct functor_real
{

    /**
     * @brief extracts positive real part from a complex value
     * 
     * @param[in] in complex value
     * @return real part 
     */
    __host__ __device__
        apsm_fp
        operator()( ThrustComplexSample& in )
    {
        return in.real();
    }
};

/**
 * @brief Functor that extracts the real part of a complex input and multiplies it by -1
 */
struct functor_neg_real
{
    // TODO (mm) remove this functor to avoid multiple read of the same value
    /**
     * @brief extracts negative real part from a complex value
     * 
     * @param[in] in complex value
     * @return negative real part 
     */
    __host__ __device__
        apsm_fp
        operator()( ThrustComplexSample& in )
    {
        return -in.real();
    }
};

/**
 * @brief Functor that extracts the imaginary part of a complex input
 */
struct functor_imag
{

    /**
     * @brief extracts positive imaginary part from a complex value
     * 
     * @param[in] in complex value
     * @return imaginary part
     */
    __host__ __device__
        apsm_fp
        operator()( ThrustComplexSample& in )
    {
        return in.imag();
    }
};

/**
 * @brief Functor that constructs complex objects out of real and imaginary input
 */
struct functor_complex
{

    /**
     * @brief combine real and imaginary part to a complex value
     * 
     * @param[in] re value representing the real part
     * @param[in] im value representing the imaginary part
     * @return comlex value
     */
    __host__ __device__
        ThrustComplexSample
        operator()( apsm_fp re, apsm_fp im )
    {
        return ThrustComplexSample( re, im );
    }
};

/**
 * @brief Construct a new CudaMatrix with given dimensions, that is filled with the given value (0 by default).
 * 
 * @param _height Number of rows
 * @param _width Number of columns
 * @param fill_value Value that will be used to fill the matrix
 */
CudaHostMatrix::CudaHostMatrix( unsigned int _height, unsigned int _width, apsm_fp fill_value )
    : height( _height )
    , width( _width )
    , elements( width * height, fill_value )
{
}

/**
 * @brief Construct a new CudaMatrix with dimensions and entries given by a ComplexSampleMatrix.
 * 
 * Depending on the complexConversion parameter, the complex values are mapped to the real values in a different way. As an example, the complex matrix
 *     | A B |
 *     | C D |
 * - with the parameter `R_I` will be mapped to
 *     | Re(A) Re(B) |
 *     | Re(C) Re(D) |
 *     | Im(A) Im(B) |
 *     | Im(C) Im(D) |
 * - with the parameter `RI_ImR` will be mapped to
 *     | Re(A)  Im(A) Re(B)  Im(B) |
 *     | Re(C)  Im(C) Re(D)  Im(D) |
 *     | Im(A) -Re(A) Im(B) -Re(B) |
 *     | Im(C) -Re(C) Im(D) -Re(D) |
 * - with the parameter `RI` will be mapped to
 *     | Re(A) Im(A) Re(B) Im(B) |
 *     | Re(C) Im(C) Re(D) Im(D) |
 * 
 * @param data Complex data that will be used to fill the matrix
 * @param complexConversion Pattern that will be used to map complex values to real values
 */
CudaHostMatrix::CudaHostMatrix( cudaStream_t& stream, const ComplexSampleMatrix& data )
    : height( data.size() )
    , width( data[ 0 ].size() * 2 )
    , elements( width * height )
{
    unsigned int data_height = data.size();
    unsigned int data_width = data[ 0 ].size();
    ThrustComplexSampleDeviceVector d_data( data_width * data_height );

    for ( size_t idx = 0; idx < data_height; idx++ )
    {
        thrust::copy( /*thrust::cuda::par.on( s1 ),*/ data[ idx ].begin(), data[ idx ].end(), d_data.begin() + ( idx * data_width ) );
    }

    typedef ThrustRealSampleDeviceVector::iterator Iterator;

    strided_range<Iterator> evens( elements.begin(), elements.end(), 2 );
    thrust::transform( thrust::cuda::par.on( stream ), d_data.begin(), d_data.end(), evens.begin(), functor_real() );

    strided_range<Iterator> odds( elements.begin() + 1, elements.end(), 2 );
    thrust::transform( thrust::cuda::par.on( stream ), d_data.begin(), d_data.end(), odds.begin(), functor_imag() );
}

/**
 * @brief Construct a new CudaMatrix with dimension and entries given by a RealSampleMatrix.
 * 
 * @param data Real data that will be used to fill the matrix
 */
CudaHostMatrix::CudaHostMatrix( const RealSampleMatrix& data )
    : height( data.size() )
    , width( data[ 0 ].size() )
    , elements( width * height )
{
    for ( size_t idx = 0; idx < height; idx++ )
    {
        thrust::copy( /*thrust::cuda::par.on( s1 ),*/ data[ idx ].begin(), data[ idx ].end(), elements.begin() + ( idx * width ) );
    }
}

/**
 * @brief Returns a vector of complex values from a CudaMatrix given in `RI` format (see constructor above for definitions).
 * 
 * @return Complex representation of CudaMatrix
 */
ThrustComplexSampleDeviceVector CudaHostMatrix::toComplexVector( cudaStream_t& stream )
{
    unsigned int num_complex_values = width * height / 2;
    ThrustComplexSampleDeviceVector out( num_complex_values );

    typedef ThrustRealSampleDeviceVector::iterator Iterator;
    strided_range<Iterator> evens( elements.begin(), elements.end(), 2 );
    strided_range<Iterator> odds( elements.begin() + 1, elements.end(), 2 );
    thrust::transform( thrust::cuda::par.on( stream ), evens.begin(), evens.end(), odds.begin(), out.begin(), functor_complex() );

    return out;
}

/**
 * @brief Compares CudaMatrix to another CudaMatrix.
 * 
 * @param other CudaMatrix to compare to
 * @return true if dimensions and all entries are equal
 * @return false otherwise
 */
bool CudaHostMatrix::operator==( const CudaHostMatrix& other ) const
{
    // create a CUDA stream
    cudaStream_t s1;
    cudaStreamCreate( &s1 );

    bool result;
    if ( height != other.height || width != other.width )
        return false;
    result = thrust::equal( thrust::cuda::par.on( s1 ), elements.begin(), elements.end(), other.elements.begin() );

    // destroy streams
    cudaStreamDestroy( s1 );

    return result;
}

/**
 * @brief Prints a human readable representation of CudaMatrix to a stream.
 * 
 * @param os Stream to print to
 * @param m CudaMatrix that should be printed
 * @return ostream& Stream given as input (for operator chaining)
 */
ostream& operator<<( ostream& os, const CudaHostMatrix& m )
{
    os << "[" << std::endl;
    for ( size_t i = 0; i < m.height; i++ )
    {
        os << "  [";
        for ( size_t j = 0; j < m.width; j++ )
        {
            os << m( i, j );
            if ( j < m.width - 1 )
                os << ", ";
        }
        os << "]" << std::endl;
    }
    os << "]" << std::endl;
    return os;
}

/**
 * @brief Prints a human readable representation of CudaRingBuffer to a stream.
 * 
 * @param os Stream to print to
 * @param m CudaMatrix that should be printed
 * @return ostream& Stream given as input (for operator chaining)
 */
ostream& operator<<( ostream& os, const CudaHostRingBuffer& m )
{
    os << "[" << std::endl;
    for ( size_t i = 0; i < m.usedHeight; i++ )
    {
        os << "  [";
        for ( size_t j = 0; j < m.width; j++ )
        {
            os << m( i, j );
            if ( j < m.width - 1 )
                os << ", ";
        }
        os << "]" << std::endl;
    }
    os << "]" << std::endl;
    return os;
}

CudaHostRingBuffer::CudaHostRingBuffer( const CudaHostMatrix& m, unsigned int heightCapacity )
    : CudaHostMatrix( heightCapacity == 0 ? m.getHeight() : heightCapacity, m.getWidth() )
    , offset( 0 )
    , usedHeight( min( m.getHeight(), getHeight() ) )
{
    unsigned int copy_offset = m.getHeight() - getHeight();
    thrust::copy( m.elements.begin() + copy_offset * getWidth(), m.elements.end(), elements.begin() );
}

void CudaHostRingBuffer::pushRowFromMatrix( cudaStream_t& stream, const CudaHostMatrix& m, unsigned int col )
{
    unsigned int nextRow;

    if ( usedHeight < height )
    {
        nextRow = usedHeight;
        usedHeight++;
    }
    else
    {
        nextRow = offset;
        offset = ( offset + 1 ) % height;
    }

    // ugly hack, because strided_range doesn't work with const objects
    CudaHostMatrix& m_nonconst = const_cast<CudaHostMatrix&>( m );
    typedef thrust::device_vector<apsm_fp>::iterator Iterator;
    strided_range<Iterator> column_iterator( m_nonconst.elements.begin() + col, m_nonconst.elements.end(), m_nonconst.width );
    thrust::copy( thrust::cuda::par.on( stream ), column_iterator.begin(), column_iterator.end(), elements.begin() + nextRow * width );
}

/**
 * @brief CUDA dictionary kernel
 * @details Copy a transposed version of the rx_data matrix into the basis matrix.
 *
 * @param[out] basis basis matrix (dictionary)
 * @param[in] rx_data data constellations
 * 
 * @return void 
 */
__global__ void kernel_apsm_dictionary( CudaDeviceRingBuffer basis, const CudaDeviceMatrix rx_data )
{
    const unsigned int blockId = getBlockIdx_1D();
    const unsigned int blockThreadId = getBlockThreadIdx_1D();

    const unsigned int batch_size = getBlockDim_1D();

    const unsigned int linearLength = rx_data.getHeight();

    for ( unsigned int batch_idx = blockThreadId; batch_idx < linearLength; batch_idx += batch_size )
    {
        basis( blockId, batch_idx ) = rx_data( batch_idx, blockId );
    }
}

void CudaHostRingBuffer::pushRowsFromMatrix( cudaStream_t& stream, const CudaHostMatrix& m, unsigned int startOffset )
{
    // temporarily raise usedHeight to allow writing to whole matrix
    usedHeight = m.getWidth();

    unsigned int vector_size = m.getWidth(); // sample length

    // compute kernel launch dimensions
    dim3 block_dim, grid_dim;
    apsm_kernel_dims( &block_dim, &grid_dim, vector_size );

    // calculate shared memory size parameter
    unsigned int sharedMemorySize = 0;

    // run kernel
    kernel_apsm_dictionary<<<grid_dim, block_dim, sharedMemorySize, stream>>>( this->toDevice(), m.toDevice() );

    usedHeight = 0;
}

void CudaHostRingBuffer::moveWindow( unsigned int count )
{
    if ( usedHeight < height )
    {
        usedHeight += count;
    }
    else
    {
        offset = ( offset + count ) % height;
    }
}

void CudaHostRingBuffer::pushEmptyRow( cudaStream_t& stream )
{
    if ( usedHeight < height )
    {
        usedHeight++;
    }
    else
    {
        thrust::fill( thrust::cuda::par.on( stream ), elements.begin() + offset * width, elements.begin() + ( offset + 1 ) * width, 0 );
        offset = ( offset + 1 ) % height;
    }
}
