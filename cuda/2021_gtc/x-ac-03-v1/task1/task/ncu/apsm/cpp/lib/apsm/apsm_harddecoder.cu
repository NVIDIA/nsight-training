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

#include "apsm_harddecoder.cuh"

/**
* Demaps single dimension of QAM constellation to grey-decoded bits.
* Symbols are placed at a distance of 2, starting at -num_symbols + 1 and ending at num_symbols - 1.
* ATTENTION: Constellation is different from the one defined in the LTE standard.
*
* @param  x           input symbol
* @param  num_symbols number of symbols
* @return             decoded bits
*/
uint32_t demap_1d( const apsm_fp x, const unsigned int num_symbols )
{
    // convert ... -7 -5 -3 -1 1 3 5 7 ... to 0 1 2 3 4 5 6 7 ...
    int32_t xi = (int32_t)round( ( x + num_symbols - 1 ) / 2 );

    // limit to the valid range
    xi = std::min( (int32_t)num_symbols - 1, std::max( 0, xi ) );

    const uint32_t xi_gc = xi ^ ( xi >> 1 );
    return xi_gc;
}

uint32_t demap( const ComplexSample x, const unsigned int bps )
{
    unsigned int num_symbols_1d = 1 << ( bps / 2 );
    uint32_t real_bits = demap_1d( x.real(), num_symbols_1d );
    uint32_t imag_bits = demap_1d( x.imag(), num_symbols_1d );
    return ( real_bits << ( bps / 2 ) ) | imag_bits;
}

std::vector<bool> ApsmModulation::hardDecode( const ComplexSampleVector& input ) const
{
    std::vector<bool> output;
    output.reserve( input.size() * getBitPerSymbol() );

    apsm_fp revScale = getRevScale();

    for ( auto& sym : input )
    {
        uint32_t bits = demap( sym * revScale, getBitPerSymbol() );
        for ( int i = 0; i < getBitPerSymbol(); i++ )
            output.push_back( bits & ( 1 << ( getBitPerSymbol() - i - 1 ) ) );
    }
    return output;
}
