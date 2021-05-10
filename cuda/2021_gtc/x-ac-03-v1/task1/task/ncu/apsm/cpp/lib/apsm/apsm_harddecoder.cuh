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
 * @file apsm_harddecoder.cuh
 * @brief APSM hard decoder header
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2019.11.14   0.01    initial version
 * @date 2020.06.30   0.02    switchable modulation schemes
 */

#pragma once

#include <algorithm>
#include <exception>

/**
 * @defgroup APSM_CUDA_LIBRARY APSM CUDA library
 *
 * @{
 */

// APSM helper
#include "apsm_types.cuh"

class ApsmModulation
{
public:
    enum Type
    {
        off = 0, ///< OFF
        bpsk = 1, ///< BPSK, 1 bit per symbol
        qpsk = 2, ///< QPSK, 2 bits per symbol
        qam16 = 4, ///< QAM16, 4 bits per symbol
        qam64 = 6, ///< QAM64, 6 bits per symbol
        qam256 = 8, ///< QAM256, 8 bits per symbol
        qam1024 = 10, ///< QAM1024, 10 bits per symbol
    };

    ApsmModulation( Type type )
        : type( type )
    {
        switch ( type )
        {
        case off:
            scale = 0; // scale: mute
            name = "OFF";
            break;
        case bpsk:
            scale = 1; // scale: 1 / sqrt( 1 ) amplitudes: ±{1}
            name = "BPSK";
            break;
        case qpsk:
            scale = 707.106781186547437e-003; // scale: 1 / sqrt( 2 ) amplitudes: ±{1}
            name = "QPSK";
            break;
        case qam16:
            scale = 316.227766016837961e-003; // scale: 1 / sqrt( 10 ) amplitudes: ±{1,3}
            name = "QAM16";
            break;
        case qam64:
            scale = 154.303349962091914e-003; // scale: 1 / sqrt( 42 ) amplitudes: ±{1,3,5,7}
            name = "QAM64";
            break;
        case qam256:
            scale = 76.696498884737039e-003; // scale: 1 / sqrt( 170 ) amplitudes: ±{1,3,5,7,9,11,13,15}
            name = "QAM256";
            break;
        case qam1024:
            scale = 38.291979053374178e-003; // scale: 1 / sqrt( 682 ) amplitudes: ±{1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31}
            name = "QAM1024";
            break;
        default:
            throw std::invalid_argument( "Unsupported modulation" );
        }
    }
    static ApsmModulation fromString( std::string name )
    {
        std::string upperName;
        std::transform( name.begin(), name.end(), upperName.begin(), ::toupper );

        if ( name == "OFF" )
            return off;
        if ( name == "BPSK" )
            return bpsk;
        if ( name == "QPSK" )
            return qpsk;
        if ( name == "QAM16" )
            return qam16;
        if ( name == "QAM64" )
            return qam64;
        if ( name == "QAM256" )
            return qam256;
        if ( name == "QAM1024" )
            return qam1024;
        throw std::invalid_argument( "Unsupported modulation" );
    }
    static ApsmModulation fromBitPerSymbol( unsigned int bps )
    {
        return static_cast<Type>( bps );
    }

    apsm_fp getScale() const
    {
        return scale;
    }
    apsm_fp getRevScale() const
    {
        return 1. / scale;
    }
    std::string getName() const
    {
        return name;
    }
    unsigned int getBitPerSymbol() const
    {
        return static_cast<unsigned int>( type );
    }

    std::vector<bool> hardDecode( const ComplexSampleVector& input ) const;

protected:
    Type type;
    apsm_fp scale;
    std::string name;
};

/**
 * @}
 */
