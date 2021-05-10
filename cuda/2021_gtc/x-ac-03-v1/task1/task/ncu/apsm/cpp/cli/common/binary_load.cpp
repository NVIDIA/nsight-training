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
 * @file binary_load.cpp
 * @brief Binary file load functions
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.10.20   0.01    initial version
 */

// function header
#include "binary_load.hpp"

void loadData( std::string filename, std::string name, ComplexSampleMatrix& data, ComplexSample scale, unsigned int loadNumAntennas, unsigned int loadNumSamples )
{
    typedef BinaryFile::MultidimArray MultidimArray;

    const bool verbose = false;
    std::map<std::string, MultidimArray> rx_data = BinaryFile::read( filename.c_str(), verbose );

    // get test vectors for received signal (SDR)
    uint32_t numSamples = rx_data[ name.c_str() ].sizes[ 0 ];
    uint32_t numAntennas = rx_data[ name.c_str() ].sizes[ 1 ];
    uint32_t num_iq = 2;

    if ( loadNumAntennas == 0 )
        loadNumAntennas = numAntennas;
    if ( loadNumSamples == 0 )
        loadNumSamples = numSamples;

    data.resize( loadNumAntennas );
    for ( unsigned int dim = 0; dim < loadNumAntennas; dim++ )
    {
        data[ dim ].resize( loadNumSamples );
        for ( unsigned int len = 0; len < loadNumSamples; len++ )
        {
            data[ dim ][ len ] = scale * ComplexSample( rx_data[ name.c_str() ].data[ 0 + len * num_iq * numAntennas + dim * num_iq ], rx_data[ name.c_str() ].data[ 1 + len * num_iq * numAntennas + dim * num_iq ] );
        }
    }
}

BinaryFileWriter::BinaryFileWriter( std::string _filename )
    : filename( _filename )
{
}

void BinaryFileWriter::addData( std::string name, const ComplexSampleMatrix& data )
{
    unsigned int num_antennas = data.size();
    unsigned int num_samples = data[ 0 ].size();
    unsigned int num_iq = 2;

    data_map[ name ] = MultidimArray( { num_samples, num_antennas, num_iq } );
    for ( unsigned int dim = 0; dim < num_antennas; dim++ )
    {
        for ( unsigned int len = 0; len < num_samples; len++ )
        {
            data_map[ name ].data[ 0 + len * num_iq * num_antennas + dim * num_iq ] = real( data[ dim ][ len ] );
            data_map[ name ].data[ 1 + len * num_iq * num_antennas + dim * num_iq ] = imag( data[ dim ][ len ] );
        }
    }
}

void BinaryFileWriter::write()
{
    BinaryFile::write( filename.c_str(), data_map );
}
