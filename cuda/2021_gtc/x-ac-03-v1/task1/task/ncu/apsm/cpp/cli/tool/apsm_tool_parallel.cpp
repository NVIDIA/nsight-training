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
 * @file apsm_tool.cpp
 * @brief APSM command line interface (cli) tool
 *
 * @author Matthias Mehlhose, HHI, matthias.mehlhose@hhi.fraunhofer.de
 *
 * @date 2020.09.29   0.01    initial version
 * @date 2020.10.14   0.02    remove boost dependencies
 */

#include <thread>
#include <vector>

// Argparse (command line argument parser)
#include <argparse/argparse.hpp>

// FILE IO
#include <binary/binary_file.hpp>

// APSM versions
#include <apsm/apsm_versions.h>

// APSM train and detect include
#include <apsm/apsm_wrapper.cuh>

//functions
#include "../common/binary_load.hpp"
#include "../common/util.hpp"
#include "apsm_tool_parallel.hpp"

int main( int argc, char* argv[] )
{
    std::string rxDataFile;
    std::string txDataFile;

    ComplexSampleMatrix txSigTraining;
    ComplexSampleMatrix txSigData;
    ComplexSampleMatrix rxSigTraining;
    ComplexSampleMatrix rxSigData;

    apsm_parameters par;
    OutputMode outputMode;

    {
        argparse::ArgumentParser arpa( "APSM_tool", "0.0.2" );

        // data parameters
        arpa.add_argument( "-s", "--synchronized-data" )
            .default_value( std::string( "../data/offline/rx/time/rxData_QAM16_alltx_converted.bin" ) )
            .required()
            .help( "synchronized data file" )
            .action( []( const std::string& value ) { return value; } );

        arpa.add_argument( "-r", "--reference-data" )
            .default_value( std::string( "../data/offline/tx/NOMA_signals_qam16_complex.bin" ) )
            .required()
            .help( "reference data file" )
            .action( []( const std::string& value ) { return value; } );

        argparseAddOutputModeArgs( arpa );
        argparseAddApsmArgs( arpa );

        try
        {
            arpa.parse_args( argc, argv );

            // data parameter
            rxDataFile = arpa.get<std::string>( "--synchronized-data" );
            txDataFile = arpa.get<std::string>( "--reference-data" );

            outputMode = argparseGetOutputMode( arpa );
            par = argparseGetApsmParams( arpa );
        }
        catch ( std::exception& err )
        {
            std::cerr << "Parsing error : " << err.what() << "\n";
            return EXIT_FAILURE;
        }
    }

    // ------------------------------------------------------------------------

    loadData( rxDataFile, "rxSigTraining", rxSigTraining );
    loadData( rxDataFile, "rxSigData", rxSigData );

    // import tx reference data
    loadData( txDataFile, "mod_training", txSigTraining, par.modulation.getScale(), 6, 685 );
    loadData( txDataFile, "mod_data", txSigData, par.modulation.getScale(), 6, 3840 );

    std::cout << "APSM Detect Version: " << apsm_get_version_string( APSM_DETECT_VERSION ) << std::endl;
    std::cout << "Parameters: " << par << std::endl;

    // process data
    processData( txSigTraining, txSigData, rxSigTraining, rxSigData, par, outputMode );

    // ------------------------------------------------------------------------

    return EXIT_SUCCESS;
}

void processData( ComplexSampleMatrix txSigTraining, ComplexSampleMatrix txSigData, ComplexSampleMatrix rxSigTraining, ComplexSampleMatrix rxSigData, apsm_parameters par, OutputMode outputMode )
{
    const unsigned int numUsers = 6;

    std::vector<std::thread> threads;
    std::vector<ComplexSampleMatrix> estSigDatas( numUsers );
    ApsmWrapper wrappers[ numUsers ];

    for ( unsigned int i = 0; i < numUsers; i++ )
    {
        par.machineLearningUser = i;
        threads.push_back( std::thread( &ApsmWrapper::wrapperChain, std::ref( wrappers[ i ] ), rxSigTraining, txSigTraining, rxSigData, std::ref( estSigDatas[ i ] ), par ) );
    }

    for ( unsigned int i = 0; i < numUsers; i++ )
    {
        threads[ i ].join();
        par.machineLearningUser = i;
        printErrors( outputMode, estSigDatas[ i ][ 0 ], txSigData[ i ], par );
    }
}
