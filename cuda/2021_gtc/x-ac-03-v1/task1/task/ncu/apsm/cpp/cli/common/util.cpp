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

#include <algorithm>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <random>
/*
// LIQUID DSP
#include <complex> // include complex first
#include <liquid/liquid.h>
*/
#include "util.hpp"

const unsigned int MAX_ANTENNAS = 16;

unsigned int printSymbolErrors( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const apsm_parameters& par )
{
    const float modScale = par.modulation.getScale();
    unsigned int symbol_errors = 0;
    const float maxConstellationVal = ( par.modulation.getBitPerSymbol() == 1 ) ? 1 : ( ( 1 << ( par.modulation.getBitPerSymbol() / 2 ) ) - 1 ) * modScale;

    for ( unsigned int ldx = 0; ldx < estimatedSamples.size(); ldx++ )
    {
        ComplexSample y = estimatedSamples[ ldx ];

        // linewrap after 80 chars
        if ( ldx % 80 == 0 )
            std::cout << std::endl;

        if ( std::abs( y.real() - trueSamples[ ldx ].real() ) > modScale
             && !( y.real() > maxConstellationVal && trueSamples[ ldx ].real() > maxConstellationVal - modScale )
             && !( y.real() < -maxConstellationVal && trueSamples[ ldx ].real() < -maxConstellationVal + modScale ) )
        {
            symbol_errors++;
            std::cout << "r";
        }
        else if ( std::abs( y.imag() - trueSamples[ ldx ].imag() ) > modScale
                  && !( y.imag() > maxConstellationVal && trueSamples[ ldx ].imag() > maxConstellationVal - modScale )
                  && !( y.imag() < -maxConstellationVal && trueSamples[ ldx ].imag() < -maxConstellationVal + modScale ) )
        {
            symbol_errors++;
            std::cout << "i";
        }
        else
        {
            std::cout << ".";
        }
    }
    int total_symbols = estimatedSamples.size();
    float symbol_error_percentage = 100. * symbol_errors / total_symbols;
    std::cout << " symbol errors for user " << par.machineLearningUser << ": " << symbol_errors << " of " << total_symbols << " - " << std::fixed << std::setprecision( 2 ) << symbol_error_percentage << " percent" << std::endl;

    return symbol_errors;
}

unsigned int printBitErrors( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const apsm_parameters& par )
{
    std::vector<bool> estimatedBits = par.modulation.hardDecode( estimatedSamples );
    std::vector<bool> trueBits = par.modulation.hardDecode( trueSamples );

    unsigned int totalBits = estimatedBits.size();
    unsigned int bitErrors = 0;
    for ( unsigned int i = 0; i < totalBits; i++ )
    {
        if ( i % ( par.modulation.getBitPerSymbol() * 80 ) == 0 )
            std::cout << std::endl;
        if ( i % par.modulation.getBitPerSymbol() == 0 )
            std::cout << " ";

        if ( estimatedBits[ i ] != trueBits[ i ] )
        {
            bitErrors++;
            std::cout << "x";
        }
        else
        {
            std::cout << ".";
        }
    }

    float bitErrorPercentage = 100. * bitErrors / totalBits;
    std::cout << " bit errors for user " << par.machineLearningUser << ": " << bitErrors << " of " << totalBits << " - " << std::fixed << std::setprecision( 2 ) << bitErrorPercentage << " percent" << std::endl;

    return bitErrors;
}

float printEvm( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const apsm_parameters& par )
{
    const float fullBarValue = 0.5; // For Error Magnitudes above this level, a full bar is shown
    const unsigned int totalSymbols = estimatedSamples.size();
    float squaredSum = 0;

    for ( unsigned int i = 0; i < totalSymbols; i++ )
    {
        // linewrap after 80 chars
        if ( i % 80 == 0 )
            std::cout << std::endl;

        float error = std::abs( estimatedSamples[ i ] - trueSamples[ i ] );
        squaredSum += error * error;
        switch ( int( error / fullBarValue * 8 ) )
        {
        case 0:
            std::cout << " ";
            break;
        case 1:
            std::cout << "\u2581";
            break;
        case 2:
            std::cout << "\u2582";
            break;
        case 3:
            std::cout << "\u2583";
            break;
        case 4:
            std::cout << "\u2584";
            break;
        case 5:
            std::cout << "\u2585";
            break;
        case 6:
            std::cout << "\u2586";
            break;
        case 7:
            std::cout << "\u2587";
            break;
        default:
            std::cout << "\u2588";
            break;
        }
    }

    float evm = sqrt( squaredSum / totalSymbols );
    std::cout << " EVM for user " << par.machineLearningUser << ": " << std::fixed << std::setprecision( 4 ) << evm << std::endl;
    return evm;
}

void printErrors( OutputMode outputMode, ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const apsm_parameters& par )
{
    switch ( outputMode )
    {
    case BER:
        printBitErrors( estimatedSamples, trueSamples, par );
        break;
    case SER:
        printSymbolErrors( estimatedSamples, trueSamples, par );
        break;
    case EVM:
        printEvm( estimatedSamples, trueSamples, par );
        break;
    case Scripting:
        unsigned int bitErrors = printBitErrors( estimatedSamples, trueSamples, par );
        unsigned int symbolErrors = printSymbolErrors( estimatedSamples, trueSamples, par );
        float evm = printEvm( estimatedSamples, trueSamples, par );
        std::cout.clear(); // Temporary enable output
        std::cout << setprecision( -1 ) << bitErrors << "," << symbolErrors << "," << evm << std::endl;
        std::cout.setstate( std::ios_base::failbit ); // Disable all outputs for quiet mode
        break;
    }
}

void argparseAddApsmArgs( argparse::ArgumentParser& arpa )
{

    arpa.add_argument( "-u", "--user" )
        .default_value( 0 )
        .help( "user to detect" )
        .action( []( const std::string& value ) { return std::stoi( value ); } );

    arpa.add_argument( "-d", "--distance" )
        .default_value( 0.001f )
        .help( "distance of modulation points" )
        .action( []( const std::string& value ) { return std::stof( value ); } );

    arpa.add_argument( "-gw", "--gaussian-weight" )
        .default_value( 0.5f )
        .help( "gaussian weight" )
        .action( []( const std::string& value ) { return std::stof( value ); } );

    arpa.add_argument( "-gv", "--gaussian-variance" )
        .default_value( 0.05f )
        .help( "gaussian variance of receiver (antenna) noise" )
        .action( []( const std::string& value ) { return std::stof( value ); } );

    arpa.add_argument( "-ws", "--window" )
        .default_value( 20 )
        .help( "window size" )
        .action( []( const std::string& value ) { return std::stoi( value ); } );

    arpa.add_argument( "-tp", "--train-passes" )
        .default_value( 1 )
        .help( " number of training passes over the training data" )
        .action( []( const std::string& value ) { return std::stoi( value ); } );

    arpa.add_argument( "-ss", "--step" )
        .default_value( 2 )
        .help( "step size" )
        .action( []( const std::string& value ) { return std::stoi( value ); } );

    arpa.add_argument( "-ds", "--dictionary-size" )
        .default_value( 685 )
        .help( "dictionary size" )
        .action( []( const std::string& value ) { return std::stoi( value ); } );

    arpa.add_argument( "-nc", "--norm-constraint" )
        .default_value( 0.0f )
        .help( "norm constraint for dictionary sparsification" )
        .action( []( const std::string& value ) { return std::stof( value ); } );

    arpa.add_argument( "-m", "--modulation" )
        .default_value( std::string( "QAM16" ) )
        .required()
        .help( "modulation scheme" )
        .action( []( const std::string& value ) { return value; } );

    arpa.add_argument( "-an", "--antenna-number" )
        .help( "number of antennas that should be used (pattern is determined randomly" )
        .action( []( const std::string& value ) { return std::stoi( value ); } );

    arpa.add_argument( "-ap", "--antenna-pattern" )
        .help( "bit string of antennas that should be used (string of 0 and 1)" )
        .action( []( const std::string& value ) { return value; } );

    arpa.add_argument( "-as", "--antenna-scheme" )
        .help( "scheme, that should be used for selecting the spedified number of antennas (random, equidistant, first)" )
        .default_value( std::string( "equidistant" ) )
        .action( []( const std::string& value ) { return value; } );
}

void argparseAddModeArgs( argparse::ArgumentParser& arpa )
{
    arpa.add_argument( "--tm", "--transmission-mode" )
        .default_value( std::string( "TIME" ) )
        .required()
        .help( "transmission mode" )
        .action( []( const std::string& value ) { return value; } );
}

void argparseAddOutputModeArgs( argparse::ArgumentParser& arpa )
{
    arpa.add_argument( "-p", "--print-mode" )
        .default_value( std::string( "SER" ) )
        .required()
        .help( "Print mode for error metric (BER = bit errors, SER = symbol errors, EVM = Error Vector Magnitude, scripting = disable all output except \"ber,ser,evm\")" )
        .action( []( const std::string& value ) { return value; } );
}

apsm_parameters argparseGetApsmParams( argparse::ArgumentParser& arpa )
{
    apsm_parameters par;

    par.linearKernelWeight = 1.0f - arpa.get<float>( "--gaussian-weight" );
    par.gaussianKernelWeight = arpa.get<float>( "--gaussian-weight" );
    par.gaussianKernelVariance = arpa.get<float>( "--gaussian-variance" );
    par.eB = arpa.get<float>( "--distance" );

    par.windowSize = arpa.get<int>( "--window" );
    par.trainPasses = arpa.get<int>( "--train-passes" );
    par.sampleStep = arpa.get<int>( "--step" );
    par.machineLearningUser = arpa.get<int>( "--user" );

    par.dictionarySize = arpa.get<int>( "--dictionary-size" ); // TODO (ds) add special value to use full dictionary
    par.normConstraint = arpa.get<float>( "--norm-constraint" );

    par.modulation = ApsmModulation::fromString( arpa.get<std::string>( "--modulation" ) );

    if ( arpa.present<std::string>( "--antenna-pattern" ) && ( arpa.present<int>( "--antenna-number" ) || arpa.present<int>( "--antenna-scheme" ) ) )
        throw std::runtime_error( "When --antenna-pattern is specified, --antenna-number or --antenna-scheme must not be used." );
    if ( arpa.present<std::string>( "--antenna-pattern" ) )
        par.antennaPattern = std::bitset<MAX_ANTENNAS>( arpa.get<std::string>( "--antenna-pattern" ) ).to_ulong();
    else if ( arpa.present<int>( "--antenna-number" ) )
    {
        unsigned int numAntennas = arpa.get<int>( "--antenna-number" );
        if ( numAntennas > MAX_ANTENNAS )
            throw std::runtime_error( "Number of antennas chosen is larger than number of available antennas." );

        std::string scheme = arpa.get<std::string>( "--antenna-scheme" );
        std::transform( scheme.begin(), scheme.end(), scheme.begin(), ::tolower );
        if ( scheme == "random" )
            par.antennaPattern = antennaPatternRandom( numAntennas );
        else if ( scheme == "equidistant" )
            par.antennaPattern = antennaPatternEquidistant( numAntennas );
        else if ( scheme == "first" )
            par.antennaPattern = antennaPatternFirst( numAntennas );
        else
            throw std::invalid_argument( "invalid antenna scheme" );

        std::cout << "Selected antenna pattern: " << std::bitset<MAX_ANTENNAS>( par.antennaPattern ).to_string() << std::endl;
    }

    return par;
}

bool argparseGetMode( argparse::ArgumentParser& arpa )
{
    std::string mode = arpa.get<std::string>( "--transmission-mode" );
    // convert to upper case
    std::transform( mode.begin(), mode.end(), mode.begin(), ::toupper );

    if ( mode == "TIME" )
    {
        std::cout << "Selected mode is TIME" << std::endl;
        return false;
    }
    else if ( mode == "OFDM" )
    {
        std::cout << "Selected mode is OFDM" << std::endl;
        return true;
    }
    else
    {
        std::cout << "Selected mode is unknown, defaulted to TIME" << std::endl;
        return false;
    }
}

OutputMode argparseGetOutputMode( argparse::ArgumentParser& arpa )
{
    std::string outputModeStr = arpa.get<std::string>( "--print-mode" );
    std::transform( outputModeStr.begin(), outputModeStr.end(), outputModeStr.begin(), ::toupper );
    if ( outputModeStr == "BER" )
        return BER;
    else if ( outputModeStr == "SER" )
        return SER;
    else if ( outputModeStr == "EVM" )
        return EVM;
    else if ( outputModeStr == "SCRIPTING" )
    {
        std::cout.setstate( std::ios_base::failbit ); // Disable all outputs for quiet mode
        return Scripting;
    }
    else
        throw std::invalid_argument( "invalid print mode" );
}

unsigned int antennaPatternRandom( unsigned int numAntennas )
{
    std::vector<unsigned int> antennas( MAX_ANTENNAS );
    for ( unsigned int i = 0; i < MAX_ANTENNAS; i++ )
        antennas[ i ] = i;

    std::random_device rd;
    std::mt19937 g( rd() );
    std::shuffle( antennas.begin(), antennas.end(), g );

    std::bitset<MAX_ANTENNAS> antennaPattern;
    for ( unsigned int i = 0; i < numAntennas; i++ )
        antennaPattern.set( antennas[ i ] );

    return antennaPattern.to_ulong();
}

unsigned int antennaPatternFirst( unsigned int numAntennas )
{
    std::bitset<MAX_ANTENNAS> antennaPattern;
    for ( unsigned int i = 0; i < numAntennas; i++ )
        antennaPattern.set( i );
    return antennaPattern.to_ulong();
}

unsigned int antennaPatternEquidistant( unsigned int numAntennas )
{
    std::bitset<MAX_ANTENNAS> antennaPattern;
    for ( unsigned int i = 0; i < numAntennas; i++ )
        antennaPattern.set( i * MAX_ANTENNAS / numAntennas );
    return antennaPattern.to_ulong();
}
