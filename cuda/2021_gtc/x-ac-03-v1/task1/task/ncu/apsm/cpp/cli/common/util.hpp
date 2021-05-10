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

#pragma once

#include <argparse/argparse.hpp>

#include <apsm/apsm_parameters.cuh>
#include <apsm/apsm_types.cuh>

enum OutputMode
{
    BER,
    SER,
    EVM,
    Scripting
};

unsigned int printSymbolErrors( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const apsm_parameters& par );
unsigned int printBitErrors( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const apsm_parameters& par );
float printEvm( ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const apsm_parameters& par );
void printErrors( OutputMode outputMode, ComplexSampleVector estimatedSamples, ComplexSampleVector trueSamples, const apsm_parameters& par );

void argparseAddApsmArgs( argparse::ArgumentParser& arpa );
void argparseAddModeArgs( argparse::ArgumentParser& arpa );
void argparseAddOutputModeArgs( argparse::ArgumentParser& arpa );

apsm_parameters argparseGetApsmParams( argparse::ArgumentParser& arpa );
bool argparseGetMode( argparse::ArgumentParser& arpa );
OutputMode argparseGetOutputMode( argparse::ArgumentParser& arpa );
unsigned int antennaPatternRandom( unsigned int numAntennas );
unsigned int antennaPatternEquidistant( unsigned int numAntennas );
unsigned int antennaPatternFirst( unsigned int numAntennas );
