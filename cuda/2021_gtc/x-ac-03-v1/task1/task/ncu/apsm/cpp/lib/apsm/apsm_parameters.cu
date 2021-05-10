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

#include <iomanip>

#include "apsm_parameters.cuh"

ostream& operator<<( ostream& os, const apsm_parameters& par )
{
    os << "{" << endl;
    os << "    % APSM kernel parameters" << endl;
    os << "    par.linearKernelWeight = " << par.linearKernelWeight << endl;
    os << "    par.gaussianKernelWeight = " << par.gaussianKernelWeight << endl;
    os << "    par.gaussianKernelVariance = " << par.gaussianKernelVariance << endl;
    os << "    % training parameters" << endl;
    os << "    par.windowSize = " << par.windowSize << endl;
    os << "    par.sampleStep = " << par.sampleStep << endl;
    os << "    par.machineLearningUser = " << par.machineLearningUser << endl;
    os << "    par.dictionarySize = " << par.dictionarySize << endl;
    os << "    par.normConstraint = " << par.normConstraint << endl;
    os << "    par.eB = " << setprecision( 4 ) << par.eB << endl;
    os << "    % other parameters" << endl;
    os << "    par.modulation = " << par.modulation.getName() << endl;
    os << "}";

    return os;
}
