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

#include <string>

enum apsm_function_id {
    APSM_UNKNOWN_ID   = 0,
    APSM_DETECT_ID    = 1
};

constexpr unsigned int version_id( const enum apsm_function_id function, const int version )
{
    return (static_cast<int>(function) << 16) + version;
}

enum apsm_versions {
    UNKNOWN = 0,
    // apsm_detect versions
    APSM_DETECT_ORIGINAL = version_id( APSM_DETECT_ID, 1 ),
    APSM_DETECT_CG       = version_id( APSM_DETECT_ID, 2 ),
    APSM_DETECT_SPB      = version_id( APSM_DETECT_ID, 3 ),
    APSM_DETECT_SPLIT    = version_id( APSM_DETECT_ID, 4 ),
    APSM_DETECT_SHMEM    = version_id( APSM_DETECT_ID, 5 ),
    APSM_DETECT_BALANCED = version_id( APSM_DETECT_ID, 6 )
};

inline const std::string apsm_get_version_string( const int version_id )
{
    switch (version_id)
    {
        case APSM_DETECT_ORIGINAL:  return "APSM Detect original";
        case APSM_DETECT_CG:        return "APSM Detect Cooperative Groups";
        case APSM_DETECT_SPB:       return "APSM Detect Multiple Samples per Block";
        case APSM_DETECT_SPLIT:     return "APSM Detect Split linear and gaussian loops";
        case APSM_DETECT_SHMEM:     return "APSM Detect Store vectors in shared memory";
        case APSM_DETECT_BALANCED:  return "APSM Detect Balance memory and computation";
        default:
            return "Unknown";
    }
}

// select version to use
#define APSM_DETECT_VERSION apsm_versions::APSM_DETECT_CG
