# Copyright (c) 2019-2021
# Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
# All rights reserved.
# 
# Licensed by NVIDIA CORPORATION with permission. 
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# 
# NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, including without 
# limitation the patents of Fraunhofer, ARE GRANTED BY THIS SOFTWARE LICENSE. 
# Fraunhofer provides no warranty of patent non-infringement with respect to 
# this software. 

file( GLOB APSM_SRC_LIST
    ../cli/common/binary_load.hpp
    ../cli/common/binary_load.cpp
    ../cli/common/util.hpp
    ../cli/common/util.cpp
)

# use APSM and CUDA lib
include_directories(
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/external/argparse/include
)

link_directories(
)

### Make the executeable ######################################################

add_executable( ${PROJECT_NAME}_tool ${APSM_SRC_LIST} ./tool/apsm_tool.cpp ./tool/apsm_tool.hpp )
target_link_libraries( ${PROJECT_NAME}_tool apsm_lib )
target_compile_options( ${PROJECT_NAME}_tool PRIVATE -Werror -Wall -Wextra )

add_executable( ${PROJECT_NAME}_tool_parallel ${APSM_SRC_LIST} ./tool/apsm_tool_parallel.cpp ./tool/apsm_tool_parallel.hpp )
target_link_libraries( ${PROJECT_NAME}_tool_parallel apsm_lib )
target_compile_options( ${PROJECT_NAME}_tool_parallel PRIVATE -Werror -Wall -Wextra )
