# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

TARGET_OS := linux
HOST_ARCH := x86_64
TARGET_ARCH := $(HOST_ARCH)

include findgllib.mk

FLAGS =
GL_FLAGS =
SRC_DIR ?=
SRC ?= imgserver.cu
EXE_NAME := imgserver

CUDA_VERSION ?= 10.2
CUDA_ROOT ?= /usr/local/cuda-$(CUDA_VERSION)
NVCC = $(CUDA_ROOT)/bin/nvcc

ifeq ("$(nogl)","1")
	FLAGS += -DNO_GL
else
	GL_FLAGS += -L $(SRC_DIR)lib/$(TARGET_ARCH) $(GLLINK) -lGLEW -lGL -lGLU -lglut
endif

ifneq ("$(step)","")
	FLAGS += -DSTEP=$(step)
endif

ifeq ("$(step)","0")
	# no extra flags, show rendering
	FLAGS += -DNO_EXTRACTION
	FLAGS += -DKERNEL_OPT_1
else ifeq ("$(step)","1")
	# debugger exercise
	FLAGS += -DGDB_EXERCISE
	FLAGS += -DKERNEL_OPT_1
	FLAGS += -G
else ifeq ("$(step)","2")
    $(error Reserved for future release)
else ifeq ("$(step)","3")
	# nsight systems initial trace
	FLAGS += -DKERNEL_OPT_1
else ifeq ("$(step)","4")
	# nsight systems buffer reuse optimization
	FLAGS += -DKERNEL_OPT_1
	FLAGS += -DREUSE_DEVICE_BUFFERS
else ifeq ("$(step)","5")
	# nsight systems render on demand optimization
	FLAGS += -DKERNEL_OPT_1
	FLAGS += -DREUSE_DEVICE_BUFFERS
	FLAGS += -DRENDER_ON_DEMAND
	FLAGS += -lineinfo
else ifeq ("$(step)","6")
	# nsight compute 32bit math optimization
	FLAGS += -DKERNEL_OPT_2
	FLAGS += -DREUSE_DEVICE_BUFFERS
	FLAGS += -DRENDER_ON_DEMAND
	FLAGS += -lineinfo
else ifeq ("$(step)","7")
	# nsight compute preprocess image optimization
	FLAGS += -DKERNEL_OPT_3
	FLAGS += -DREUSE_DEVICE_BUFFERS
	FLAGS += -DRENDER_ON_DEMAND
	FLAGS += -lineinfo
else
	# step 0
	FLAGS += -DKERNEL_OPT_1
endif

ifeq ("$(graphs)","1")
	FLAGS += -DCUDA_GRAPHS_1
else ifeq ("$(graphs)","2")
	FLAGS += -DCUDA_GRAPHS_2
endif

.PHONY: all clean

all: imgserver

clean:
	rm -f imgserver

imgserver: $(SRC)
	$(NVCC) -std=c++11 -Xcompiler=-fPIC -g -o $(EXE_NAME) $^ \
	$(FLAGS) \
	-Xlinker=--unresolved-symbols=ignore-in-shared-libs \
	-I $(SRC_DIR)inc -I $(CUDA_ROOT)/include \
	$(GL_FLAGS)
