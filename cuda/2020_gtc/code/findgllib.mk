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
################################################################################
#
#  findgllib.mk is used to find the necessary GL Libraries for specific distributions
#               this is supported on Mac OSX and Linux Platforms
#
################################################################################

# Determine OS platform and unix distribution
ifeq ("$(TARGET_OS)","linux")
   # first search lsb_release
   DISTRO  = $(shell lsb_release -i -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
   ifeq ("$(DISTRO)","")
     # second search and parse /etc/issue
     DISTRO = $(shell more /etc/issue | awk '{print $$1}' | sed '1!d' | sed -e "/^$$/d" 2>/dev/null | tr "[:upper:]" "[:lower:]")
     # ensure data from /etc/issue is valid
     ifeq (,$(filter $(DISTRO),ubuntu fedora red rhel centos suse))
       DISTRO =
     endif
     ifeq ("$(DISTRO)","")
       # third, we can search in /etc/os-release or /etc/{distro}-release
       DISTRO = $(shell awk '/ID/' /etc/*-release | sed 's/ID=//' | grep -v "VERSION" | grep -v "ID" | grep -v "DISTRIB")
     endif
   endif
endif

ifeq ("$(TARGET_OS)","linux")
    # $(info) >> findgllib.mk -> LINUX path <<<)
    # Each set of Linux Distros have different paths for where to find their OpenGL libraries reside
    UBUNTU = $(shell echo $(DISTRO) | grep -i ubuntu      >/dev/null 2>&1; echo $$?)
    FEDORA = $(shell echo $(DISTRO) | grep -i fedora      >/dev/null 2>&1; echo $$?)
    RHEL   = $(shell echo $(DISTRO) | grep -i 'red\|rhel' >/dev/null 2>&1; echo $$?)
    CENTOS = $(shell echo $(DISTRO) | grep -i centos      >/dev/null 2>&1; echo $$?)
    SUSE   = $(shell echo $(DISTRO) | grep -i 'suse\|sles' >/dev/null 2>&1; echo $$?)
    ifeq ("$(UBUNTU)","0")
      ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        GLPATH := /usr/arm-linux-gnueabihf/lib
        GLLINK := -L/usr/arm-linux-gnueabihf/lib
        ifneq ($(TARGET_FS),)
          GLPATH += $(TARGET_FS)/usr/lib/arm-linux-gnueabihf
          GLLINK += -L$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
      else ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-aarch64)
        GLPATH := /usr/aarch64-linux-gnu/lib
        GLLINK := -L/usr/aarch64-linux-gnu/lib
        ifneq ($(TARGET_FS),)
          GLPATH += $(TARGET_FS)/usr/lib
          GLPATH += $(TARGET_FS)/usr/lib/aarch64-linux-gnu
          GLLINK += -L$(TARGET_FS)/usr/lib/aarch64-linux-gnu
        endif
      else ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-ppc64le)
        GLPATH := /usr/powerpc64le-linux-gnu/lib
        GLLINK := -L/usr/powerpc64le-linux-gnu/lib
      else
        UBUNTU_PKG_NAME = $(shell which dpkg >/dev/null 2>&1 && dpkg -l 'nvidia-*' | grep '^ii' | awk '{print $$2}' | head -1)
        ifneq ("$(UBUNTU_PKG_NAME)","")
          GLPATH    ?= /usr/lib/$(UBUNTU_PKG_NAME)
          GLLINK    ?= -L/usr/lib/$(UBUNTU_PKG_NAME)
        endif

        DFLT_PATH ?= /usr/lib
      endif
    endif
    ifeq ("$(SUSE)","0")
      GLPATH    ?= /usr/X11R6/lib64
      GLLINK    ?= -L/usr/X11R6/lib64
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(FEDORA)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(RHEL)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(CENTOS)","0")
      GLPATH    ?= /usr/lib64/nvidia
      GLLINK    ?= -L/usr/lib64/nvidia
      DFLT_PATH ?= /usr/lib64
    endif

  # find libGL, libGLU
  GLLIB  := $(shell find -L $(GLPATH) $(DFLT_PATH) -name libGL.so  -print 2>/dev/null)
  GLULIB := $(shell find -L $(GLPATH) $(DFLT_PATH) -name libGLU.so -print 2>/dev/null)

  ifeq ("$(GLLIB)","")
      $(info >>> WARNING - libGL.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif
  ifeq ("$(GLULIB)","")
      $(info >>> WARNING - libGLU.so not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif

  HEADER_SEARCH_PATH ?= $(TARGET_FS)/usr/include
  ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
      HEADER_SEARCH_PATH += /usr/arm-linux-gnueabihf/include
  else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-aarch64-linux)
      HEADER_SEARCH_PATH += /usr/aarch64-linux-gnu/include
  endif

  GLHEADER  := $(shell find -L $(HEADER_SEARCH_PATH) -name gl.h -print 2>/dev/null)
  GLUHEADER := $(shell find -L $(HEADER_SEARCH_PATH) -name glu.h -print 2>/dev/null)

  ifeq ("$(GLHEADER)","")
      $(info >>> WARNING - gl.h not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif
  ifeq ("$(GLUHEADER)","")
      $(info >>> WARNING - glu.h not found, refer to CUDA Getting Started Guide for how to find and install them. <<<)
      SAMPLE_ENABLED := 0
  endif
else
    # This would be the Mac OS X path if we had to do anything special
endif

