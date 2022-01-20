# GTC 2020 Nsight Developer Tools Lab
In this hands-on lab, you'll learn about efficiently debugging, profiling, and optimizing CUDA applications on Linux.
Through a set of exercises, you'll use the latest features in NVIDIA's suite of tools to detect and fix common issues of correctness and performancein their applications.

This repository contains the source and project files necessary to try this lab in your own Linux machine. You can find the instruction videos for each step on the GTC website:
- Step 0: Building your project with Nsight Eclipse Edition: https://developer.nvidia.com/gtc/2020/video/t21395-1
- Step 1: Debugging with cuda-gdb: https://developer.nvidia.com/gtc/2020/video/t21395-2
- Step 3-4: Application-level performance optimization with Nsight Systems: https://developer.nvidia.com/gtc/2020/video/t21395-3
- Step 5-7: Kernel-level performance optimization with Nsight Compute: https://developer.nvidia.com/gtc/2020/video/t21395-4

The detailed steps, including all requirements and setup instructions, are in the [DevTools GTC 2020 Lab Content pdf](DevTools_GTC_2020_Lab_Content.pdf).

# Story
The exercise represents a cloud-based image classification application. It contains a database
(DB) of flower images. Users can submit their own flower pictures and the app classifies them
against the DB and sends the result back to the user. The application allows users to tweak
several factors including the threshold, number of source threads generating requests, and the
number of worker threads processing these requests.

For simplicity, the DB contains a fixed number of flower images, and users only submit pictures
of one of those flowers. For further simplification, the user images are identical to the DB
versions, except for added white noise to simulate some random errors when taking the
snapshot.

The exercise is to first fix several bugs in the application using correctness analysis tools to
make the classification work properly. Then, users must optimize the application (framework and
kernels) so that requests are served within some defined time threshold (e.g. 100ms). It is
structured into steps that build on each other, but it is possible to start at any chosen step, as
each one will contain all previous exercise content (e.g. all bugs are fixed in the first
performance analysis exercise).

This exercise uses NVIDIA® Nsight™ Eclipse Edition but all steps can be completed when
compiling and running directly from the command line. These steps are marked as “CLI
Alternative” below.

# Requirements
- Ubuntu 18.04 host (other OS variants may provide different results)
- Eclipse 4.9 (2018-09) for C++ developers
- default-jre apt package
- python apt package
- CUDA 10.2 toolkit
- Turing GPU (tested on RTX 2080Ti). If the device has multiple GPUs, ensure that compute (CUDA) and display (OpenGL) are on the same device, i.e. set CUDA_VISIBLE_DEVICES to the one with the display. Otherwise, CUDA plus OpenGL will result in high memory mapping overhead.
- libglew-dev apt package
- freeglut3-dev package
