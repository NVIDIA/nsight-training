# GTC 2021 -  Debugging and Analyzing Correctness of CUDA Applications

In this hands-on lab, you will learn about using the cuda-gdb and
compute-sanitizer tools in order to debug and analyze the correctness of CUDA
applications. It is made of two parts:

* In the first part, you'll learn the basics of Nisght Eclipse Edition and
CUDA-GDB: how to build a project, how to start the debugger, detect and
remediate hardware exceptions and inspect and modify GPU memory at runtime. 
* In the second part, you will learn how to build an application to run with the
compute-sanitizer, how to use memcheck, initcheck and racecheck tools, and how
to interpret their reports as well as some of their optional features.

This repository contains training content for the [NVIDIA Nsight GTC 2021 lab](https://gtc21.event.nvidia.com/media/Debugging%20and%20Analyzing%20Correctness%20of%20CUDA%20Applications%20%5BT2504%5D/1_ee1hrgbn). Follow the link to watch the session recording on the GTC website.

## Instructions

This lab is designed to be run inside docker containers for which build
instructions are provided.

If you do not wish to follow the below steps to setup the docker containers,
the lab instructions can be found in the folder `task/`, in the form of Jupyter
notebooks. The code for the applications can be found in
`correctness/fs/home/nvidia/`.

### Requirements

* nvidia-docker : please refer to the nvidia-docker installation guide
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
* docker-compose
* This lab has been tested on NVIDIA Turing GPUs but is expected to work with
  Maxwell GPUs or more recent

### Usage

In the current folder, run `docker-compose up -d`. This will build and start the
needed docker containers. Once the containers are started, you can access the
lab instruction by accessing http://127.0.0.1:9333/lab/ in your browser.
When you are finished you can run `docker-compose down` to stop the
containers.
