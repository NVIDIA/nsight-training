# apsm_dictionary

Example program for APSM dictionary buffer.

## Requirements

- CMake 3.14.3 or newer
- C++ Compiler with C++11 support
- NVIDIA CUDA SDK
- Google Test v1.10.0 (will be pulled in when cloning the repo)

**Note:** You could use an older version of CMake (3.10 or newer) but you will need to specify

CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES to be your CUDA include directory.

## Building

### Clone the repository

```console
git clone --recurse-submodules git@gitlab.hhi.fraunhofer.de:nvidia/OGPU-project.git
```

if updating an existing repository, it will be needed to initialize the submodules when downloading:

```console
git submodule update --init --recursive --remote
```

### Generate and Compile

```console
cd cpp
mkdir build
cd build
cmake ..
make -j$(nproc)
```

if a different CUDA version is needed (or one in an alternate location), it can be specified as:

```console
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-<version>/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-<version> -DCMAKE_CUDA_FLAGS='-g -G -arch=sm_50 -gencode=arch=compute_70,code=sm_70' -v ..
```

## Test and run

```console
cd build
bin/APSM_cli
```

you should get as output

```console
linearBasisLength : 0
gaussianBasisLength : 0
windowLength : 0
```

```console
valgrind --leak-check=full ./bin/APSM_cli
```

```console
cuda-memcheck ./bin/APSM_cli
```

```console
cuda-gdb ./bin/APSM_cli
```
