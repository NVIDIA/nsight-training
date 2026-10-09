# Nsight VLOG Series: Memory Workload Analysis

[Nsight compute website](https://developer.nvidia.com/nsight-compute)

This is the source repository to the [video tutorial](https://youtube.com/playlist?list=PL5B692fm6--vScfBaxgY89IRWFzDt0Khm&si=2KpBgqkER44zgAG5) on Nsight memory workload analysis section.
Documentation of the memory chart can be found [here](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart).


To configure and build this project you should have CUDA and CMake installed. 
After that the project setup is as simple as:
```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=RelWithDebInfo . 
cmake --build build --paralle
```

To execute the benchmark mode I recommend the following executable:
```
$ ./build/memory_workload_benchmark
Elapsed time for optimization level UNALIGNED: 29.115200 ms
Elapsed time for optimization level ALIGNED: 1.814528 ms
Elapsed time for optimization level MULTIPLE_LOADS: 1.850368 ms
Elapsed time for optimization level WIDE_LOADS: 0.522240 ms
```
For profiling it is more sensible to execute `./build/nsight_vlog_memory_workload` as it only launches the kernel once and Nsight Compute takes care of accurate timing.
You can specify the optimization mode via an integer command line argument:
```
$ ./build/nsight_vlog_memory_workload
Please specify one of the optimization modes: 
	Mode 0:UNALIGNED
	Mode 1:ALIGNED
	Mode 2:MULTIPLE_LOADS
	Mode 3:WIDE_LOADS
```
