# Profiling Python and AI workloads with Nsight Compute (GTC 2026)

Course ID: x-ac-39-v1

Author: Felix Schmitt, Mark Mason

Maintainer: Josh Wyatt

Contributors: Peter Labus, Magnus Strengert

Recording: https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81545


## Summary

Learn how to use the Nsight Compute tool to profile CUDA and AI GPU
kernels in the Python framework of your choice: Numba, Triton or PyTorch.
Understand the optimization workflows provided by NVIDIA Nsight Compute UI
and its ncu command line interface. Use the new scoreboard dependency analysis
view and source page improvements like opcode to metric pipeline associations.
Learn how to use the Nsight Jupyterlab Extension for profiling and streaming
reports directly in Jupyter. Try the latest Nsight Python decorators
for profiling from within any Python script.


## Agenda

- Introduction to Nsight Compute
- Debug and Profile Numba CUDA Kernels with the Nsight Jupyterlab Extension
- Profile PyTorch LLM Inference with the ncu command line
- Optimizing Self-Attention Kernels written in Triton
- The Nsight Python Profiling API


## Disclaimer

The content was specifically developed and tested for NVIDIA H100 GPUs.
While it can be run on many other NVIDIA GPUs, concrete performance results will vary.
Parts of the content require native GPU support for FP8 precision formats.


## Usage

After cloning this repo move into the `task1` directory and build and start the containers:

`docker-compose up -d --build`

When finished, spin down containers from within the `task1` directory with:

`docker-compose down`


## Accessing JupyterLab

After running `docker compose up -d` (see above), visit http://127.0.0.1/lab in your browser.


