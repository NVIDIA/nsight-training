# Optimizing CUDA Machine Learning Codes with Nsight Profiling Tools

This lab teaches how to use NVIDIA's Nsight tools for analyzing and optimizing CUDA applications. Attendees will be using Nsight Systems to analyze the overall application structure and explore parallelization opportunities. Nsight Compute will be used to analyze and optimize CUDA kernels, using an online machine learning code for 5G.

This repository contains training content for the [NVIDIA Nsight GTC 2021 lab](https://events.rainfocus.com/widget/nvidia/nvidiagtc/sessioncatalog?search=dlit1605). Follow the link to watch the session recording on the GTC website, or take the training by simply using the instructions in the jupyter notebooks.

## Docker Compose Usage

After cloning this repo move into the `task1` directory and then do:

`docker-compose up -d`

If you need to rebuild containers instead do:

`docker-compose up -d --build`

To check on running containers (for example to get their ports) do:

`docker ps`

To get logs for all running containers do:

`docker-compose logs`

If you want to follow logs add the `-f` flag:

`docker-compose logs -f`

If you only want logs for one container, get its service name from `docker-compose.yml` and do:

`docker logs -f <service-name>` e.g. `docker-logs -f nsight`

When finished, please spin down containers from within the `task1` directory with:

`docker-compose down`

## Accessing JupyterLab

After running `docker-compose up -d` (see above) and confirming the containers have spun up, visit 127.0.0.1:9333/lab in your browser. You can modify the mapped port (e.g. `9333`) via `task1/.env` by modifying `NGINX_DEV_PORT`.

## Working With the Remote Desktop

The remote desktop can be accessed at the `/nsight/` endpoint of the same host and port that the Jupyter notebook is running on. For example, if Jupyter is accessed at `127.0.0.1:9333` than the remote desktop is at `127.0.0.1:9333/nsight/`.

The password is `nvidia`.

There is a shared file system mount between the `lab` (Jupyter) container and the `nsight` (remote desktop) container where the `lab` container's `/dli/task` directory is mounted into the `nsight` container's `/root/Desktop/reports/` directory. See `task1/docker-compose.yml` for details or to edit.


