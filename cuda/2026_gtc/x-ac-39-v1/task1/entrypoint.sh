#!/bin/bash

# JUPYTER_TOKEN will empty in development, but set for
# deployments. It will be applied from the environment
# automatically when running `docker-compose up`.
# For more details see `docker-compose.production.yml`,
# `docker-compose.override.yml`, and
# `nginx.conf`.

sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'

#Define cleanup procedure
cleanup() {
    echo "Container stopped, performing cleanup..."
    rm -f /tmp/.X[12]-lock
    rm -f /tmp/.X11-unix/X[12]
}

#Trap SIGTERM
trap 'cleanup' SIGTERM

nvidia-smi -pm 1

rm -f /root/.jupyter/lab/workspaces/*.jupyterlab-workspace

jupyter lab \
        --ip 0.0.0.0                               `# Run on localhost` \
        --allow-root                               `# Enable the use of sudo commands in the notebook` \
        --no-browser                               `# Do not launch a browser by default` \
        --NotebookApp.base_url="/"                 `# Allow value to be passed in for production` \
        --NotebookApp.token="$JUPYTER_TOKEN"       `# Do not require token to access notebook` \
        --NotebookApp.password=""                  `# Do not require password to run jupyter server`
