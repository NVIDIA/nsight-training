#!/usr/bin/env sh
port=$1
/opt/websockify/run ${port} --web=/opt/noVNC --wrap-mode=ignore -- vncserver ${DISPLAY} -3dwm