#!/usr/bin/env bash
# Launch container and export DISPLAY

IMAGE_NAME=rigidity:1.0

# Make sure UID/GID are consistent inside the container with the current user
TMP_DIR=/tmp
CURR_DIR=$(pwd)
UID_TO_USE=$(stat -c '%u' "${0}")
GID_TO_USE=$(stat -c '%g' "${0}")

FAUX_DIR=$(mktemp -d -p "${TMP_DIR}")
FAUX_PASSWD="$FAUX_DIR/passwd"
FAUX_GRP="$FAUX_DIR/group"

echo "developer:x:$UID_TO_USE:$GID_TO_USE:developer:/:/bin/bash" > $FAUX_DIR/passwd
echo "developer:x:$GID_TO_USE:developer" > $FAUX_DIR/group

nvidia-docker run \
		--user=${UID_TO_USE}:${GID_TO_USE} \
		-v ${HOME}/.ssh:/.ssh \
		-v $FAUX_PASSWD:/etc/passwd:ro \
		-v $FAUX_GRP:/etc/group:ro \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $HOME/.Xauthority:/.Xauthority \
		-v $CURR_DIR:/rigidity \
		-e DISPLAY=unix$DISPLAY \
		-it \
		--rm \
		"${IMAGE_NAME}"

