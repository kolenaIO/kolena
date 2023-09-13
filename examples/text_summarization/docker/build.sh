#!/usr/bin/env bash

set -eu
PROJECT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )

pushd "$PROJECT_DIR" >/dev/null
IMAGE_VERSION="$(git describe --tags --dirty)"

IMAGE_NAME="text-summarization"
IMAGE_TAG="$IMAGE_NAME:$IMAGE_VERSION"

echo "building $IMAGE_TAG..."

docker build \
    --platform linux/amd64 \
    --tag "$IMAGE_TAG" \
    --file "docker/text-summarization.dockerfile" \
    .

popd
