#!/usr/bin/env bash

set -e

poetry run pip install git+ssh://git@github.com/kolenaIO/mkdocs-material-insiders.git
poetry run pip install git+ssh://git@github.com/kolenaIO/mkdocstrings-python.git
