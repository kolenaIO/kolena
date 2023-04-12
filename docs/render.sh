#!/usr/bin/env bash

set -eu

echo "generating reST templates..."
poetry run sphinx-apidoc \
    --force \
    --separate \
    --templatedir "source/templates" \
    --output-dir "source" \
    "../kolena"
    # add any directories here, e.g. ../kolena/new_module, to exclude before module is released

echo "generating HTML pages..."
make clean
make html
