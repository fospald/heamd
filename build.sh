#!/bin/bash

HOST=$(uname -n | tr " -." "___")
HEAMD_DIR=$(cd $(dirname "$0") && pwd)

BUILD_DIR="$HEAMD_DIR/build/$HOST"
mkdir -p "$BUILD_DIR" || exit
cd "$BUILD_DIR" || exit
cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF "$@" ../.. || exit
make || exit

echo ""
echo "Build successful!"
echo "Please update your environmet variables, i.e.:"
echo "export PATH=\$PATH:$HEAMD_DIR/bin"
echo "export PYTHONPATH=\$PYTHONPATH:$HEAMD_DIR/lib"

