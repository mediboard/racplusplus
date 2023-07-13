#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Download Eigen
curl -OL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip

# Unzip Eigen
unzip eigen-3.4.0.zip

# Create build directory
mkdir eigen-3.4.0/build
cd eigen-3.4.0/build

# Configure
cmake ..

# Install
make install

# Get the Python interpreter path
PYTHON_PATH=$(which python)

# Install pybind11 using the correct Python interpreter
$PYTHON_PATH -m pip install pybind11

# Move Eigen headers to /usr/local/include to make them available system-wide
# cp -r eigen-3.4.0/Eigen /usr/local/include/

# Set CC and CXX environment variables to clang
export CC=$(which clang)
export CXX=$(which clang++)

# Print the environment variables
echo "CC: $CC"
echo "CXX: $CXX"

# export EIGEN3_INCLUDE_DIR=/usr/local/include/eigen3
