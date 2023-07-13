#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

BUILD_IDENTIFIER=${CIBW_BUILD}
if [[ $BUILD_IDENTIFIER == *"_x86_64"* ]]; then
  ARCH="x86_64"
elif [[ $BUILD_IDENTIFIER == *"_arm64"* ]]; then
  ARCH="arm64"
else
  echo "Unknown architecture"
  exit 1
fi

OMP_BUILD_DIR="openmp-12.0.1.src/build_${ARCH}"

# Download and unpack OpenMP source code
curl -OL https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/openmp-12.0.1.src.tar.xz
tar xvf openmp-12.0.1.src.tar.xz

# Create build directory and navigate to it
mkdir ${OMP_BUILD_DIR}
cd ${OMP_BUILD_DIR}

# Configure and build OpenMP
cmake ..
make

# Install OpenMP
make install

# Exit build directory
cd ../../..

# Download Eigen
curl -OL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip

# Unzip Eigen
unzip eigen-3.4.0.zip

EIGEN_BUILD_DIR="eigen-3.4.0/build_${ARCH}"

# Create build directory
mkdir ${EIGEN_BUILD_DIR} 
cd ${EIGEN_BUILD_DIR} 

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
