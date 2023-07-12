#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Download Eigen
curl -OL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip

# Unzip Eigen
unzip eigen-3.4.0.zip

# Move Eigen headers to /usr/local/include to make them available system-wide
cp -r eigen-3.4.0/Eigen /usr/local/include/

# Set CC and CXX environment variables to clang
export CC=$(which gcc)
export CXX=$(which g++)
export LOCAL_INCLUDE_DIRS=/usr/local/include

# Print the environment variables
echo "CC: $CC"
echo "CXX: $CXX"
echo "LOCAL_INCLUDE_DIRS: $LOCAL_INCLUDE_DIRS"
