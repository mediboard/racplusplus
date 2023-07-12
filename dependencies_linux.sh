#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update the package list
sudo apt-get update

# Install g++ and clang if they are not already installed
sudo apt-get install -y g++ clang

# Install libomp-dev (for OpenMP)
sudo apt-get install -y libomp-dev

# Install unzip if it's not already installed
sudo apt-get install -y unzip

# Download Eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip

# Unzip Eigen
unzip eigen-3.4.0.zip

# Move Eigen headers to /usr/local/include to make them available system-wide
sudo cp -r eigen-3.4.0/Eigen /usr/local/include/

# Install pip if it's not already installed
if ! command -v pip &> /dev/null
then
    sudo apt-get install -y python3-pip
fi

# Install pybind11
pip install pybind11

# Set CC and CXX environment variables to clang
export CC=$(which clang)
export CXX=$(which clang++)
export LOCAL_INCLUDE_DIRS=/usr/local/include

# Print the environment variables
echo "CC: $CC"
echo "CXX: $CXX"
echo "LOCAL_INCLUDE_DIRS: $LOCAL_INCLUDE_DIRS"
