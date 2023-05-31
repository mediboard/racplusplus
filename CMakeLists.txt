cmake_minimum_required(VERSION 3.26.0)

project(RACplusplus)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Python3 REQUIRED Development)
find_package(Python3 REQUIRED Interpreter)

#Must have Eigen3
find_package(Eigen3 REQUIRED)

#Link to pybind11, create the python module shared library with pybind11
find_package(pybind11 REQUIRED)

# Use the pybind11_INCLUDE_DIRS variable to access the include directories
include_directories(${pybind11_INCLUDE_DIRS})
#add the racplusplus module
pybind11_add_module(racplusplus src/racplusplus.cpp)
#link the binder, pybind11, and racplusplus together
target_link_libraries(racplusplus PRIVATE ${pybind11_LIBRARIES})



