cmake_minimum_required(VERSION 3.26.0)

project(RACplusplus)

set(CMAKE_CXX_STANDARD 17)

#Must have Eigen3
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen version ${Eigen3_VERSION}: ${Eigen3_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Eigen not found. Please install Eigen.")
endif()

#Link to pybind11, create the python module shared library with pybind11
find_package(pybind11 REQUIRED)
if(pybind11_FOUND)
    # Use the pybind11_INCLUDE_DIRS variable to access the include directories
    include_directories(${pybind11_INCLUDE_DIRS})
    set(SOURCES
        src/racplusplus.cpp
    )
    pybind11_add_module(racplusplus ${SOURCES})
    target_link_libraries(racplusplus PRIVATE ${pybind11_LIBRARIES})
else()
    message(FATAL_ERROR "pybind11 not found. Please install pybind11.")
endif()


