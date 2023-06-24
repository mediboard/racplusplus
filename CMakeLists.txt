cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(racplusplus VERSION "0.0.1")

message(STATUS "\n-- Setting LLVM and Python3 Directories...\n")
list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt")

# Set LLVM directories
find_package(LLVM REQUIRED CONFIG)
message(STATUS "-- -- Found LLVM ${LLVM_PACKAGE_VERSION}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# Set Python3.11 directories
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
message(STATUS "-- -- Found Python ${Python3_VERSION}")
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

message(STATUS "\n-- Setting C and C++ Compilers to llvm...\n")
set(ENV{CC} "${LLVM_TOOLS_BINARY_DIR}/clang")
set(ENV{CXX} "${LLVM_TOOLS_BINARY_DIR}/clang++")
set(CMAKE_CXX_COMPILER "${LLVM_TOOLS_BINARY_DIR}/clang++")
set(ENV{LDFLAGS} "-L${LLVM_LIBRARY_DIRS}")
set(ENV{CPPFLAGS} "-I${LLVM_INCLUDE_DIRS}")

#Set Opt, Warning Flags, Linker
message(STATUS "\n-- Setting C++ Warning and Optimization Flags, Setting Linker to lld...\n")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fuse-ld=lld -Wsign-compare -Wunreachable-code -fwrapv -Wall -fcolor-diagnostics -fansi-escape-codes -march=native")

#-------------------Apply OpenMP if Possible (FIXME: should be using find_package instead, but it is broken)------------
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fopenmp" COMPILER_FLAG_SUPPORTED)
if (COMPILER_FLAG_SUPPORTED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else ()
  message(WARNING "The compiler does not support -fopenmp. Not optimizing with OpenMP.")
endif ()
#-------------------------------------------------------------------------------------------------------------------

#-------------------required packages-------------------------
find_package(Python3 REQUIRED Development)
find_package(Python3 REQUIRED Interpreter)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)
#-------------------------------------------------------------

#-------------------------------------Include Directories----------------------------
#include python directory for Python.h header file
include_directories(${PYTHON311_INCLUDE_DIRS})
# Use the pybind11_INCLUDE_DIRS variable to access the include directories
include_directories(${pybind11_INCLUDE_DIRS})
#include eigen directories
include_directories(${EIGEN3_INCLUDE_DIRS} )
#------------------------------------------------------------------------------------

#-----------------------------------Make Library---------------------------------------
set(python_module_name _racplusplus)
pybind11_add_module(${python_module_name} MODULE
     src/racplusplus/_racplusplus.cpp
    )
target_link_libraries(${python_module_name} PRIVATE
    -L${PYTHON311_LIBRARIES}
    ${pybind11_LIBRARIES}
    c++
    c++abi
    Eigen3::Eigen
)
#----------------------------------------------------------------------------------------

#Install the Cmake Target in src directory specified in setup.py
install(TARGETS ${python_module_name}
        DESTINATION .) 