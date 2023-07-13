cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(racplusplus VERSION "0.0.5")

# Set the OpenMP flags here
if(DEFINED ENV{OpenMP_LIBRARY})
    message(STATUS "OpenMP_LIBRARY = $ENV{OpenMP_LIBRARY}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
    include_directories("/usr/local/include")
endif()

message(STATUS "\n-- Finding Prefix Paths...\n")
list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt")
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Loaded environment variable: CONDA_PREFIX=${CONDA_PREFIX}")
else()
    message(STATUS "Environment variable CONDA_PREFIX is not set.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${CONDA_PREFIX})


message(STATUS "\n-- Setting LLVM and Python3 Directories...\n")
# Set LLVM directories
# find_package(LLVM REQUIRED CONFIG)
# message(STATUS "-- -- Found LLVM ${LLVM_PACKAGE_VERSION}")
# list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# include_directories(${LLVM_INCLUDE_DIRS})
# link_directories(${LLVM_LIBRARY_DIRS})

# Set Python3.11 directories
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ${Python_SITEARCH})
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

message(STATUS "-- -- Found Python ${Python3_VERSION}")
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# Set compilers
message(STATUS "\n-- Setting C and C++ Compilers to llvm...\n")
# print cc and cxx env variables
message(STATUS "CC: $ENV{CC}")
message(STATUS "CXX: $ENV{CXX}")

# set(ENV{CC} "${CONDA_PREFIX}/bin/clang")
# set(ENV{CXX} "${CONDA_PREFIX}/bin/clang++")
set(CMAKE_CXX_COMPILER "$ENV{CXX}")
# set(ENV{LDFLAGS} "-L${CONDA_PREFIX}/lib")
set(ENV{CPPFLAGS} "-I${LOCAL_INCLUDE_DIRS}")

# Print out compiler information
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

#Set Opt, Warning Flags, Linker
message(STATUS "\n-- Setting C++ Warning and Optimization Flags, Setting Linker to lld...\n")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wsign-compare -Wunreachable-code -fwrapv -Wall")

#-------------------Apply OpenMP if Possible (FIXME: should be using find_package instead, but it is broken)------------
# I can add the flag fine??
# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag("-fopenmp" COMPILER_FLAG_SUPPORTED)
# if (COMPILER_FLAG_SUPPORTED)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# else ()
#   message(WARNING "The compiler does not support -fopenmp. Not optimizing with OpenMP.")
# endif ()
#-------------------------------------------------------------------------------------------------------------------

execute_process(
    COMMAND bash -c "pip show pybind11"
    OUTPUT_VARIABLE PYBIND11_OUTPUT
)

message(STATUS "Pybind11 pip show output: ${PYBIND11_OUTPUT}")

#-------------------required packages-------------------------
find_package(Python3 REQUIRED Development.Module)
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

#-----------------------------------Link Libraries-------------------------------------
target_link_libraries(${python_module_name} PRIVATE
    -L${PYTHON311_LIBRARIES}
    ${pybind11_LIBRARIES}
    c++
    Eigen3::Eigen)
#----------------------------------------------------------------------------------------

#Install the Cmake Target in src directory specified in setup.py
install(TARGETS ${python_module_name}
        DESTINATION .) 