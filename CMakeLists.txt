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












#NOT WORKING:
#set(OpenMP_C_FLAGS "~/llvm-project/openmp")
#set(OpenMP_C_LIB_NAMES "openmp")

#find_package(OpenMP REQUIRED)
#find_package(PythonExtensions REQUIRED)

#---------------------------Fetch OpenMP and Eigen-------------------------
#include(FetchContent)
#set(FETCHCONTENT_QUIET OFF)
#message(STATUS "\nFetching OpenMP. This may take a few minutes...\n")
## Download and include OpenMP
#FetchContent_Declare(
#  openmp
#  URL https://github.com/llvm/llvm-project/archive/main.tar.gz
#)
#FetchContent_MakeAvailable(openmp)
## Specify OpenMP flags and libraries
#set(OpenMP_CXX_FLAGS ${openmp_CXX_FLAGS})
#set(OpenMP_CXX_LIBRARIES ${openmp_CXX_LIBRARIES})

#message(STATUS "\nFetching Eigen3. This may take a few minutes...\n")
## Download and include Eigen3
#FetchContent_Declare(
#  eigen
#  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
#)
#FetchContent_MakeAvailable(eigen)

#------------------------End Fetch OpenMP and Eigen-------------------------

#NOT NEEDED:

#message(STATUS "Setting CMake executable linker flags to link to LLVM libraries and python3.11.")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LLVM_LIBRARIES} -lpython3.11 -Wl,-rpath,${LLVM_LIBRARIES}")
#cmake_policy(SET CMP0057 NEW)

#find_package(OpenMP REQUIRED COMPONENTS CXX)
#if(OpenMP_FOUND)
#  message(STATUS "OpenMP found")
#  message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
#  message(STATUS "OpenMP CXX libraries: ${OpenMP_CXX_LIBRARIES}")
#else()
#  message(FATAL_ERROR "OpenMP not found")
#endif()

##Apple Specific Stuff
#
#set(CMAKE_OSX_SYSROOT ${MACOS_SDK})
#
#
## Set SDK directory
#execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE MACOS_SDK OUTPUT_STRIP_TRAILING_WHITESPACE)
#message(STATUS "Found macOS SDK at ${MACOS_SDK}")

