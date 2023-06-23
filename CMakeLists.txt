cmake_minimum_required(VERSION 3.7)
project(racplusplus VERSION "0.0.1")

message(STATUS "Setting LLVM, Python3.11, SDK Directories")
#----------------------------set directories------------------------
set(LLVM_LIBRARIES /Users/danielfrees/llvm-project/llvm/lib)
set(PYTHON311_LIBRARIES /usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib)
set(PYTHON311_INCLUDE_DIRS /usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11)
set(MACOS_SDK /Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk)
#----------------------------end set directories------------------------

message(STATUS "Setting C and C++ Compilers to llvm...")
set(ENV{CC} "/usr/local/opt/llvm/bin/clang")
set(ENV{CXX} "/usr/local/opt/llvm/bin/clang++")
set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++) #needed to find lld linker
set(ENV{LDFLAGS} "-L/usr/local/opt/llvm/lib")
set(ENV{CPPFLAGS} "-I/usr/local/opt/llvm/include")

message(STATUS "Setting CMake Standard to 17...")
set(CMAKE_CXX_STANDARD 17)

message(STATUS "Setting C++ Warning and Optimization Flags, Setting Linker to lld...")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fuse-ld=lld -Wsign-compare -Wunreachable-code -fwrapv -Wall -fcolor-diagnostics -fansi-escape-codes -march=native")

message(STATUS "Setting CMake executable linker flags to link to LLVM libraries and python3.11.")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LLVM_LIBRARIES} -lpython3.11 -Wl,-rpath,${LLVM_LIBRARIES}")

set(CMAKE_OSX_SYSROOT ${MACOS_SDK})

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
    #OpenMP::OpenMP_CXX
)
#----------------------------------------------------------------------------------------

#Set target
# Specify the files or artifacts to be installed
install(TARGETS ${python_module_name}
        DESTINATION .) #will install in the src directory specified in setup.py

#NOT WORKING:

#set(OpenMP_C_FLAGS "~/llvm-project/openmp")
#set(OpenMP_C_LIB_NAMES "openmp")

#find_package(OpenMP REQUIRED)
#find_package(PythonExtensions REQUIRED)