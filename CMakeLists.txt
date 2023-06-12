cmake_minimum_required(VERSION 3.0)
project(racplusplus)

#enable if blocks etc. in new cmake functionality
cmake_policy(SET CMP0057 NEW)

#-------------------required packages-------------------------
find_package(Python3 REQUIRED Development)
find_package(Python3 REQUIRED Interpreter)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)
#-------------------------------------------------------------

#----------------------------set directories------------------------
set(LLVM_LIBRARIES /Users/porterhunley/opt/anaconda3/envs/racpp_native/lib)
set(PYTHON311_LIBRARIES /Users/porterhunley/opt/anaconda3/envs/racpp_native/lib/python3.11)
set(PYTHON311_INCLUDE_DIRS /Users/porterhunley/opt/anaconda3/envs/racpp_native/include/python3.11)
set(MACOS_SDK /Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk)
#----------------------------end set directories------------------------

set(ENV{CXX} "/Users/porterhunley/opt/anaconda3/envs/racpp_native/bin/clang++")
set(CMAKE_CXX_COMPILER /Users/porterhunley/opt/anaconda3/envs/racpp_native/bin/clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wsign-compare -Wunreachable-code -fwrapv -Wall -fcolor-diagnostics -fansi-escape-codes -mcpu=native -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LLVM_LIBRARIES} -L${PYTHON311_LIBRARIES} -lpython3.11 -Wl,-rpath,${LLVM_LIBRARIES}")
set(CMAKE_OSX_SYSROOT ${MACOS_SDK})

#-------------------------------------Include Directories----------------------------
#include python directory for Python.h header file
include_directories(${PYTHON311_INCLUDE_DIRS})
# Use the pybind11_INCLUDE_DIRS variable to access the include directories
include_directories(${pybind11_INCLUDE_DIRS})
#include eigen directories
include_directories( ${EIGEN3_INCLUDE_DIRS} )
#------------------------------------------------------------------------------------

#------------------------------------Make Executable----------------------------------
add_executable(racplusplus_exe src/racplusplus.cpp)
target_link_libraries(racplusplus_exe 
    c++ 
    c++abi 
)
#--------------------------------------------------------------------------------------

#-----------------------------------Make Library---------------------------------------
pybind11_add_module(racplusplus src/racplusplus.cpp)
target_link_libraries(racplusplus PRIVATE
    -L${PYTHON311_LIBRARIES}
    c++
    c++abi
    ${pybind11_LIBRARIES}
)
#----------------------------------------------------------------------------------------
