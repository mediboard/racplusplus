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
set(LLVM_LIBRARIES /Users/danielfrees/llvm-project/llvm/lib)
set(PYTHON311_LIBRARIES /usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib)
set(PYTHON311_INCLUDE_DIRS /usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11)
set(MACOS_SDK /Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk)
#----------------------------end set directories------------------------

set(ENV{CXX} "/usr/local/bin/clang++")
set(CMAKE_CXX_COMPILER /usr/local/bin/clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wsign-compare -Wunreachable-code -fwrapv -Wall -fuse-ld=lld -fcolor-diagnostics -fansi-escape-codes -march=native -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LLVM_LIBRARIES} -L${PYTHON311_LIBRARIES} -lpython3.11 -Wl,-rpath,${LLVM_LIBRARIES}")
set(CMAKE_OSX_SYSROOT ${MACOS_SDK})

#-------------------------------------Include Directories----------------------------
#include python directory for Python.h header file
include_directories(${PYTHON311_INCLUDE_DIRS})
# Use the pybind11_INCLUDE_DIRS variable to access the include directories
include_directories(${pybind11_INCLUDE_DIRS})
#------------------------------------------------------------------------------------

#------------------------------------Make Executable----------------------------------
add_executable(racplusplus_exe src/racplusplus.cpp)
target_link_libraries(racplusplus_exe 
    c++ 
    c++abi 
    omp
)
#--------------------------------------------------------------------------------------

#-----------------------------------Make Library---------------------------------------
pybind11_add_module(racplusplus src/racplusplus.cpp)
target_link_libraries(racplusplus PRIVATE
    -L${PYTHON311_LIBRARIES}
    c++
    c++abi
    omp
    ${pybind11_LIBRARIES}
)
#----------------------------------------------------------------------------------------