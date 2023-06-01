cmake_minimum_required(VERSION 3.0)
project(racplusplus)

#-------------------required packages-------------------------
find_package(Python3 REQUIRED Development)
find_package(Python3 REQUIRED Interpreter)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)
#-------------------------------------------------------------

#enable if blocks etc. in new cmake functionality
cmake_policy(SET CMP0057 NEW)

#-----------------------------CMAKE Vars and Flags------------------------------
#set CXX env variable to point to llvm clang++ (this dir is a ln to llvm's clang++ for me)
#ln -s /usr/local/Cellar/llvm/16.0.4/bin/clang-16 /usr/local/bin/clang++""
message(STATUS "Setting Env Vars and Flags..")

set(ENV{CC} "/usr/local/opt/llvm/bin/clang")
set(ENV{CXX} "/usr/local/opt/llvm/bin/clang++")
set(ENV{LDFLAGS} "-L/usr/local/opt/llvm/lib")
set(ENV{CPPFLAGS} "-I/usr/local/opt/llvm/include")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} - llvm-openmp -O3 -Wall -Wsign-compare -Wunreachable-code -fwrapv -fcolor-diagnostics -fansi-escape-codes -march=native -fopenmp")

find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} 
         ${OpenMP_EXE_LINKER_FLAGS}")
endif()
#------------------------------------------------------------------------------

#-------------------------------------Include Directories----------------------------
#include python directory for Python.h header file
include_directories(/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11)
# Use the pybind11_INCLUDE_DIRS variable to access the include directories
include_directories(${pybind11_INCLUDE_DIRS})
#------------------------------------------------------------------------------------

#Built a racplusplus library target 
pybind11_add_module(racplusplus src/racplusplus.cpp)

#--------------------------------Linking Options + Libraries--------------------------------
target_link_options(racplusplus PRIVATE
    -fuse-ld=lld
    -lc++
    -lc++abi
    -lomp
    -L/Users/danielfrees/llvm-project/llvm/lib
    -L/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib
    -Wl,-rpath,/Users/danielfrees/llvm-project/llvm/lib
)
#link python3.11 library for python symbols at link time
target_link_libraries(racplusplus PRIVATE
    python3.11
)
#link the binder, pybind11, and racplusplus together
target_link_libraries(racplusplus PRIVATE ${pybind11_LIBRARIES})
#----------------------------------------------------------------------------------------------


