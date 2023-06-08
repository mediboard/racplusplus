**Guide: Building the wrapper package.**

*Prereqs: C++, llvm clang++, openmp, llvm lld linker, macOS 13 (13.4)*

`1.` Make a new conda environment for racplusplus.

*On the env:*
`2.` Install necessary packages (numpy, pandas, pybind11, jupyter, scipy,  etc. lmk if you’d like my full “conda list” or a yaml for my condo environment)
`3.` Upgrade python to 3.11.
`4.` Make sure ipykernel is updated to 3.11 as well if you want to work out of a notebook:
python3.11 -m pip install ipykernel
python3.11 -m ipykernel install --user --name=python311

*On your filesystem:*
`5.` Create a folder for racplusplus.
`6.` Clone into my GitHub repo https://github.com/mediboard/racplusplus 
`7.` Change your symbolic links in this folder via `ln`.	Location for pybind11 for me: “/Users/danielfrees/miniconda3/envs/racplusplus/lib/python3.11/site-packages/pybind11”
	Location for Eigen for me: “/usr/local/include/eigen3/Eigen”

*Building the package and executable:*
`8.` Update the 4 vars at the top of the CMake:
	- set(LLVM_LIBRARIES /Users/danielfrees/llvm-project/llvm/lib)
	- set(PYTHON311_LIBRARIES /usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib)
	- set(PYTHON311_INCLUDE_DIRS /usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11)
	- set(MACOS_SDK /Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk)
`9.` mkdir build
`10.` cd build
`11.` cmake ..
`12.` make racplusplus (or make racplusplus_exe if you want the simple fully wrapped RAC test executable)
`13.` Start a python environment
	- Open a python3.11 shell or
	- Run a python script with python3.11 (Make sure to import racplusplus before bumpy due to a glitch with OpenMP linkage) or 
	- Run a jupyter notebook (with kernel = python311) [NOT YET WORKING]
	To make the kernel: 
	`python -m ipykernel install --user --name python3.11 --display-name "Python 3.11”`

`14.` Import racplusplus and have fun!
`15.` See "test/test_wrapper.py" for example usage. Example output running my python testing script:

```bash
~/Desktop/racplusplus  main ✗                                            0m ⚑
▶ python3 test/test_wrapper.py
Sys Path: ['/Users/danielfrees/Desktop/racplusplus/test', '/Users/danielfrees/miniconda3/envs/racplusplus/lib/python311.zip', '/Users/danielfrees/miniconda3/envs/racplusplus/lib/python3.11', '/Users/danielfrees/miniconda3/envs/racplusplus/lib/python3.11/lib-dynload', '/Users/danielfrees/miniconda3/envs/racplusplus/lib/python3.11/site-packages', '/Users/danielfrees/Desktop/racplusplus/test/test_wrapper.py/../../build', '/Users/danielfrees/Desktop/racplusplus/build']
Python Version: 3.11.3 (main, Apr 19 2023, 18:51:09) [Clang 14.0.6 ]

First Python racplusplus package test (Basic IO):

This is a simple pybind I/O Test.

Basic IO test complete.

Second Python racplusplus package test (Fully wrapped RAC):

Starting Randomized RAC Test
Number of Processors Found for Program Use: 4
Actually running RAC now...
Time taken to calculate initial dissimilarities: 5412ms
RAC Finished!
9996
0
0
0
0
0
838

Fully wrapped RAC test complete.

Third Python racplusplus package test (Numpy & Scipy <> RAC Interface):

Generating sparse unweighted connectivity matrix (This takes around 20 seconds for size 10k x 10k)...
Done generating sparse unweighted connectivity matrix.

Running RAC from Python using numpy data matrix and scipy sparse csc connectivity matrix.
Time taken to calculate initial dissimilarities: 9180ms
Point Cluster Assignments: [   0    1    2 ... 2353 9998 9999]
Numpy & Scipy <> RAC Interface test complete.

Fourth Python racplusplus package test (Numpy & Scipy <> RAC Interface w/o Conn Matrix):
Running RAC from Python using numpy data matrix and empty scipy sparse lil connectivity matrix.
Time taken to calculate initial dissimilarities: 4480ms
Point Cluster Assignments: [   0    1    2 ... 9997 9998 9999]
Numpy & Scipy <> RAC Interface w/o Conn Matrix test complete.

Python Script racplusplus package test complete!
```

*For development:* After the initial setup all you’ll need to do if you edit the C++ package is save edits and repeat steps 11-14 to see your changes reflected.


*Additional notes:*

Current working clang++ compilation command below.

```bash
/usr/local/bin/clang++ src/racplusplus.cpp -o build/racplusplus -O3 
-Wsign-compare -Wunreachable-code -fwrapv -Wall -lc++ -lc++abi -lomp 
-fuse-ld=lld -fcolor-diagnostics -fansi-escape-codes -std=c++17 
-march=native -fopenmp -L/Users/danielfrees/llvm-project/llvm/lib 
-L/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib 
-lpython3.11 -Wl,-rpath,/Users/danielfrees/llvm-project/llvm/lib 
-I/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11  
-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk
```
