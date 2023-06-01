Current working clang++ compilation command: 

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
