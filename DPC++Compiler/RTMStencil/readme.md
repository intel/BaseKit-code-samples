# RTMStencil
Stencil computation is the basis for the Reverse Time Migration algorithm in seismic computing. The underlying mathematical problem is to solve the wave equation using finite difference method. This sample computes a 3D 25 points stencil. The computation contains 4 layer loops for each dimention and time duration. This nbody sample code is implemented using C++ and SYCL language for CPU and GPU. 
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)
| What you will learn               | How to offload the computation to GPU using Intel DPC++ compiler and OpenMP offload pragmas
| Time to complete                  | 15 minutes

Performance number tabulation [if applicable]

| RTMStencil sample                      | Performance data
|:---                               |:---
| Scalar baseline -O2               | 1.0
| SYCL                              | 2x speedup
| OpenMP offload                    | 2x speedup

  
## Key implementation details 
SYCL implementation explained. 
OpenMP offload implementation explained. 

## License  
This code sample is licensed under MIT license

Based on original code: Book "Structured Parallel Programming" by Michael McCool, Arch Robison, James Reinders
* source: http://parallelbook.com/sites/parallelbook.com/files/code_0.zip
* license: http://opensource.org/licenses/BSD-3-Clause  

## How to Build  

### on Linux  
   * Build nbbody program  
    cd RTMStencil &&  
    mkdir build &&  
    cd build &&  
    cmake ../. &&  
    make VERBOSE=1  

   * Run the program  
    make run  

   * Clean the program  
    make clean  

### on Windows - command line - Build the program using MSBuild
   MSBuild RTMStencil.sln /t:Rebuild /p:Configuration="debug"
   
### on Windows - Visual Studio 2017 or newer
   * Open Visual Studio 2017
   * Select Menu "File > Open > Project/Solution", find "RTMStencil" folder and select "RTMStencil.sln"
   * Select Menu "Project > Build" to build the selected configuration
   * Select Menu "Debug > Start Without Debugging" to run the program

