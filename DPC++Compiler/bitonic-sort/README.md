# Bitonic Sort sample
This code sample demonstrates the implementation of bitonic sort using Intel Data Parallel C++ to
offload the computation to the kernel. In this implementation, a random sequence of 2**n is given
(n is a positive number). Unified Shared Memory (USM) is used for data management.


For comprehensive instructions regarding DPC++ Programming, go to
https://software.intel.com/en-us/oneapi-programming-guide
and search based on relevant terms noted in the comments.

  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta); Intel C++ Compiler (beta)
| What you will learn               | Implement bitonic sort using Intel DPC++ compiler
| Time to complete                  | 15 minutes


## License  
This code sample is licensed under MIT license  

## How to Build  

### on a Linux* System  
   * Build bitonic-sort
    
    cd bitonic-sort &&  
    mkdir build &&
    cd build &&
    cmake .. &&
    make -j

   * Run the program

    make run  
   
   * Clean the program  
    make clean

### on Windows
    * Build the program using VS2017 or VS2019
      Right click on the solution file and open using either VS2017 or VS2019 IDE.
      Right click on the project in Solution explorer and select Rebuild.
      From top menu select Debug -> Start without Debugging.

    * Build the program using MSBuild
      Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for
 VS2019"
      Run - MSBuild bitonic-sort.sln /t:Rebuild /p:Configuration="Release"

## How to Run  
   * Application Parameters   
	
        Usage: bitonic-sort <exponent> <seek>

