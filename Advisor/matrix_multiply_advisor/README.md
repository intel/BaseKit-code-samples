# matrix multiply sample
A sample containing multiple implementations of matrix multiplication. This sample code is implemented using C++ and SYCL language for CPU and GPU. 
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; Windows 10
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel(R) oneAPI DPC++ Compiler (beta); Intel C++ Compiler xxx beta; Intel(R) VTune(TM) Profiler and Intel(R) Advisor
| What you will learn               | How to profile an application using Intel(R) VTune(TM) Profiler and Intel(R) Advisor
| Time to complete                  | 15 minutes

 

## License  
This code sample is licensed under MIT license

## How to Build  

This sample contains 5 version of matrix multiplication using DPC++:

    multiply1 – basic implementation of matrix multiply using DPC++
    multiply1_1 – basic implementation that replaces the buffer store with a local accessor “acc” to reduce memory traffic
    multiply1_2 – basic implementation plus the local accessor and matrix tiling
    multiply2 – cache-blocked matrix multiply using sub-ranges
    multiply2_1 – cache-blocked matrix multiply using sub-ranges + local accessor

Edit the line in multiply.h to select the version of the multiply function:
#define MULTIPLY multiply1

Ensure the variable DPCPP is definded at build time to use the DPC++ functions

NOTE: The multiply1_1, multiply1_2, and multiply2_1 versions are only available in the DPC++ version (i.e. when DPCPP is defined).

### on Linux  
	To build DPC++ version:
	cd <sample dir>
	cmake .
	make 
	
	To build other versions using included Makefile
	cd <sample dir>/linux
	make <clang/gcc/icc/mkl>

    Clean the program  
    make clean  

### on Windows - Visual Studio 2017 or newer
   * Open Visual Studio 2017
   * Select Menu "File > Open > Project/Solution", find "matrix_multiply" folder and select "matrix_multiply.sln"
   * Select Menu "Project > Build" to build the selected configuration
   * Select Menu "Debug > Start Without Debugging" to run the program
   * Define "DPCPP" or "USE_THR" in Project Properties > DPC++ > Preprocessor > Preprocessor Definition to choose a DPC++ or threaded version of the sample

### on Windows - command line - Build the program using MSBuild
    DPCPP Configurations:
    Release - MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Release"
    Debug - MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Debug"
    USE_THR Configurations:
    Release - MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Release_THR"
    Debug - MSBuild matrix_multiply.sln /t:Rebuild /p:Configuration="Debug_THR"   


## Running an Intel Advisor analysis
------------------------------------------

See the Advisor Cookbook here: https://software.intel.com/en-us/advisor-cookbook

