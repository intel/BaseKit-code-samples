# oneDNN Getting Started sample
 This oneDNN Getting Started sample code is implemented using C++ for CPU. 

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; 
| Hardware                          | Intel CPU
| Software                          | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++ Compiler, Intel oneAPI DPC++ Library (oneDPL), Intel oneAPI Threading Building Blocks (oneTBB)
| What you will learn               | basic oneDNN programming model for Intel CPU and GPU
| Time to complete                  | 15 minutes



## What you will learn
* How to create oneDNN memory objects.
* How to get data from user's buffer into a oneDNN memory object.
* How tensor's logical dimensions and memory object formats relate.
* How to create oneDNN primitives.
* How to execute the primitives.

## Pre-requirement

The sample below require the following components, which are part of the Intel oneAPI Base Toolkit (Base Kit)

* Intel oneAPI Deep Neural Network Library (oneDNN)
* Intel oneAPI DPC++ Compiler
* Intel oneAPI DPC++ Library (oneDPL)
* Intel oneAPI Threading Building Blocks (oneTBB)

The sample also requires OpenCL driver. Please refer System Requirements for OpenCL driver installation.

You can refer to this page [oneAPI](https://software.intel.com/en-us/oneapi) for toolkit installation.


## How to Build  

#### Using DPC++ Compiler 

------

By using DPC++ compiler, this sample demostrates basic oneDNN operations on Intel CPU and Intel GPU.

##### on Linux  

- Build Getting Started example with DPC++  

 First, please use a clean console environment without exporting any none default environment variables.
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_dpcpp_gpu_dpcpp
```
or
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh
```
  dnnl-configuration is set cpu_dpcpp_gpu_dpcpp to by default if users don't input --dnnl-configuraition argument.
  
  Make sure that both the enviroments of compiler and oneDNN are properly set up before you process following steps.
  If setvars.sh complains "not found" for compiler or oneDNN, please check your installation first.
  
```   
cd oneapi-toolkit/oneDNN/oneDNN_Getting_Started
mkdir dpcpp
cd dpcpp
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp
make getting-started-cpp
```

## How to Run  

### on Linux  
- Run the program  on CPU
  ```
  ./out/getting-started-cpp cpu
  ```
- Run the program  on GPU
  ```
  ./out/getting-started-cpp gpu
  ```



## Result Validation 

### on Linux  

- Enable oneDNN Verbose log

  ```
  export DNNL_VERBOSE=1
  ```

- Run the program on CPU or GPU following [How to Run Session](#how-to-run)

- CPU Results
  ```
  dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
  dnnl_verbose,info,Detected ISA is Intel AVX2
  dnnl_verbose,exec,cpu,eltwise,jit:avx2,forward_inference,data_f32::blocked:acdb:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,1x3x13x13,704.982
  Example passes
  ```

- GPU Results
  ```
  dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
  dnnl_verbose,info,Detected ISA is Intel AVX2
  dnnl_verbose,exec,gpu,eltwise,ocl:ref:any,forward_inference,data_f32::blocked:acdb:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,1x3x13x13
  Example passes
  ```

## Implementation Details

  This example is from oneDNN project, and you can refer to [ Line by Line Explanation ](https://intel.github.io/mkl-dnn/getting_started_cpp.html) for implementation details. 
  
