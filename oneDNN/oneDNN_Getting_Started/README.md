# oneDNN Getting Started sample
 This oneDNN Getting Started sample code is implemented using C++ for CPU.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; Windows 10
| Hardware                          | Intel CPU
| Software                          | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++ Compiler, Intel oneAPI DPC++ Library (oneDPL), Intel oneAPI Threading Building Blocks (oneTBB)
| What you will learn               | basic oneDNN programming model for Intel CPU and GPU
| Time to complete                  | 15 minutes

A oneDNN_Getting_Started.ipynb is also included.
This Jupyter Notebook demonstrates how to compile a oneDNN sample with different releases via batch jobs on the Intel oneAPI DevCloud (check below Notice)
>  Notice : Please use Intel oneAPI DevCloud as the environment for jupyter notebook samples. \
Users can refer to [DevCloud Getting Started](https://devcloud.intel.com/oneapi/get-started/) for using DevCloud \
Users can use JupyterLab from DevCloud via "One-click Login in", and download samples via "git clone" or the "oneapi-cli" tool \
Once users are in the JupyterLab with downloaded jupyter notebook samples, they can start following the steps without further installion needed.


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

> NOTE: The source file "getting_started.cpp" will be copied from ${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/dpcpp to dpcpp/src folder.
Users can rebuild the getting_started.cpp by typing "make" under dpcpp folder.


#### Using Microsoft Visual Studio version 2015 or Newer

------

By using MSVC compiler, this sample supports CNN FP32 on Intel CPU and it uses GNU OpenMP for CPU parallelization.

##### on Windows 10

- Build Getting Started  example with MSVC

 First, please open a Intel oneAPI command prompt for Microsoft Visual Studio.
```
C:\Program Files (x86)\intel\oneapi> oneDNN\latest\env\vars.bat --dnnl-configuration=cpu_vcomp
```
  Make sure that both the enviroments of compiler and oneDNN are properly set up before you process following steps.

```
cd oneapi-toolkit/oneDNN/oneDNN_Getting_Started
mkdir cpu_vcomp
cd cpu_vcomp
cmake -G "Visual Studio 16 2019" ..
cmake --build .
```

> NOTE: Users can open the oneDNN_CNN.sln inside cpu_vcomp folder to edit source code via MSVC IDE

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
>  NOTE: Zero Level runtime is enabled by default. Please make sure proper installation of zero level driver \
including level-zero-devel package following installation guide. \
If users still encounter runtime issue such as "could not create a primitive", \
Please apply workaround to set SYCL_BE=PI_OPENCL before running a DPC++ program \
 \
For applying the workaround in this sample, users can add `export SYCL_BE=PI_OPENCL` in CMakeLists.txt. \
After applying the worklaround, sample use OpenCL runtime instead.\

### on Windows


- Run the program  on CPU
```
    out\Debug\getting-started-cpp.exe
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

### on Windows

- Enable oneDNN Verbose log

```
set DNNL_VERBOSE=1

```

- Run the program on CPU or GPU following [How to Run Session](#how-to-run)

- CPU Results
  ```
  dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
  dnnl_verbose,info,Detected ISA is Intel AVX2
  dnnl_verbose,exec,cpu,eltwise,jit:avx2,forward_inference,data_f32::blocked:acdb:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,1x3x13x13,704.982
  Example passes
  ```

## Implementation Details

  This example is from oneDNN project, and you can refer to [Line by Line Explanation](https://oneapi-src.github.io/oneDNN/getting_started_cpp.html) for implementation details.

