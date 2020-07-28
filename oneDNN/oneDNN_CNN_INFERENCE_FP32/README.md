# oneDNN CNN FP32 Inference sample
This oneDNN CNN FP32 Inference sample is implemented using C++ . By using DPC++ as the backend/engine, this example can also run on both CPU and GPU.

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; Windows 10
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++ Compiler, Intel oneAPI DPC++ Library (oneDPL), Intel oneAPI Threading Building Blocks (oneTBB), GNU Compiler , Intel C++ Compiler
| What you will learn               | run a simple CNN on both Intel CPU and GPU with sample C++ codes.
| Time to complete                  | 15 minutes

oneDNN_CPU2GPU_Porting.ipynb is also included.
This Jupyter Notebook demonstrates how to port a oneDNN sample from CPU-only version to CPU&GPU version by using DPC++ on the Intel oneAPI DevCloud (check below Notice)
>  Notice : Please use Intel oneAPI DevCloud as the environment for jupyter notebook samples. \
Users can refer to [DevCloud Getting Started](https://devcloud.intel.com/oneapi/get-started/) for using DevCloud \
Users can use JupyterLab from DevCloud via "One-click Login in", and download samples via "git clone" or the "oneapi-cli" tool \
Once users are in the JupyterLab with downloaded jupyter notebook samples, they can start following the steps without further installion needed.

## License
This code sample is licensed under MIT license



## What you will learn
* How to run a simple FP32 CNN network on different Intel CPU and GPU.

* How to compile examples with different compilers like DPC++, Intel Compiler, and GNU compiler

* How to switch between OpenMP and TBB for CPU parallelization

* How tensors are implemented and submitted to primitives.

* How primitives are created.

* How primitives are sequentially submitted to the network, where the output from primitives is passed as input to the next primitive. The latter specifies a dependency between the primitive input and output data.

* Specific 'inference-only' configurations.

* Limiting the number of reorders performed that are detrimental to performance.



## Pre-requirement


##### using Intel C++ Compiler

-----

Using Intel C++ Compiler also requires the following component which is part of the [Intel oneAPI HPC Toolkit (HPC Kit)](https://software.intel.com/en-us/oneapi/hpc-kit)
*  oneAPI Intel C++ Compiler


##### using TBB for CPU parallelization

-----

Using Threading Building Blocks also requires the following component which is part of the [Intel oneAPI Base Toolkit (Base Kit)](https://software.intel.com/en-us/oneapi/oneapi-kit)
*  Intel oneAPI Threading Building Blocks (oneTBB)


### GPU and CPU

-----

The sample below require the following components which are part of the [Intel oneAPI Base Toolkit (Base Kit)](https://software.intel.com/en-us/oneapi/oneapi-kit)
*  Intel oneAPI Deep Neural Network Library (oneDNN)
*  Intel oneAPI DPC++ Compiler
*  Intel oneAPI DPC++ Library (oneDPL)
*  Intel oneAPI Threading Building Blocks (oneTBB)

The sample also requires OpenCL driver. Please refer [System Requirements](https://software.intel.com/en-us/articles/intel-oneapi-base-toolkit-system-requirements) for OpenCL driver installation.


You can refer to this page [oneAPI](https://software.intel.com/en-us/oneapi) for toolkit installation.



## How to Build

### CPU
------

#### Using GNU C++ Compiler

------

By using GNU C++ compiler, this sample supports CNN FP32 on Intel CPU and it uses GNU OpenMP for CPU parallelization.

##### on Linux

- Build CNN Inference example with GCC

 First, please use a clean console environment without exporting any none default environment variables.
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_gomp
```
  Make sure that both the enviroments of compiler and oneDNN are properly set up before you process following steps.
  If setvars.sh complains "not found" for compiler or oneDNN, please check your installation first.

```
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir cpu_gomp
cd cpu_gomp
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
make cnn-inference-f32-cpp
```

> NOTE: The source file "cnn_inference_f32.cpp" will be copied from ${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/cpu_gomp to cpu_gomp/src folder.
Users can rebuild the cnn_inference_f32.cpp by typing "make" under cpu_gomp folder.

#### Using Intel C++ Compiler

------

By using Intel C++ compiler, this sample supports CNN FP32 on Intel CPU with more compiler optimization than GNU C++ compiler and it uses Intel OpenMP for CPU parallelization.

##### on Linux

- Build CNN Inference example with ICC

 First, please use a clean console environment without exporting any none default environment variables.
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_iomp
```
  Make sure that both the enviroments of compiler and oneDNN are properly set up before you process following steps.
  If setvars.sh complains "not found" for compiler or oneDNN, please check your installation first.

```
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir cpu_iomp
cd cpu_iomp
cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
make cnn-inference-f32-cpp
```

> NOTE: The source file "cnn_inference_f32.cpp" will be copied from ${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/cpu_iomp to cpu_iomp/src folder.
Users can rebuild the cnn_inference_f32.cpp by typing "make" under cpu_iomp folder.

#### Using Intel Threading Building Blocks

------

oneDNN supports both Intel OpenMP and TBB for CPU parallelization.
Users can switch to TBB from OpenMP by below steps.

##### on Linux

- Build CNN Inference example with Intel TBB

 First, please use a clean console environment without exporting any none default environment variables.
```
source ${INTEL_ONEAPI_INSTALL_FOLDER}/setvars.sh --dnnl-configuration=cpu_tbb
```

  Make sure that both the enviroments of compiler and oneDNN are properly set up before you process following steps.
  If setvars.sh complains "not found" for compiler or oneDNN, please check your installation first.

```
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir cpu_tbb
cd cpu_tbb
cmake ..
make cnn-inference-f32-cpp
```

> NOTE: The source file "cnn_inference_f32.cpp" will be copied from ${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/cpu_tbb to cpu_tbb/src folder.
Users can rebuild the cnn_inference_f32.cpp by typing "make" under cpu_tbb folder.

#### Using Microsoft Visual Studio version 2015 or Newer

------

By using MSVC compiler, this sample supports CNN FP32 on Intel CPU and it uses GNU OpenMP for CPU parallelization.

##### on Windows 10

- Build CNN Inference example with MSVC

 First, please open a Intel oneAPI command prompt for Microsoft Visual Studio.
```
C:\Program Files (x86)\intel\oneapi> oneDNN\latest\env\vars.bat --dnnl-configuration=cpu_vcomp
```
  Make sure that both the enviroments of compiler and oneDNN are properly set up before you process following steps.

```
cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
mkdir cpu_vcomp
cd cpu_vcomp
cmake -G "Visual Studio 16 2019" ..
cmake --build .
```

> NOTE: Users can open the oneDNN_CNN.sln inside cpu_vcomp folder to edit source code via MSVC IDE

## CPU and GPU

------

#### Using DPC++ Compiler

------

By using DPC++ compiler, this sample supports CNN FP32 both on Intel CPU and GPU.

##### on Linux

   - Build CNN Inference example with DPC++ compiler

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
    cd oneapi-toolkit/oneDNN/oneDNN_CNN_INFERENCE_FP32
    mkdir dpcpp
    cd dpcpp
    cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp
    make cnn-inference-f32-cpp
```

> NOTE: The source file "cnn_inference_f32.cpp" will be copied from ${INTEL_ONEAPI_INSTALL_FOLDER}/oneDNN/latest/dpcpp to dpcpp/src folder.
Users can rebuild the cnn_inference_f32.cpp by typing "make" under dpcpp folder.

## How to Run

### on Linux


- Run the program  on CPU
```
    ./out/cnn-inference-f32-cpp
```
- Run the program  on GPU

```
    ./out/cnn-inference-f32-cpp gpu
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
    out\Debug\cnn-inference-f32-cpp.exe
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
...
/oneDNN VERBOSE LOGS/
...
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,1x256x6x6,0.032959
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:abcd:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic256ih6iw6oc4096,5.4458
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc4096,2.50317
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc1000,0.634033
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ab:f0,,,1x1000,0.0290527
Use time 33.22
```

- GPU Results

```
dnnl_verbose,info,DNNL v1.90.1 (commit 9151ddc657e4c6775f17f3bcec46872e5fac47ee)
dnnl_verbose,info,Detected ISA is Intel AVX2
...
/DNNL VERBOSE LOGS/
...
dnnl_verbose,exec,gpu,reorder,ocl:simple:any,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,1x256x6x6
dnnl_verbose,exec,gpu,inner_product,ocl:gemm,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:abcd:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic256ih6iw6oc4096
dnnl_verbose,exec,gpu,inner_product,ocl:gemm,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc4096
dnnl_verbose,exec,gpu,inner_product,ocl:gemm,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc1000
dnnl_verbose,exec,gpu,reorder,ocl:simple:any,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ab:f0,,,1x1000
Use time 106.29
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
...
/DNNL VERBOSE LOGS/
...
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:aBcd8b:f0 dst_f32::blocked:abcd:f0,,,1x256x6x6,0.032959
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:abcd:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic256ih6iw6oc4096,5.4458
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc4096,2.50317
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ab:f0 bia_f32::blocked:a:f0 dst_f32::blocked:ab:f0,,,mb1ic4096oc1000,0.634033
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ab:f0,,,1x1000,0.0290527
Use time 33.22
```


## Implementation Details

  This example is from oneDNN project, and you can refer to [Line by Line Explanation](https://oneapi-src.github.io/oneDNN/cnn_inference_f32_cpp.html) for implementation details.
