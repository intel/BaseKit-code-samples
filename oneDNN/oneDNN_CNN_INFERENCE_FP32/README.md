# oneDNN CNN FP32 Inference sample
This oneDNN CNN FP32 Inference sample is implemented using C++ . By using DPC++ as the backend/engine, this example can also run on both CPU and GPU. 

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; 
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel oneAPI Deep Neural Network Library (oneDNN), Intel oneAPI DPC++ Compiler, Intel oneAPI DPC++ Library (oneDPL), Intel oneAPI Threading Building Blocks (oneTBB), GNU Compiler , Intel C++ Compiler
| What you will learn               | run a simple CNN on both Intel CPU and GPU with sample C++ codes.
| Time to complete                  | 15 minutes


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

#### Using Intel Threading Building Block  

------

DNNL supports both Intel OpenMP and TBB for CPU parallelization.
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
/DNNL VERBOSE LOGS/
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

## Implementation Details

  This example is from oneDNN project, and you can refer to [ Line by Line Explanation ](https://intel.github.io/mkl-dnn/cnn_inference_f32_cpp.html) for implementation details. 
