Below is a sample readme and all Samples for Intel(r) oneAPI are expected to follow this
# oneCCL Getting Started sample
Those CCL sample codes are implemented using C++, C and DPC++ language for CPU and GPU. 

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04; 
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel oneAPI Collective Communications Library (oneCCL), Intel oneAPI DPC++ Compiler, Intel oneAPI DPC++ Library (oneDPL), GNU Compiler
| What you will learn               | basic oneCCL programming model for both Intel CPU and GPU
| Time to complete                  | 15 minutes

## List of Samples
| C++ API | C API | Collective Operation |
| ------ | ------ | ------ |
| sycl_allgatherv_cpp_test.cpp  | sycl_allgatherv_test.cpp | [Allgatherv](https://intel.github.io/oneccl/spec/communication_primitives.html#allgatherv) |
| sycl_allreduce_cpp_test.cpp | sycl_allreduce_test.cpp |[Allreduce](https://intel.github.io/oneccl/spec/communication_primitives.html#allreduce) |
| sycl_alltoall_cpp_test.cpp  | sycl_alltoall_test.cpp | [Alltoall](https://intel.github.io/oneccl/spec/communication_primitives.html#alltoall) |
| sycl_bcast_cpp_test.cpp | sycl_bcast_test.cpp | [Broadcast](https://intel.github.io/oneccl/spec/communication_primitives.html#broadcast)|
| sycl_reduce_cpp_test.cpp  | sycl_reduce_test.cpp | [Reduce](https://intel.github.io/oneccl/spec/communication_primitives.html#reduce) |
| cpu_allreduce_cpp_test.cpp | cpu_allreduce_test.cpp/cpu_allreduce_bfp16.c |[Allreduce](https://intel.github.io/oneccl/spec/communication_primitives.html#allreduce) |
|oneCCL_Getting_Started.ipynb (check below Notice)| | |
>  Notice : Please use Intel oneAPI DevCloud as the environment for jupyter notebook samples. \
Users can refer to [DevCloud Getting Started](https://devcloud.intel.com/oneapi/get-started/) for using DevCloud \
Users can use JupyterLab from DevCloud via "One-click Login in", and download samples via "git clone" or the "oneapi-cli" tool \
Once users are in the JupyterLab with download jupytered notebook samples, they can start following the steps without further installion needed.

## License  
Those code samples are licensed under MIT license

## Pre-requirement

### CPU

-----

The samples below require the following components, which are part of the [Intel oneAPI DL Framework Developer Toolkit (DLFD Kit)
](https://software.intel.com/en-us/oneapi/dldev-kit)
*  Intel oneAPI Collective Communications Library (oneCCL)

You can refer to this page [oneAPI](https://software.intel.com/en-us/oneapi) for toolkit installation.


### GPU and CPU

-----

The samples below require the following components, which are part of the [Intel oneAPI Base Tookit](https://software.intel.com/en-us/oneapi/oneapi-kit)
*  Intel oneAPI Collective Communications Library (oneCCL)
*  Intel oneAPI DPC++ Compiler
*  Intel oneAPI DPC++ Library (oneDPL)

The samples also require OpenCL driver. Please refer [System Requirements](https://software.intel.com/en-us/articles/intel-oneapi-base-toolkit-system-requirements) for OpenCL driver installation.


You can refer to this page [oneAPI](https://software.intel.com/en-us/oneapi) for toolkit installation.




## How to Build  

### on Linux  

#### CPU only:

- Build the samples  with GCC for CPU only \
  please replace ${ONEAPI_ROOT} for your installation path. \
  ex : /opt/intel/inteloneapi \
  Don't need to replace {DPCPP_CMPLR_ROOT} 
  ```
  source ${ONEAPI_ROOT}/setvars.sh --ccl-configuration=cpu_icc

  cd oneapi-toolkit/oneCCL/oneCCL_Getting_Started   
  mkdir build  
  cd build 
  cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  make
  ```

#### GPU and CPU:

- Build the samples  with SYCL for GPU and CPU \
  please replace ${ONEAPI_ROOT} for your installation path. \
  ex : /opt/intel/inteloneapi \
  Don't need to replace {DPCPP_CMPLR_ROOT} 
  ```
  source ${ONEAPI_ROOT}/setvars.sh --ccl-configuration=cpu_gpu_dpcpp

  cd oneapi-toolkit/oneCCL/oneCCL_Getting_Started  
  mkdir build  
  cd build 
  cmake ..  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=dpcpp
  make
  ```

## How to Run  

### on Linux  

#### CPU only:
- Run the program \
  take cpu_allreduce_cpp_test for example. \
  you can apply those steps for all other sample binaries. \
  please replace the {NUMBER_OF_PROCESSES} with integer number accordingly

  ```
  mpirun -n ${NUMBER_OF_PROCESSES} ./out/cpu_allreduce_cpp_test 
  ```
  
  ex: 
  ```
  mpirun -n 2 ./out/cpu_allreduce_cpp_test
  ``` 
  

#### GPU and CPU:
- Run the program \
  take sycl_allreduce_cpp_test for example. \
  you can apply those steps for all other sample binaries. \
  please replace the {NUMBER_OF_PROCESSES} with integer number accordingly

  ```
  mpirun -n ${NUMBER_OF_PROCESSES} ./out/sycl_allreduce_cpp_test gpu|cpu|host|default
  ```
  
  ex: run on GPU
  ```
  mpirun -n 2 ./out/sycl_allreduce_cpp_test gpu
  ``` 
  

## Result Validation 

### on Linux 
- Run the program on CPU or GPU following [How to Run Session](#how-to-run)
- CPU Results

  ```
  Provided device type: cpu
  Running on Intel(R) Core(TM) i7-7567U CPU @ 3.50GHz
  Example passes
  ```
  please note that name of running device may vary according to your environment
  

- GPU Results
  ```
  Provided device type: gpu
  Running on Intel(R) Gen9 HD Graphics NEO
  Example passes
  ```
  please note that name of running device may vary according to your environment
