# Intel oneAPI Deep Neural Network Library (oneDNN)

Deep Neural Networks Library for Deep Neural Networks (oneDNN) is an open-source performance library for deep learning applications. The library includes basic building blocks for neural networks optimized for Intel Architecture Processors and Intel Processor Graphics. oneDNN is intended for deep learning applications and framework developers interested in improving application performance on Intel CPUs and GPUs

Github : https://github.com/oneapi-src/oneDNN

## License  
The code samples are licensed under MIT license

# oneDNN samples

| Type      | Name                 | Description                                                  |
| --------- | ----------------------- | ------------------------------------------------------------ |
| Component | oneDNN_Getting_Started    | This C++ API example demonstrates basic of oneDNN programming model by using a ReLU operation. |
| Component | oneDNN_SYCL_InterOps      | This C++ API example demonstrates oneDNN SYCL extensions API programming model by using a custom SYCL kernel and a ReLU operation . |
| Component | oneDNN_CNN_FP32_Inference | This C++ API example demonstrates building/runing a simple CNN fp32 inference against different oneDNN pre-built binarie. |
| Component | oneDNN_Getting_Started.ipynb|This Jupyter Notebook demonstrates how to compile a oneDNN sample with different releases via batch jobs on the Intel oneAPI DevCloud (check below Notice)|
| Component | oneDNN_CPU2GPU_Porting.ipynb|This Jupyter Notebook demonstrates how to port a oneDNN sample from CPU-only version to CPU&GPU version by using DPC++ on the Intel oneAPI DevCloud (check below Notice)|
>  Notice : Please use Intel oneAPI DevCloud as the environment for jupyter notebook samples. \
Users can refer to [DevCloud Getting Started](https://devcloud.intel.com/oneapi/get-started/) for using DevCloud \
Users can use JupyterLab from DevCloud via "One-click Login in", and download samples via "git clone" or the "oneapi-cli" tool \
Once users are in the JupyterLab with downloaded jupyter notebook samples, they can start following the steps without further installion needed.
