# Code Samples of Intel DPC++ compiler

| Code sample name                          | Supported Intel(r) Architecture(s) | Description
|:---                                       |:---                                |:---
| DPC++Compiler/Vector-add                                | FPGA, GPU, CPU                     | Simple vector-add program
| DPC++Compiler/sepia-filter                              | GPU, CPU                     | Color image conversion using 1D range
| DPC++Compiler/bootstrapping                | GPU, CPU                     | a simple data transfer program using DPC++
| DPC++Compiler/complex_mult                | GPU, CPU                     | Complex number Multiplication
| DPC++Compiler/Poly_Integral                | GPU, CPU                     | Polynomial Integral
| DPC++Compiler/Projectile_motion                | GPU, CPU                     | Projectile Motion
| DPC++Compiler/simple-vector-inc                | GPU, CPU                     | Simple vector increment
| DPC++Compiler/oneDPL/gamma-correction          | GPU, CPU                     | gamma correction using Parallel STL
| DPC++Compiler/oneDPL/stable_sort_by_key        | GPU, CPU                     | stable sort by key using `counting_iterator` and `zip_iterator`
| FPGATutorials/BestPractices/double_buffering| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/BestPractices/local_memory_cache| FPGA, CPU               | See details under FPGATutorials
| FPGATutorials/BestPractices/n_way_buffering| FPGA, CPU                  | See details under FPGATutorials
| FPGATutorials/BestPractices/triangular_loop| FPGA, CPU                  | See details under FPGATutorials
| FPGATutorials/Compilation/compile_flow| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/Compilation/device_link| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/Compilation/use_library| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/FPGAExtensions/LoopAttributes/loop_ivdep| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/FPGAExtensions/LoopAttributes/loop_unroll| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/FPGAExtensions/LoopAttributes/max_concurrency| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/FPGAExtensions/Other/fpga_register| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/FPGAExtensions/Other/no_accessor_aliasing| FPGA, CPU                 | See details under FPGA Tutorials
| FPGATutorials/FPGAExtensions/Other/system_profiling| FPGA, CPU                 | See details under FPGATutorials
| FPGATutorials/FPGAExtensions/MemoryAttributes/memory_attributes_overview| FPGA, CPU               | See details under FPGATutorials
| FPGATutorials/FPGAExtensions/Pipes/pipe_array| FPGA                           | See details under FPGATutorials
| FPGATutorials/FPGAExtensions/Pipes/pipes| FPGA                           | See details under FPGATutorials
| FPGAExampleDesigns/crr| FPGA, CPU                        | See details under FPGAExampleDesigns
| FPGAExampleDesigns/gzip| FPGA                       | See details under FPGAExampleDesigns
| FPGAExampleDesigns/grd| FPGA, CPU                        | See details under FPGAExampleDesigns
| Debugger/array-transform                              | GPU, CPU                     | Array transform
| ThreadingBuildingBlocks/tbb-async-sycl             | GPU, CPU  | The calculations are split between TBB Flow Graph asynchronous node that calls SYCL kernel on GPU while TBB functional node does CPU part of calculations.
| ThreadingBuildingBlocks/tbb-task-sycl              | GPU, CPU  | One TBB task executes SYCL code on GPU while another TBB task performs calculations using TBB parallel_for.
| VideoProcessingLibrary/Simple decode                     | CPU, GPU | shows how to use VPL to perform a simple video decode
| VideoProcessingLibrary/Decode with accelerator selection | CPU, GPU | shows how to select an accelerator to use for video decode
| VideoProcessingLibrary/Decode with video post-processing | CPU, GPU | shows how to select a color format and output resolution when decoding with VPL
| VideoProcessingLibrary/Demux and decode                  | CPU, GPU | shows how to use VPL to decode a video stream from a media container
| VideoProcessingLibrary/Memory integration                | CPU, GPU | shows how to use VPL memory functions to access output pixel data for integration into user pipelines

## License  
The code samples are licensed under MIT license 

## Known issues or limitations 
### On Windows Platform 

1.  If you are using Visual Studio 2019, Visual Studio 2019 version 16.3.0 or newer is required. 
2.  To build samples on Windows, a certion version of Windows SDK is required: 10.0.17763.0. If it is not installed, follow the instructions below to avoid build failure: 
    1.  open the code sample's .sln with Visual Studio 2017 or 2019, right click on the project name in "Solution Explorer" and select "Properties". It pops up the project property dialog. 
    2.  on the project property dialog, make sure to select "General" tab on the left, on the right side of the dialog, 2nd item is "Windows SDK Version". Click on the drop-down icon to select a version that is installed on your system. 
    3.  click on [Ok] to save. Now you should be able to build the code sample. 
3.  For beta, FPGA samples support Windows through FPGA-emulator. 
4.  If you encounter a compilation error like below when building a sample program, one reason is that the directory path of the sample is too long; the work around is to move the sample to a directory like "c:\temp\sample_name".
 
> Error MSB6003 The specified task executable "dpcpp-cl.exe" could not be run ...... 

