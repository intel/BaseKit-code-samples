# FPGA Tutorial: Device Link

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel(R) Programmable Acceleration Card (PAC) with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel(R) oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware / Generating HTML Optimization Report is not supported in Windows*_

## Purpose
This tutorial demonstrates how to use device link in your FPGA compilation flow to save development time.

## Device Link

Intel(R) oneAPI DPC++ Compiler (Beta) only supports ahead-of-time compilation for FPGA, which means that an FPGA device image is generated at compile time. FPGA device image generation processes can take hours to complete. If you want to recompile your host code only, the process is time-consuming.

The device link mechanism allows you to separate device compilation and host compilation. When the code change only applies to host-only files, an FPGA device image is not regenerated. 


### Example
Consider the following example where a program is separated into two files, `main.cpp` and `kernel.cpp`. Only the `kernel.cpp` file contains the device code. 

In the normal compile process, an FPGA device image generation happens during link time. This indicates that any change to `main.cpp` or `kernel.cpp` file triggers regeneration of an FPGA device image. 

```
# normal compile command
dpcpp -fintelfpga main.cpp kernel.cpp -Xshardware -o link.fpga
```

The following graph depicts this compilation process:

![](normal_compile.png)


If you want to iterate on the host code and avoid long compile time for your FPGA device, consider using a device link to separate device and host compilation:

```
# device link command
dpcpp -fintelfpga -fsycl-link=image <input files> [options]
```

The compilation is a 3-step process:

1. Compile device part (hours) using: 

   ```
   dpcpp -fintelfpga -fsycl-link=image kernel.cpp -o dev_image.a -Xshardware
   ```
   For generic program, input files should include all source files that contain device code.


2. Compile host part (seconds) using:
   
   ``` 
   dpcpp -fintelfpga main.cpp -c -o host.o
   ```
   For generic program, input files should include all source files that only contain host code.


3. Create device link (seconds) using:

   ```
   dpcpp -fintelfpga host.o dev_image.a -o device_link.fpga
   ```
   For generic program, input should have N (N >= 0) host object file and one device image file.

**NOTE:** You only need to perform steps 2 and 3 when modifying host-only files.

The following graph depicts device link compilation process:

![](device_link.png)



## Building the `device_link` Example (Linux)

Perform the following steps:
1. Install the design into the `build` directory from the design directory by running `cmake`:

   ```
   mkdir build
   cd build
   cmake ..
   ```

2. Compile the design through the generated `Makefile`. The following three build targets are provided, matching the recommended development flow:

   **NOTE:** For FPGA emulator target and FPGA target, device link is used in `Makefile`. You can attempts to modify the `main.cpp` file and recompile.

   * Compile and run for emulation uisng: 

      ```
      make fpga_emu
      ./device_link.fpga_emu 
      ```

   * Compile and run on an FPGA hardware using:    

     ```
     make fpga 
     ./device_link.fpga
     ```

   * Compile and run on a CPU hardware (unoptimized): 

     ```
     make cpu_host
     ./device_link.cpu_host
     ```
3. Download the design, compiled for FPGA hardware, from this location: [download page](https://www.intel.com/content/www/us/en/programmable/products/design-software/high-level-design/one-api-for-fpga-support.html)



## Building the `device_link` Example (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 
Note: Ensure that Microsoft Visual Studio* (2019 Version 16.4 or newer) with "Desktop development with C++" workload is installed on your system.

Perform the following steps:

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following three build targets are provided, matching the recommended development flow:

   **NOTE:** For FPGA emulator target and FPGA target, device link is used in `Makefile`. You can attempts to modify the `main.cpp` file and recompile.

   * Compile and run for emulation uisng: 

      ```
      ninja fpga_emu
      device_link.fpga_emu 
      ```

   * **Not supported yet:**  Compile and run on an FPGA hardware

   * Compile and run on a CPU hardware: 

     ```
     ninja cpu_host
     device_link.cpu_host.exe
     ```

## Building the `device_link` Example in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel(R) oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
