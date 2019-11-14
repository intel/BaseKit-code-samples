# FPGA Tutorial: Use Library

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Intel(R) Programmable Acceleration Card (PAC) with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel(R) oneAPI DPC++ Compiler (Beta) 
 
## Purpose
This tutorial demonstrates how to use a pre-built device library in your SYCL* design. A library is useful for reusing code or separating code for testing purposes. A library is also useful for providing features from some programming languages that do not exist in other languages.

## Prerequisite
You need the Intel(R) High Level Synthesis (HLS) Compiler version 19.3 to create a library. This is neither included with the Intel(R) oneAPI DPC++ Compiler (Beta) nor the Intel(R) FPGA Add-on for oneAPI Base Toolkit. For more information about the HLS compiler, refer to https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/hls-compiler.html.

## Library Generation
You can create a library using the following steps:

**NOTE:** The library is already generated for this tutorial. Therefore these steps are for reference only. 


1. By using the `fpga_crossgen` tool, create object files that contain representation for target devices (CPU and FPGA) and FPGA emulator. For example, execute the following command: 

   ```
   fpga_crossgen <source>.cpp --target syclfpga -o <library_object>.aoco
   ```
   This command generates the `<library_object>.aoco` file for creating a library file by the `fpga_libtool` and `<library_object>.o` implicitly for SYCL* host linking.
   
2. By using the `fpga_libtool` tool, collect one or more object(s) into a SYCL* library file. For example, execute the following command:

   ```
   fpga_libtool <library_object>.aoco --target syclfpga --create <library>.fpgalib
   ```
   This command generates the `<library>.fpgalib` file, which is used by the Intel(R) FPGA SDK for OpenCL(TM) when compiling the device code.

## Library Use
To use the precompiled library, add the following flags to the `dpcpp` command:

| Flags | Description |
| ------ | ------ |
| `-XsL<path_to_library_file>` | Specifies path to the `.fpgalib` library file. |
| `-Xsl<library>.fpgalib` | Specifies the SYCL* library file to the underlying Intel(R) FPGA SDK for OpenCL(TM). | 
| `-Xsoverride-library-version` | Bypasses checks between the library version and the current compiler version. |
| `lib.o` | Specifies the object file used for host linking. |

For more information about the `dpcpp` command, refer to the `Makefile`.

When using HLS or OpenCL source libraries as in this tutorial, the following warning should be ignored:

```warning: Linking two modules of different target triples: ' is 'spir64-unknown-unknown-intelfpga' whereas '' is 'spir64-unknown-unknown'```.

## Building the `use_library` Tutorial
Perform the following steps:
1. Install the tutorial into the `build` directory from the design directory by running `cmake`:

   ```
   mkdir build
   cd build
   cmake ..
   ```

2. Compile the tutorial design through the generated `Makefile`. The following four targets are provided, matching the recommended development flow:

    * Compile and run for emulation (fast compile time, targets emulated FPGA device) using the following command:

       ```
       make fpga_emu
       ./use_library_emu.fpga 
       ```

    * Generate HTML optimization reports using the following command: 

       ```
       make report
       ``` 
      Locate the report in the `use_library_report.prj/reports/report.html` directory.

    * Compile and run on an FPGA hardware (longer compile time, targets FPGA device) using the following command:

       ```
       make fpga 
       ./use_library.fpga
       ```

    * Compile and run on a CPU hardware (unoptimized) using the following command:

       ```
       make cpu_host
       ./use_library.cpu_host
       ```
3. Download the design, compiled for FPGA hardware, from this location: [download page](https://www.intel.com/content/www/us/en/programmable/products/design-software/high-level-design/one-api-for-fpga-support.html)

## Building the `use_library` Tutorial in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*). For instructions, refer to the following link: [Intel(R) oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)