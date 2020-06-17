# FPGA Example Design: GZIP Compression

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; Intel Xeon® CPU E5-1650 v2 @ 3.50GHz
| Software                          | Intel® oneAPI DPC++ Compiler (Beta)

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

Performance 

| GZIP                              | Performance Data (see performance disclaimers section below)
|:---                               |:---
| SYCL                              | 3.4 GB/s

## Description

This example design implements a compression algorithm. The implementation is optimized for the FPGA device. The compression result is GZIP-compatible and can be decompressed with GUNZIP.
The GZIP output file format is compatible with GZIP's DEFLATE algorithm, and follows a fixed subset of [RFC 1951](https://www.ietf.org/rfc/rfc1951.txt). See the References section for more specific references. 

The algorithm uses a GZIP-compatible Limpel-Ziv 77 (LZ77) algorithm for data de-duplication, and a gzip compatible Static Huffman algorithm for bit reduction. The implementation includes three FPGA accelerated tasks (LZ77, Static Huffman and CRC). 


## License  
This code sample is licensed under MIT license.

## Building the GZIP Design (Linux)

1. Install the design into a directory `build` from the design directory by running `cmake`:

   ```
   mkdir build
   cd build
   ```

   If you are compiling for the Intel® PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:

   ```
   cmake ..
   ```

   If instead you are compiling for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:

   ```
   cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
   ```

2. Compile the design through the generated `Makefile`. The following three build targets are provided, matching the recommended development flow:

    * Compile and run for emulation (fast compile time, targets emulated FPGA device). Use a small file for emulation, for example, `<src/gzip.cpp>`.

       ```
       make fpga_emu
       ./gzip.fpga_emu ../src/gzip.cpp -o=test.gz 
       ```

       NOTE: The emulator compile has been defaulted to have debugging disabled. To enable debugging in your emulator executable, edit the src/CMakeList.txt file prior to step 1, removing "-g0" from the EMULATOR_COMPILE_FLAGS variable. In step 2, prior to running gzip.fpga_emu, use the following environment variable definition: "export CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=768KB".

    * Generate HTML optimization report. You can find the report in `gzip_report.prj/reports/report.html`.

       ```
       make report
       ``` 

    * Compile and run on FPGA hardware (longer compile time, targets FPGA device). Use a larger file (> 8MB) for performance evaluation, for example, the executable `/usr/bin/snap`.

       ```            
       make fpga
       ./gzip.fpga /usr/bin/snap -o=test.gz
       ``` 


(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://www.intel.com/content/dam/altera-www/global/en_US/others/support/examples/download/gzip.fpga" download>here</a>.

## Building the GZIP Design (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following three build targets are provided, matching the recommended development flow:

    * Compile and run for emulation (fast compile time, targets emulated FPGA device). Use a small file for emulation, for example, `<src/gzip.cpp>`.

      ```
      ninja fpga_emu
      gzip.fpga_emu.exe ../src/gzip.cpp -o=test.gz 
      ```

    * Generate HTML optimization report. You can find the report in `../src/gzipkernel_report.prj/reports/report.html`.

      ```
      ninja report
      ``` 

      If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/gzipkernel_s10_pac_report.prj/reports/report.html`.

      ```
      ninja report_s10_pac
      ```

    * **Not supported yet:** Compile and run on FPGA hardware.

## Building the GZIP Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this example design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Running the Example Design
GZIP is invoked as follows:

```
./gzip.fpga <input_file> [-o=<output_file>]
```

For highest performance, the application is submitting to the accelerator a set of commands that allows overlapping the execution of all kernels and data transfers from and to the device. In addition, the LZ77 and Static Huffman kernels are interconnected through kernel to kernel pipes to reduce memory bandwidth requirements.

In addition to the actual compression ratio, the example design assesses the compression performance, by running compression on the same file for a number of times.    

### Sample Output

```
./gzip.fpga_emu /usr/bin/snap -o=test.gz
Running on device:  pac_a10 : Intel PAC Platform (pac_f400000)
GB/s 3.71741
Compression Ratio 43.089%
PASSED
```
### Arguments

| Argument | Description
---        |---
| `<input_file>` | Mandatory argument that specifies the file to be compressed.
| `-o=<output_file>` | Optional argument that specifies the name of the output file. The default name of the output file is `<input_file>.gz`. 

## Design Structure

| Kernel                     | Description
---                          |---
| LZ Reduction               | Implements a LZ77 algorithm for data deduplication. The algorithm produces distance and length information that is compatible with GZIP's DEFLATE implementation. 
| Static Huffman             | Uses the same Static Huffman codes used by GZIP's DEFLATE algorithm when it chooses a Static Huffman coding scheme for bit reduction. This choice maintains compatibility with GUNZIP. 
| CRC                        | Adds a CRC checksum based on the input file; this is required by the gzip file format 

## Source Code Explanation

| File                         | Description 
---                            |---
| `gzip.cpp`                   | Contains the `main()` function and the top-level interfaces to the SYCL* GZIP functions.
| `gzipkernels.cpp`            | Contains the SYCL* kernels used to implement GZIP. 
| `CompareGzip.cpp`            | Contains code to compare a GZIP-compatible file with the original input.
| `WriteGzip.cpp`              | Contains code to write a GZIP compatible file. 
| `crc32.cpp`                  | Contains code to calculate a 32-bit CRC that is compatible with the GZIP file format and to combine multiple 32-bit CRC values. It is used to account only for the CRC of the last few bytes in the file, which are not processed by the accelerated CRC kernel. 
| `kernels.h`                  | Contains miscellaneous defines and structure definitions required by the LZReduction and Static Huffman kernels.
| `crc32.h`                    | Header file for `crc32.cpp`.
| `gzipkernels.h`              | Header file for `gzipkernels.cpp`.
| `CompareGzip.h`              | Header file for `CompareGzip.cpp`.
| `WriteGzip.h`                | Header file for `WriteGzip.cpp`. 

## Backend compiler flags used when compiling the design 

| Flag | Description
---    |---
`-Xshardware` | target FPGA hardware
`-Xsno-accessor-aliasing` | indicates that the arguments are independent from each other
`-Xsparallel=2` | uses 2 cores when compiling the bitstream through Quartus
`-Xsseed=1` | uses seed 1 during Quartus, yields slightly higher fmax

## Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of October 1, 2019 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on October 1, 2019

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.
      
## References
[Khronous SYCL Resources](https://www.khronos.org/sycl/resources)

[Intel GZIP OpenCL Design Example](https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/gzip-compression.html)

[RFC 1951 - DEFLATE Data Format](https://www.ietf.org/rfc/rfc1951.txt)

[RFC 1952 - GZIP Specification 4.3](https://www.ietf.org/rfc/rfc1952.txt)

[OpenCL Intercept Layer](https://github.com/intel/opencl-intercept-layer)


