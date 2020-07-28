# FPGA Example Design: QR Decomposition of Matrices

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA; Intel Xeon® CPU E5-1650 v2 @ 3.50GHz
| Software                          | Intel® oneAPI DPC++ Compiler (Beta)


Performance 

| QR decomposition                  | Performance data (see performance disclaimers section below)
|:---                               |:---
| SYCL                              | 25k matrices/s

## Description

This example design demonstrates QR decomposition of matrices (complex numbers), a common operation employed in linear algebra. Matrix A (input) is decomposed into a product of an orthogonal matrix Q and an upper triangular matrix R.

## License  
This code sample is licensed under MIT license.

## Building the Example Design (Linux)

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

2. Compile the design through the generated `Makefile`. The following four targets are provided, matching the recommended development flow:

    * Compile and run for emulation (fast compile time, targets emulated FPGA device). This step generates a random matrix and computes QR decomposition.

       ```
       make fpga_emu
       CL_CONFIG_CPU_FORCE_PRIVATE_MEM_SIZE=32MB ./qrd.fpga_emu 
       ```

    * Generate HTML performance report. Find the report in `qrd_report.prj/reports/report.html`directory.

       ```
       make report
       ``` 

    * Compile and run on an FPGA hardware (longer compile time, targets FPGA device). This step performs the following:
      * Generates 32768 random matrices.
      * Computes QR decomposition on all matrices. NOTE: The design has been optimized for maximum performance when running on a large number of matrices that are a power of 2.
      * Evaluates performance.

       ```
       make fpga 
       ./qrd.fpga 32768 
       ```

       for the Intel® PAC with Intel Stratix® 10 SX FPGA, run on the hardware with the command:
       ```
       ./qrd.fpga 40960 
       ```
       
(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://software.intel.com/content/dam/develop/external/us/en/documents/qrd.fpga.tar.gz" download>here</a>.

## Building the Example Design (Windows)
**Not supported for this release**

## Building the Example Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this example design in the Eclipse* IDE (in Linux*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Using the Example design
You can apply QR decomposition to a number of matrices. Invoke it as shown in the following: 

```
./qrd.fpga [<num>]
```

### Sample Output

```
Device name: pac_a10 : Intel PAC Platform (pac_f000000)
Generating 32768 random matrices
Running QR decomposition of 32768 matrices repeatedly
   Total duration:   41.3764 s
Throughput: 25.3424k matrices/s
Verifying results on matrix 0 16384 32767
PASSED
```

### Arguments

| Argument | Description
---        |---
| `<num>`  | Optional argument that specifies the number of matrices to decompose. Its default value is `1`.

## Backend compiler flags used when compiling the design 

| Flag | Description
---    |---
`-Xshardware` | target FPGA hardware
`-Xsclock=330MHz` | the FPGA backend attempts to achieve 330 MHz
`-Xsfp-relaxed` | allows backend to relax the order of additions 
`-Xsparallel=2` | uses 2 cores when compiling the bitstream through Quartus
`-Xsseed=2` | uses seed 2 during Quartus, yields slightly higher fmax
`-DFIXED_ITERATIONS=64` | uses the value 64 for the constant FIXED_ITERATIONS. This constant is passed to the ivdep attribute for a loop in the design.

NOTE: the Xsseed and DFIXED_ITERATIONS values differ depending on the board being targeted.


## Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of October 1, 2019 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on October 1, 2019

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.
      
