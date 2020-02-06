# FPGA Example Design: QR Decomposition of Matrices

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel(R) Programmable Acceleration Card (PAC) with Intel(R) Arria(R) 10 GX FPGA; Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz
| Software                          | Intel(R) oneAPI DPC++ Compiler (Beta)

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

Performance 

| QR decomposition                  | Performance data (see performance disclaimers section below)
|:---                               |:---
| SYCL                              | 27k matrices/s

## Description

This example design demonstrates QR decomposition of matrices (complex numbers), a common operation employed in linear algebra. Matrix A (input) is decomposed into a product of an orthogonal matrix Q and an upper triangular matrix R.

## License  
This code sample is licensed under MIT license.

## Building the Example Design (Linux)

1. Install the design into a directory `build` from the design directory by running `cmake`:

```
mkdir build
cd build
cmake ..
```

2. Compile the design through the generated `Makefile`. The following four targets are provided, matching the recommended development flow:

    * Compile and run for emulation (fast compile time, targets emulated FPGA device). This step generates a random matrix and computes QR decomposition.

       ```
       make fpga_emu
       ./qrd.fpga_emu 
       ```

    * Generate HTML performance report. Find the report in `qrd_report.prj/reports/report.html`directory.

       ```
       make report
       ``` 

    * Compile and run on an FPGA hardware (longer compile time, targets FPGA device). This step performs the following:
      * Generates 32768 random matrices. 
      * Computes QR decomposition on all matrices. 
      * Evaluates performance.

       ```
       make fpga 
       ./qrd.fpga 32768 
       ```

    * Compile and run on CPU hardware (not optimized). This step generates a random matrix and computes QR decomposition.

       ```
       make cpu_host
       ./qrd.cpu_host 
       ```
3. Download the design, compiled for FPGA hardware, from this location: [download](https://www.intel.com/content/dam/altera-www/global/en_US/others/support/examples/download/qrd.fpga)

## Building the Example Design (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following four targets are provided, matching the recommended development flow:

    * Compile and run for emulation (fast compile time, targets emulated FPGA device). This step generates a random matrix and computes QR decomposition.

       ```
       ninja fpga_emu
       ./qrd.fpga_emu 
       ```

    * Generate HTML performance report. Find the report in `../src/qrd.prj/reports/report.html`directory.

       ```
       ninja report
       ``` 

    * **Not supported yet:** Compile and run on an FPGA hardware.

    * Compile and run on CPU hardware (not optimized). This step generates a random matrix and computes QR decomposition.

       ```
       ninja cpu_host
       ./qrd.cpu_host 
       ```

## Building the Example Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this example design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel(R) oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Using the Example design
You can apply QR decomposition to a number of matrices. Invoke it as shown in the following: 

```
./qrd.fpga [<num>]
```

### Sample Output

```
Device name: pac_a10 : Intel PAC Platform (pac_f400000)
Generating 32768 random matrices
Running QR decomposition of 32768 matrices repeatedly
   Total duration:   39.5882 s
Throughput: 27.1k matrices/s
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
`-Xsno-accessor-aliasing` | indicates that the arguments are independent from each other
`-Xsfmax=300` | the FPGA backend attempts to achieve 300 MHz
`-Xsfp-relaxed` | allows backend to relax the order of additions 
`-Xsparallel=2` | uses 2 cores when compiling the bitstream through Quartus
`-Xsseed=2` | uses seed 2 during Quartus, yields slightly higher fmax


## Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of October 1, 2019 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on October 1, 2019

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.
      
