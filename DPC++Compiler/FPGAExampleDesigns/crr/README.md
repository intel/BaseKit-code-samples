

# FPGA Example Design: CRR Binomial Tree Model for Option Pricing

| Optimized for                     | Description
---                                 |---
|OS                                 | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel(R) Programmable Acceleration Card (PAC) with Intel(R) Arria(R) 10 GX FPGA; Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz
| Software                          | Intel(R) oneAPI DPC++ Compiler (Beta)

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

Performance 

| CRR                              | Performance Data (see performance disclaimers section below)
|:---                              |:---
| SYCL                             | 60 assets/s

## Description
This sample implements the Cox-Ross-Rubinstein (CRR) binomial tree model that is used in the finance field for American exercise options with five Greeks (delta, gamma, theta, vega and rho). The simple idea is to model all possible assets price paths using a binomial tree. 

## License  
This code sample is licensed under MIT license.

## Design Details
### Design Inputs
This design reads inputs from the `ordered_inputs.csv` file. The inputs are:  

| Input                     | Description
---                         |---
| `nSteps` | Number of time steps in the binomial tree. The maximum `nSteps` in this design is 8191.
| `CP` | -1 or 1 represents put and call options, respectively.
| `Spot` | Spot price of the underlying price. 
| `Fwd` | Forward price of the underlying price.
| `Strike` | Exercise price of the option.
| `Vol` |  Percent volatility that the design reads as a decimal value.
| `DF` | `Spot` or `Fwd`
| `T` | Time, in years, to the maturity of the option.

### Design Outputs
This design writes outputs to the `ordered_outputs.csv` file. The outputs are:

| Output                     | Description
---                         |---
| `value` | Option price
| `delta` | Measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price.
| `gamma` | Measures the rate of change in the `delta` with respect to changes in the underlying price. 
| `vega` | Measures sensitivity to volatility. 
| `theta` | Measures the sensitivity of the value of the derivative to the passage of time.
| `rho` | Measures sensitivity to the interest of rate.

### Design Correctness
This design tests the correctness by comparing the FPGA results and CPU results. 

### Design Performance 
This design tests the FPGA performance to determine how many assets can be processed per second, with five valid inputs. Five inputs are sufficient to test the performance. 

## Building the CRR Design (Linux)
Perform the following steps:

1. Install the design into the `build` directory from the design directory by running `cmake`:

```
mkdir build
cd build
cmake .. 
```
2. Compile the design through the generated `Makefile`. The following four build targets are provided, matching the recommended development flow:
   * Compile and run for emulation (fast compile time, targets emulated FPGA device). It runs one input data from the input file for emulation.
      ```
      make fpga_emu
      ./crr.fpga_emu ./src/data/ordered_inputs.csv -o=./src/data/ordered_outputs.csv
      ```
   * Generate HTML optimization report. You can find the report in `crr_report.prj/reports/crr_report.html`. 
      ```
      make report
      ```  

      
   * Compile and run on the FPGA hardware (longer compile time, targets FPGA device). It runs all the input data from the input file. 

      ```            
      make fpga
      ./crr.fpga ./src/data/ordered_inputs.csv -o=./src/data/ordered_outputs.csv
      ``` 
   * Compile and run on the CPU hardware (unoptimized) using: 

      ```            
      make cpu_host
      ./crr.cpu_host ./src/data/ordered_inputs.csv -o=./src/data/ordered_outputs.csv
      ``` 
3. Download the design, compiled for FPGA hardware, from this location: [download](https://www.intel.com/content/dam/altera-www/global/en_US/others/support/examples/download/crr.fpga)

## Building the CRR Design (Windows)
Perform the following steps:
Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following four build targets are provided, matching the recommended development flow:
   * Compile and run for emulation (fast compile time, targets emulated FPGA device). It runs one input data from the input file for emulation.
      ```
      ninja fpga_emu
      crr.fpga_emu.exe ./data/ordered_inputs.csv -o=./data/ordered_outputs.csv 
      ```
   * Generate HTML optimization report. You can find the report in `../src/main.prj/reports/report.html`. 
      ```
      ninja report
      ```  
      
   * **Not supported yet:** Compile and run on the FPGA hardware.
   
   * Compile and run on the CPU hardware (unoptimized) using: 

      ```            
      ninja cpu_host
      crr.fpga_emu.exe ./data/ordered_inputs.csv -o=./data/ordered_outputs.csv 
      ```

## Building the CRR Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this example design in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel(R) oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Running the Example Design
Invoke the CRR example design as follows:

```
./crr.fpga <input_file> [-o=<output_file>]
```

### Sample Output
```
============ Correctness Test =============

Running analytical correctness checks...

CPU-FPGA Equivalence: PASS

============ Throughput Test =============

Avg throughput: 60.8 assets/s
```
### Arguments

| Argument | Description
---        |---
| `<input_file>` | Optional argument that provides the input data. The default file is `/data/ordered_inputs.csv`
| `-o=<output_file>` | Optional argument that specifies the name of the output file. The default name of the output file is `ordered_outputs.csv`. 

## Source Code Explanation

| File                         | Description 
---                            |---
| `main.cpp`        | Contains both host code and SYCL* kernel code.
| `CRR_common.h`               | Header file for `main.cpp`. Contains the data structures needed for both host code and SYCL* kernel code.

## Backend compiler flags used when compiling the design 

| Flag | Description
---    |---
`-Xshardware` | target FPGA hardware
`-Xsno-accessor-aliasing` | indicates that the arguments are independent from each other
`-Xsfpc` | allows backend to remove intermediate roundings 
`-Xsparallel=2` | uses 2 cores when compiling the bitstream through Quartus
`-Xsseed=2` | uses seed 2 during Quartus, yields slightly higher fmax
      
## Performance disclaimers

Tests document performance of components on a particular test, in specific systems. Differences in hardware, software, or configuration will affect actual performance. Consult other sources of information to evaluate performance as you consider your purchase.  For more complete information about performance and benchmark results, visit [www.intel.com/benchmarks](www.intel.com/benchmarks).

Performance results are based on testing as of October 1, 2019 and may not reflect all publicly available security updates.  See configuration disclosure for details.  No product or component can be absolutely secure.

Intel technologies’ features and benefits depend on system configuration and may require enabled hardware, software or service activation. Performance varies depending on system configuration. Check with your system manufacturer or retailer or learn more at [intel.com](www.intel.com).

The performance was measured by Intel on October 1, 2019

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

(C) Intel Corporation.
      
## References
[Khronous SYCL Resources](https://www.khronos.org/sycl/resources)

[Binomial options pricing model](https://en.wikipedia.org/wiki/Binomial_options_pricing_model)

[Wike page for finance Greeks](https://en.wikipedia.org/wiki/Greeks_(finance))

[OpenCL Intercept Layer](https://github.com/intel/opencl-intercept-layer)

