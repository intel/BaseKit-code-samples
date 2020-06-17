# FPGA Tutorial: Double Buffering to Overlap Kernel Execution with Buffer Transfers and Host-Processing

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates how to parallelize host-side processing and buffer transfers between host and device with kernel execution, which can improve overall application performance.

## Key Concepts
This tutorial helps you learn the following concepts:
* Determine when is double buffering possible
* Measure the impact of double buffering 

## Background

In an application where the FPGA kernel is executed multiple times, the host must perform the following processing and buffer transfers before each kernel invocation.
1. The output data from the *previous* invocation must be transferred from device to host and then processed by the host. Examples of this processing include: 
   * Copying the data to another location
   * Rearranging the data 
   * Verifying it in some way. 
2. The input data for the *next* invocation must be processed by the host and then transferred to the device. Examples of this processing include: 
   * Copying the data from another location 
   * Rearranging the data for kernel consumption 
   * Generating the data in some way

Without double buffering, host processing and buffer transfers occur *between* kernel executions. Therefore, there is a gap in time between kernel executions, which you can refer as kernel *downtime* (see diagram below). If these operations overlap with kernel execution, the kernels can execute back-to-back with minimal downtime, thereby increasing overall application performance.


## Determining When is Double Buffering Possible

Let's define the required variables:
* **R** = Time to transfer the kernel's output buffer from device to host.
* **Op** = Host-side processing time of kernel output data (*output processing*)
* **Ip** = Host-side processing time for kernel input data (*input processing*)
* **W** = Time to transfer the kernel's input buffer from host to device.
* **K** = Kernel execution time

![](downtime.png)

In general, **R**, **Op**, **Ip**, and **W** operations must all complete before the next kernel is launched. To maximize performance, while one kernel is executing on the device, these operations should execute simultaneously on the host and operate on a second set of buffer locations. They should complete before the current kernel completes, thus allowing the next kernel to be launched immediately with no downtime. In general, to maximize performance, the host must launch a new kernel every **K**.

This leads to the following constraint:

```c++
R + Op + Ip + W <= K, in order to minimize kernel downtime.
```
If the above constraint is not satisfied, a performance improvement may still be observed because *some* overlap (perhaps not complete overlap) is still possible. Further improvement is possible by extending the double buffering concept to N-way buffering (see the corresponding tutorial).

## Measuring the Impact of Double Buffering

You must get a sense of the kernel downtime to identify the degree to which this technique can help improve performance.

This can be done by querying the total kernel execution time from the runtime and comparing it to the overall application execution time. In an application where kernels execute with minimal downtime, these two numbers will be close. However, if kernels have a lot of downtime, overall execution time will notably exceed kernel execution time. The tutorial code exemplifies how to do this.

## Implementation Notes

The basic idea is to: 
1. Perform the input processing for the first two kernel executions and queue them both. 
2. Immediately call the `process_output()` method (automatically blocked by the SYCL* runtime) on the first kernel completing because of the implicit data dependency. 
3. When the first kernel completes, the second kernel begins executing immediately because it was already queued. 
4. While the second kernel runs, the host processes the output data from the first kernel and prepares the input data for the third kernel. 
5. As long as the above operations complete before the second kernel completes, the third kernel is queued early enough to allow it to be launched immediately after the second kernel. 

The process then repeats.

## Sample Results

A test compile of this tutorial design achieved a maximum frequency (f<sub>MAX</sub>) of approximately 340 MHz on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The results with and without double buffering are shown in the following table:

Configuration | Overall Execution Time (ms) | Total Kernel Execution time (ms)
-|-|-
Without double buffering | 23462 | 15187
With double buffering | 15145 | 15034

In both runs, the total kernel execution time is similar, as expected. However, without double buffering, the overall execution time notably exceeds the total kernel execution time, implying there is downtime between kernel executions. With double buffering, the overall execution time is close to the the total kernel execution time.

## Building the `double_buffering` Design (Linux)

1. Install the design in `build` directory from the design directory by running `cmake`:

  ```
  mkdir build
  cd build
  ```

  If you are compiling for the Intel PAC with Intel Arria® 10 GX FPGA, run `cmake` using the command:

  ```
  cmake ..
  ```

  If instead you are compiling for the Intel® PAC with Intel Stratix® 10 SX FPGA, run `cmake` using the command:

  ```
  cmake .. -DFPGA_BOARD=intel_s10sx_pac:pac_s10
  ```

2. Compile the design through the generated `Makefile`. The following four build targets are provided that match the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device): 

     ```
     make fpga_emu
     ./double_buffering.fpga_emu 
     ```

   * Generate HTML optimization reports using:
     
     ```
     make report
     ```
     Locate the report under the `double_buffering_report.prj/reports/report.html` directory.
   
   * Compile and run on FPGA hardware (longer compile time, targets FPGA device): 

      ```
     make fpga 
     ./double_buffering.fpga
     ```

   * Compile and run on CPU hardware (not optimized): 

     ```
     make cpu_host
     ./double_buffering.cpu_host
     ```


(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://www.intel.com/content/dam/altera-www/global/en_US/others/support/examples/download/double-buffering.fpga" download>here</a>.


## Building the `double_buffering` Design (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following four build targets are provided that match the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device): 

     ```
     ninja fpga_emu
     double_buffering.fpga_emu.exe 
     ```

   * Generate HTML optimization reports using:
  
     ```
     ninja report
     ```
     Locate the report under the `../src/double_buffering_report.prj/reports/report.html` directory.

     If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/double_buffering_s10_pac_report.prj/reports/report.html`.

     ```
     ninja report_s10_pac
     ```

   * **Not supported yet:** Compile and run on FPGA hardware (longer compile time, targets FPGA device): 

   * Compile and run on CPU hardware (not optimized): 

     ```
     ninja cpu_host
     double_buffering.cpu_host.exe 
     ```

## Building the `double_buffering` Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
