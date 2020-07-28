# FPGA Tutorial: Overlapping Kernel Execution with Buffer Transfers and Host-Processing

This tutorial demonstrates how to parallelize host-side processing and buffer transfers between host and device with kernel execution. This can improve overall application performance.
This is different from the 'double buffering' tutorial in that it demonstrates how to perform this overlap when the host-processing time exceeds kernel execution time.


| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates how to parallelize host-side processing and buffer transfers between the host and device with kernel execution that improves the overall application performance.
This is different from the 'double buffering' tutorial, which demonstrates how to perform this overlap when the host-processing time exceeds kernel execution time.

## Key Concepts
* N-Way buffering technique
* Measuring the Impact of N-Way Buffering

## Background

In an application where the FPGA kernel is executed multiple-times, the host must perform the following processing and buffer transfers before each kernel invocation: 
1. The output data from the *previous* invocation must be transferred from the device to host and then processed by the host. Examples of this processing include the following: 
   * Copying the data to another location
   * Rearranging the data 
   * Verifying it in some way. 
2. The input data for the *next* invocation must be processed by the host and then transferred to the device. Examples of this processing include: 
   * Copying the data from another location
   * Rearranging the data for kernel consumption
   * Generating the data in some way.

Without the technique described in this tutorial, host processing and buffer transfers occur *between* kernel executions. Therefore, there is a gap in time between kernel executions, which you can refer as kernel "downtime" (see diagram below). If these operations overlap with kernel execution, the kernels can execute back-to-back with minimal downtime, thereby increasing overall application performance.

## N-Way Buffering

This technique is sometimes referred as *N-Way Buffering* but often referred as *double buffering* specifically when N=2.

Let's first define some variables:

| Variable | Description |
| ------ | ------ |
| **R** | Time to transfer the kernel's output buffer from device to host. |
| **Op** | Host-side processing time of kernel output data (*output processing*). | 
| **Ip** | Host-side processing time for kernel input data (*input processing*). | 
| **W** | Time to transfer the kernel's input buffer from host to device. | 
| **K** | Kernel execution time. | 
| **N** | Number of buffer sets used. | 
| **C** | Number of host-side CPU cores. | 



![](downtime.png)

In general, the **R**, **Op**, **Ip**, and **W** operations must all complete before the next kernel is launched. To maximize performance, while one kernel is executing on the device, these operations should run in parallel and operate on a separate set of buffer locations. You should complete before the current kernel completes, thus allowing the next kernel to be launched immediately with no downtime. In general, to maximize performance, the host must launch a new kernel every **K**.

If these host-side operations are executed serially, this leads to the following constraint:

```c++
R + Op + Ip + W <= K, to minimize kernel downtime.
```

In the above example, if the constraint is satisfied, the application requires two sets of buffers. In this case, **N**=2.

However, the above constraint may not be satisfied in some applications (i.e., if host-processing takes longer than the kernel execution time).

**NOTE**: A performance improvement may still be observed because kernel downtime may still be reduced (perhaps not maximally reduced). 

In this case, to further improve performance, the reduce host-processing time through multi-threading. Rather than executing the above operations serially, perform the input- and output-processing operations in parallel using two threads, leading to the following constraint:

```c++
Max (R+Op, Ip+W) <= K
and
R + W <= K, to minimize kernel downtime.
````

If the above constraint is still unsatisfied, the technique can be extended beyond two sets of buffers to **N** sets of buffers to help improve the degree of overlap. In this case, the constraint becomes:

```c++
Max (R + Op, Ip + W) <= (N-1)*K
and
R + W <= K, to minimize kernel downtime.
```

The idea of N-way buffering is to prepare **N** sets of kernel input buffers, launch **N** kernels, and when the first kernel completes, begin the subsequent host-side operations. These operations may take a long time (longer than **K**), but they do not cause kernel downtime because an additional **N**-1 kernels have already been queued and can launch immediately. By the time these first **N** kernels complete, the aforementioned host-side operations would have also completed and the **N**+1 kernel can be launched with no downtime. As additional kernels complete, corresponding host-side operations are launched on the host, in a parallel fashion, using multiple threads. Although the host operations take longer than **K**, if **N** is chosen correctly, they will complete with a period of **K**, which is required to ensure we can launch a new kernel every **K**. To reiterate, this scheme requires multi-threaded host-operations because the host must perform processing for up to **N** kernels in parallel in order to keep up. 

The above formula can be used to calculate the **N** required to minimize downtime. However, there are some practical limits: 
* **N** sets of buffers are required on both the host and device, therefore both must have the capacity for this many buffers. 
* If the input and output processing operations are launched in separate threads, then (**N**-1)*2 cores are required, so **C** can be become the limiting factor.

## Measuring the Impact of N-Way Buffering

You must get a sense of the kernel downtime to identify the degree to which this technique can help improve performance.

This can be done by querying total kernel execution time from the runtime and comparing it to with overall application execution time. In an application where kernels execute with minimal downtime, these two numbers are close. However, if kernels have a lot of downtime, overall execution time notably exceeds the kernel execution time. The tutorial code exemplifies how to do this.

## Implementation Notes

The example code runs with multiple iterations to illustrate how performance improves as **N** increases and as multi-threading is used.

It is useful to think of the execution space as having **N** slots where the slots execute in chronological order, and each slot has its own set of buffers on the host and device. At the beginning of execution, the host prepares the kernel input data for the **N** slots and launches **N** kernels. When slot-0 completes, slot-1 begins executing immediately because it was already queued. The host begins both the output and input processing for slot-0. These two operations must complete before the host can queue another kernel into slot-0. The same is true for all slots. 

After each kernel is launched, the host-side operations (that occur *after* the kernel in that slot completes) are launched immediately from the `main()` program. They block until the kernel execution for that slot completes (this is enforced by the runtime).

## Sample Results

A test compile of this tutorial design achieved an f<sub>MAX</sub> of approximately 340 MHz on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The results are shown in the following table:

Configuration | Overall Execution Time (ms) | Total Kernel Execution time (ms)
-|-|-
1-way buffering, single-threaded | 64401 | 15187
1-way buffering, multi-threaded | 53540 | 15187
2-way buffering, multi-threaded | 27281 | 15187
5-way buffering, multi-threaded | 16284 | 15188

In all runs, the total kernel execution time is similar, as expected. In the first three configurations, the overall execution time notably exceeds the total kernel execution time, implying there is downtime between kernel executions. However, as we switch from single-threaded to multi-threaded host operations and increase the number of buffer sets used, the overall execution time approaches the kernel execution time.

## Building the `n_way_buffering` Design (Linux)

1. Install the design into the `build` directory from the design directory by running `cmake`:

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

2. Compile the design through the generated `Makefile`. The following four build targets are provided that matches the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device) using: 

      ```
      make fpga_emu
      ./n_way_buffering.fpga_emu 
      ```

   * Generate HTML optimization reports using: 

      ```
      make report
      ``` 
      Locate the report in the `n_way_buffering_report.prj/reports/report.html` directory.

   * Compile and run on FPGA hardware (longer compile time, targets FPGA device) using: 

      ```
      make fpga 
      ./n_way_buffering.fpga
      ```

(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://software.intel.com/content/dam/develop/external/us/en/documents/n_way_buffering.fpga.tar.gz" download>here</a>.

## Building the `n_way_buffering` Design (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following four build targets are provided that matches the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device) using: 

     ```
     ninja fpga_emu
     n_way_buffering.fpga_emu.exe 
     ```

   * Generate HTML optimization reports using: 
     
     ```
     ninja report
     ```
     Locate the report under the `../src/n_way_buffering_report.prj/reports/report.html` directory.

     If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/n_way_buffering_s10_pac_report.prj/reports/report.html`.

     ```
     ninja report_s10_pac
     ```
     
   * **Not supported yet:** Compile and run on FPGA hardware (longer compile time, targets FPGA device) using: 

## Building the `n_way_buffering` Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
