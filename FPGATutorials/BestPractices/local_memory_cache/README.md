# FPGA Tutorial: Caching Local Memory to Reduce Loop Initiation Interval (II)

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel® Arria® 10 GX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates how to build a simple cache (implemented in FPGA registers) to store recently-accessed memory locations so that the compiler can achieve II=1.

## Background
Ideally, the compiler achieves an initiation interval (II) of 1 on performance-critical loops so that new loop iterations is launched on every clock cycle, thereby maximizing the throughput of the loop. However, if the loop contains a loop-carried variable that is implemented in local memory, the compiler may not be able to achieve II=1 because the memory access may take longer than one clock cycle. If it's possible for the updated memory location to be needed on the next loop iteration, the next iteration must be delayed to allow time for the update, hence II > 1.

## When to Use This Technique

View the **Optimization Report > Throughput Analysis > Loop Analysis** section. The report lists the II of all loops and explains why a lower II is not achievable. This technique is applicable if the reason is along the lines of:
* "Compiler failed to schedule this loop with smaller II due to memory dependency."
* "Most critical loop feedback path during scheduling" -- with the local memory load/store operation on the critical path.

or

* The latency of the load operation is 1. See **Optimization Report > System Viewers > Kernel Memory Viewer**, select the corresponding local memory from the Kernel Memory List, and mouse over the load operation "LD".

The compiler is capable of reducing the latency of the memory access by potentially sacrificing f<sub>MAX</sub>, thereby making it possible to achieve II=1 without this technique in some cases. Even if II=1 is already achieved, this technique can allow the compiler to implement a higher latency memory access, which can improve f<sub>MAX</sub> while still maintaining II=1. Therefore, even if the report shows II=1, a latency=1 load operation can be a sign that this technique is still applicable.


## Implementation Notes

The tutorial demonstrates the technique using a histogram. The histogram operation simply accepts an input vector of values, separates the values into buckets, and counts the number of values per bucket. For each input value, an output bucket location is determined, and the count for the bucket is incremented. This count is stored in the local memory and the increment operation requires reading from the memory, performing the increment, and storing the result. This read-modify-write operation is the critical path that can result in II > 1.

To reduce II, the idea is to store recently-accessed values in a cache that is capable of a 1-cycle read-modify-write operation. The cache is implemented in FPGA registers. If the memory location required on a given iteration exists in the cache, it is pulled from there. The updated count is written back to *both* the cache and the local memory. The `ivdep` pragma effectively informs the compiler that if a loop-carried variable (namely, the variable storing the histogram output) is needed within `CACHE_DEPTH` iterations, it is guaranteed to be available right away.

While any value of `CACHE_DEPTH` result in functional hardware, the ideal value of `CACHE_DEPTH` requires some experimentation. The depth of the cache needs to roughly cover the latency of the local memory access. To determine the correct value, it is suggested to start with a value of 2 and then increase it until both II = 1 and load latency > 1. In this tutorial, a `CACHE_DEPTH` of 5 is needed. 

Each iteration should only take a few moments by running 'make report' (refer to the section below on how to build the design). It is important to find the *minimal* value of `CACHE_DEPTH` that results in a maximal performance increase. Unnecessarily large values of `CACHE_DEPTH` consumes more resources and possibly reduce f<sub>MAX</sub>. Therefore, at a `CACHE_DEPTH` that results in II=1 and load latency = 1, if further increases to `CACHE_DEPTH` show no improvement in f<sub>MAX</sub>, then Intel® recommends to avoid increasing the `CACHE_DEPTH` any further.

The histogram kernel is implemented in two loops, with and without caching. The report shows II > 1 for the loop without caching and II = 1 for the loop with caching.

## Sample Results

A test compile of this tutorial design achieved an f<sub>MAX</sub> of approximately 250 MHz on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The results are shown in the following table:

Configuration | Execution Time (ms) | Throughput (MB/s)
-|-|-
Without caching | 0.153 | 418
With caching | 0.08 | 809

When caching is used, performance notably increases. As previously mentioned, this technique should result in an II reduction, which should lead to a throughput improvement. The technique can also improve FMAX if the compiler had previously implemented a latency=1 load operation. The f<sub>MAX</sub> increase should result in a further throughput improvement.

## Building the `local_memory_cache` Design (Linux)

1. Install the design in `build` directory from the design directory by running `cmake`:
 
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
     ./local_memory_cache.fpga_emu 
     ```

   * Generate HTML optimization reports using: 

     ```
     make report
     ``` 
     Locate the report in the `local_memory_cache_report.prj/reports/report.html` directory.
     
   * Compile and run on FPGA hardware (longer compile time, targets FPGA device) using: 

     ```
     make fpga 
     ./local_memory_cache.fpga 
     ```

(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://software.intel.com/content/dam/develop/external/us/en/documents/local_memory_cache.fpga.tar.gz" download>here</a>.


## Building the `local_memory_cache` Design (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following four build targets are provided that matches the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device) using: 

     ```
     ninja fpga_emu
     local_memory_cache.fpga_emu.exe 
     ```

   * Generate HTML optimization reports using: 
     
     ```
     ninja report
     ```
     Locate the report under the `../src/local_memory_cache_report.prj/reports/report.html` directory.

     If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/local_memory_cache_s10_pac_report.prj/reports/report.html`.

     ```
     ninja report_s10_pac
     ```

   * **Not supported yet:** Compile and run on FPGA hardware (longer compile time, targets FPGA device) using: 

## Building the `local_memory_cache` Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
