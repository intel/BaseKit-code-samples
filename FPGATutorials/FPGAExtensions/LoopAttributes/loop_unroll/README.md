# FPGA Tutorial: Unrolling Loops

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria®; 10 GX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates a simple example of unrolling loops to improve the throughput of a DPC++ FPGA program. 

## Key Concepts
This tutorial helps you learn the following concepts:
* Basics of loop unrolling.
* Unrolling loops in your program.
* Determining the correct unroll factor for your program.


## Building the `loop_unroll` Design (Linux)

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

2. Compile the design using the generated `Makefile`. The following four build targets are provided that match the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulates an FPGA device) using: 

     ```
     make fpga_emu
     ./loop_unroll.fpga_emu 
     ```
   
   * Generate HTML optimization reports using: 

     ```
     make report
     ```
     Locate the report in the `loop_unroll_report.prj/reports/report.html` directory.
     
   
   * Compile and run on FPGA hardware (longer compile time, targets an FPGA device) using: 

     ```
     make fpga 
     ./loop_unroll.fpga 
     ```
     >**NOTE**: Only the FPGA hardware flow illustrates the performance difference among different `unroll` factors.
     
(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://software.intel.com/content/dam/develop/external/us/en/documents/loop_unroll.fpga.tar.gz" download>here</a>.

## Building the `loop_unroll` Design (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following four build targets are provided that match the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulates an FPGA device) using: 

     ```
     ninja fpga_emu
     ./loop_unroll.fpga_emu.exe
     ```
   
   * Generate HTML optimization reports. 

     ```
     ninja report
     ```
     Locate the report the `../src/loop_unroll_report.prj/reports/report.html` directory.

     If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/loop_unroll_s10_pac_report.prj/reports/report.html`.

     ```
     ninja report_s10_pac
     ```

   * **Not supported yet:** Compile and run on FPGA hardware.
     

## Building the `loop_unroll` Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Basics of Loop Unrolling
Use the loop unrolling mechanism to increase program parallelism by duplicating the compute logic within a loop. The number of times the loop logic duplicates is referred as the *unroll factor*. Depending on whether the *unroll factor* is equal to the number of loop iterations or not, loop unroll methods can be categorized as *full-loop unrolling* and *partial-loop unrolling*.

### Example: Full-Loop Unrolling
```c++
// Before unrolling loop
#pragma unroll
for(i = 0 ; i < 5; i++){
  a[i] += 1;
}

// After fully unrolling the loop by a factor of 5, the loop is flattened. There is no loop after unrolling.

a[0] += 1;
a[1] += 1;
a[2] += 1;
a[3] += 1;
a[4] += 1;

```


### Example: Partial-Loop Unrolling

```c++
// Before unrolling loop
#pragma unroll 4
for(i = 0 ; i < 20; i++){
  a[i] += 1;
}

// After the loop is unrolled by a factor of 4, the loop has five (20 / 4) iterations
for(i = 0 ; i < 5; i++){
  a[i * 4] += 1;
  a[i * 4 + 1] += 1;
  a[i * 4 + 2] += 1;
  a[i * 4 + 3] += 1;
}
```

You can observe that a full unroll is a special case where the unroll factor is equal to the number of loop iterations.

In the partial unroll example, each loop iteration in second one is equivalent to four iterations in the first. The Intel® oneAPI DPC++ Compiler (Beta) instantiates four adders instead of one adder. Because there is no data dependency between iterations in the loop (which is true in this case), the compiler executes four adds in parallel.

In an FPGA design, unrolling loops is a common strategy to trade on-chip resources for throughput. This tutorial demonstrates this trade-off with a simple vector add example.



## Testing the Design
1. In the `loop_unroll.cpp` file, apply unroll factors of 1, 2, 4, 8, and 16 on the kernel.
    ```c++
    vec_add<1>(A, B, C1, array_size);
    vec_add<2>(A, B, C2, array_size);
    vec_add<4>(A, B, C3, array_size);
    vec_add<8>(A, B, C4, array_size);
    vec_add<16>(A, B, C5, array_size);
   ```
2. Compile for FPGA hardware and execute on FPGA. Refer to *Building the `loop_unroll` Design* section.

   If nothing goes wrong, you can see the following output in the console:

   > `PASSED: The results are correct`

3. View the FPGA resource use in the report generated by the compiler. The following table summarizes the execution time (in ms), throughput (in GFlops), and number of DSPs used for unroll factors of 1, 2, 4, 8, and 16 for a default input array size of 64M floats (2 ^ 26 floats) on Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA:

Unroll Factor  | Kernel Time (ms) | Throughput (GFlops) | Num of DSPs
------------- | ------------- | -----------------------| -------
1   | 242 | 0.277 | 1
2   | 127 | 0.528 | 2
4   | 63  | 1.065 | 4
8   | 46  | 1.459 | 8
16  | 44  | 1.525 | 16

When the unroll factor goes from 1 -> 2 -> 4, the kernel execution time decreases by a factor of two. Kernel throughput also doubles every time.

However, when the unroll factor goes from 4 -> 8 -> 16, the throughput change does not scale by a factor of 2 at each step. The design is now bound by memory bandwidth limitations instead of compute unit limitations even though the hardware is replicated.

## Determining the Correct Unroll Factor
As a programmer, you might wonder how to select the correct unroll factor for a specific design. The intent is to improve throughput while maximizing resource utilization and not be limited by memory bandwidth.

The memory bandwidth on an Intel® Programmable Acceleration Card with Intel Arria® 10 GX FPGA system is about 6 GB/s. The design is running at about 300 MHz according to the compiler report. In this design, the FPGA design can process new iterations in every cycle in a pipelined fashion. The theoretical computation limit for 1 adder is:

**GFlops**: 300 MHz \* 1 float = 0.3 GFlops

**Computation Bandwidth**: 300 MHz \* 1 float * 4 Bytes   = 1.2 GB/s

You can do the same calculation for different unroll factors:

Unroll Factor  | GFlops (GB/s) | Compuation Bandwidth (GB/s)
------------- | ------------- | -----------------------
1   | 0.3 | 1.2
2   | 0.6 | 2.4
4   | 1.2 | 4.8
8   | 2.4 | 9.6
16  | 4.8 | 19.2

You can observe that the program is memory-bandwidth limited when unroll factor grows from 4 -> 8. The gain is no longer obvious after that. This matches what you observe in the *Testing the Design* section.

## 
