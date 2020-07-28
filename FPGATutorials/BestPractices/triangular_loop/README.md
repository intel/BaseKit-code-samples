# FPGA Tutorial: Triangular Loop Optimization

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates how to improve the performance of nested triangular loops with loop-carried dependencies.

## Background

A triangular loop is a loop nest where the inner-loop range depends on the outer loop variable in such a way that the inner-loop trip-count shrinks or grows. This is better explained with an example:

```c++
  for (int x = 0; x < n; x++) {
    for (int y = x + 1; y < n; y++) {
      local_buf[y] = local_buf[y] + something_complicated(local_buf[x]);
    }
  }
```

In this example, the inner-loop executes fewer iterations as overall execution progresses.
Each iteration of the inner-loop performs a read from index `[x]` and a read-modify-write on indices `[y]=x+1` to `[y]=n-1`.
Expressed graphically (with n=10), these operations look like:

```c++
    y=0 1 2 3 4 5 6 7 8 9  
==========================
x=0   o x x x x x x x x x 
x=1     o x x x x x x x x
x=2       o x x x x x x x
x=3         o x x x x x x
x=4           o x x x x x
x=5             o x x x x
x=6               o x x x
x=7                 o x x
x=8                   o x
x=9       

Legend: read="o", read-modify-write="x"
```

The triangular shape of this picture is where the name comes from.

In the above example, the table shows that in outer-loop iteration `x=0`, the
program reads `local_buf[x=0]` and reads, modifies, and writes the values 
from `local_buf[y=1]` through `local_buf[y=9]`. This pattern of memory accesses 
results in a loop-carried dependency across the outer loop iterations. For 
example, the read at `x=2` depends on the value that was written at `x=1,y=2`. 

Generally, a new iteration is launched on every cycle as long as a sufficient number of inner-loop 
iterations are executed *between* any two iterations that are dependent on one another.

However, the challenge in the triangular loop pattern is that the trip-count of the inner-loop
progressively shrinks as `x` increments. In the worst case of `x=7`, the program writes to `local_buf[y=8]` in the first `y` iteration, but
have only one intervening `y` iteration at `y=9` before the value must be read again at `x=8,y=8`. This may not allow enough time
for the write operation to complete. The compiler compensates for this by increasing the initiation interval (II) of the
inner-loop to allow more time to elapse between iterations. This has the consequence of reducing the throughput of the inner-loop
by a factor of II.

A key point is that this increased II is only functionally necessary when the inner-loop trip-count becomes small.
Furthermore, the II of a loop is static and applies for all invocations of that loop.
Therefore, if the *outer-loop* trip-count is large, then most of the invocations of the inner-loop unnecessarily suffer the aforementioned throughput
degradation. The optimization technique exemplified in this tutorial addresses this throughpout degradation.

## Optimization

Since the increase of II is only necessary when the inner-loop trip-count becomes small, the optimization technique ensures that the trip-count
never falls below some minimum (M). In other words, you execute extra 'dummy' iterations of the inner-loop but skip the actual computation
during these dummy iterations. These dummy iterations allow time for the loop-carried dependency to resolve. To be clear, the extra iterations
are only executed on inner-loop invocations that require them. When the inner-loop trip-count is large, extra iterations are not required. This 
technique allows the compiler to achieve II=1. 

The value of M should be approximately equal to the II of the unoptimized inner loop, which is listed in the Loops Analysis report. Using this as a
starting value, if the compiler can achieve II=1, you can experiment to find the minimum M by reducing M until II increases. Similarly, if the compiler does not achieve II=1, you can increase
M until it does.

In the above example, the execution graph with M=6 appears as follows:

```c++
    y=0 1 2 3 4 5 6 7 8 9 
==========================
x=0   o x x x x x x x x x   
x=1     o x x x x x x x x   
x=2       o x x x x x x x   
x=3         o x x x x x x   
x=4           o x x x x x   
x=5           - o x x x x   
x=6           - - o x x x   
x=7           - - - o x x   
x=8           - - - - o x   
x=9          
              <---M=6--->

Legend: read="o", read-modify-write="x", dummy iteration="-"
```

This technique requires the nested loop to be merged into a single loop. Explicit `x` and `y` induction variables are maintained to achieve
the triangular iteration pattern.

The trip-count of the merged loop can be calculated by simply summing the total number of iterations in the above execution graph.
Consider the iterations as consisting of the following two triangles of "real" and
"dummy" iterations.

```c++
    y=0 1 2 3 4 5 6 7 8 9                     y=0 1 2 3 4 5 6 7 8 9
=========================                 =========================
x=0   o x x x x x x x x x                 x=0
x=1     o x x x x x x x x                 x=1
x=2       o x x x x x x x                 x=2
x=3         o x x x x x x                 x=3
x=4           o x x x x x       +         x=4
x=5             o x x x x                 x=5           -
x=6               o x x x                 x=6           - -
x=7                 o x x                 x=7           - - -
x=8                   o x                 x=8           - - - -
x=9 
                                                        <(M-2)>  
                                                        <---M=6--->
```
The number of iterations on the left is 10+9+8+7+6+5+4+3+2 = 54. The formula for a
descending series from `n` is `n*(n+1)/2`. Since there is no iteration at
`x=9,y=9`, subtract 1 from this formula (i.e., `n*(n+1)/2 - 1`). In the example where
n=10, this formula gives 10*11/2 - 1 = 54 as earlier.

The dummy iterations on the right are equal to 4+3+2+1 = 10. The largest number
in this series is M-2. Using the same formula for a descending series and substituting
M-2, you get `(M-2)*(M-1)/2`. For this specific example of M=6, this formula gives 4*5/2 = 10 as above.

The bound on the loop is sum of these two values: `(n-M)*(n-M+1)/2 + M*(n-1)`.
Computation inside the loop is guarded by the condition `y > x`.

Since the loop is restructured to ensure that a minimum of M iterations are executed, use the
`[[intelfpga::ivdep(M)]]` to hint to the compiler that iterations with dependencies are always
separated by at least M iterations.

## Sample Results

A test compile of this tutorial design achieved an f<sub>MAX</sub> of approximately 210 MHz on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA. The results with and without the optimization are shown in the following table:

Configuration | Overall Execution Time (ms) | Throughput (MB/s)
-|-|-
Without optimization | 4972 | 25.74
With optimization | 161 | 796.59

Without optimization, the compiler achieved an II of 30 on the inner-loop. With the optimization, the compiler achieves an II of 1 and the throughput increases by approximately 30x.

## Building the `triangular_loop` Example (Linux)

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

2. Compile the design through the generated `Makefile`. The following three targets are provided, matching the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device) using: 

      ```
      make fpga_emu
      ./triangular_loop.fpga_emu 
      ```

   * Generate HTML optimization reports using: 

     ```
        make report
        ``` 
        Locate the report in the `triangular_loop_report.prj/reports/report.html` directory.

   * Compile and run on the FPGA hardware (longer compile time, targets FPGA device) using: 

      ```
      make fpga 
      ./triangular_loop.fpga
      ```

(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://software.intel.com/content/dam/develop/external/us/en/documents/triangular_loop.fpga.tar.gz" download>here</a>.


## Building the `triangular_loop` Example (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design through the generated `Makefile`. The following three targets are provided, matching the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device) using: 

     ```
     ninja fpga_emu
     triangular_loop.fpga_emu.exe 
     ```

   * Generate HTML optimization reports
     
     ```
     ninja report
     ```
     Locate the report under the `../src/triangular_loop_report.prj/reports/report.html` directory.

     If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/triangular_loop_s10_pac_report.prj/reports/report.html`.

     ```
     ninja report_s10_pac
     ```

   * **Not supported yet:** Compile and run on the FPGA hardware (longer compile time, targets FPGA device)

## Building the `triangular_loop` Example in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
