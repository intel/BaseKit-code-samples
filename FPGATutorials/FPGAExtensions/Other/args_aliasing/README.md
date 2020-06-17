# FPGA Tutorial: Applying `-no-accessor-aliasing` on Kernel Arguments

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates a simple example of using the
`-no-accessor-aliasing` compiler flag to improve the throughput of a SYCL*
FPGA program. 

## Key Concepts
You will learn the following concepts:

* Basic definition of the `-no-accessor-aliasing` compiler flag.
* Determine how to use the flag in a SYCL* program.

## Using the `-no-accessor-aliasing` Flag
The `-no-accessor-aliasing` flag helps in informing the compiler that pointer
arguments in a kernel and all pointers derived from them, never refers to
the same memory location. With this information, the compiler can
generate more efficient hardware. This flag affects all kernels in the program.

### Example: Kernel Pointer in SYCL*
```c++
queue_event = device_queue.submit([&](handler &cgh) {
  auto accessor_A = buffer_A.get_access<sycl_read>(cgh);
  auto accessor_B = buffer_B.get_access<sycl_read>(cgh);
  auto accessor_C = buffer_C.get_access<sycl_write>(cgh);
  auto n_items = n;
  cgh.single_task<class SimpleVadd>([=]() {
    for (int i = 0; i < n_items; i += 4) {
      accessor_C[i    ] = accessor_A[i    ] + accessor_B[i    ];
      accessor_C[i + 1] = accessor_A[i + 1] + accessor_B[i + 1];
      accessor_C[i + 2] = accessor_A[i + 2] + accessor_B[i + 2];
      accessor_C[i + 3] = accessor_A[i + 3] + accessor_B[i + 3];
    }
  });
});
```

In the `for` loop above, the same three operations are repeated four times with
different indices:
  - A load from `accessor_A`
  - A load from `accessor_B`
  - A store to `accessor_C`

The compiler is not able to prove that `accessor_A` and `accessor_B` at index `i`
do not point to the same memory location as `accessor_C` at index `i+1`. In
fact, it cannot prove that `accessor_A`, `accessor_B`, and `accessor_C` never
points to the same memory location under any combination of indices. This
causes the compiler to be conservative about possible dependencies between
the memory instructions. For example, eight loads cannot perform simultaneously as
soon as a loop iteration starts, because one of the stores might change the
value of a load.

By applying the `-no-accessor-aliasing` flag, the compiler is assured 
that `accessor_A`, `accessor_B`, and `accessor_C` never points to the same
memory location. This information allows the compiler to parallelize most of
the memory operations contained in this loop. For example, all eight loads can now
perform in parallel as soon as the loop iteration starts, since no store in
the loop ever changes the value returned by the loads.

## Building the Tutorial (Linux)

The kernel in `no_accessor_aliasing.cpp` contains the same loop as the example
above. The goal is to compile this kernel twice, once without any flags and
once with the `-no-accessor-aliasing` flag. By doing this, a significant increase in
throughput is observed.

**NOTE**: CMake is required to build the design.

1. Generate the `Makefile`.

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

2. Test correctness with emulation.

```
make fpga_emu
./args_aliasing.fpga_emu
```

You are expected to see the following output in the console:

> `RESULT: The results are correct`

3. Generate the HTML optimization report:

```
make report
```

The reports are available in the `no_accessor_aliasing_alias_report.prj/` and
`no_accessor_aliasing_noalias_report.prj/` directories. You can compare how the two designs
differ in various optimization metrics.

4. Compile and run on FPGA hardware:

```
make fpga
```

This generates two executables, `args_aliasing_alias.fpga` and
`args_aliasing_noalias.fpga`. Run both of them:

```
./args_aliasing_alias.fpga
./args_aliasing_noalias.fpga
```

You are expected to see the following output in the console for each executable:

> `RESULT: The results are correct`



The following table summarizes the execution time (in ms) and throughput (in
GB/s) for the same kernel compiled with and without the `-no-accessor-aliasing`
flag on the Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA:

With aliasing flag? | Kernel Time (ms) | Throughput (GB/s)
------------- | ------------- | -----------------------
No   | 560  | 4.039
Yes   | 1 | 0.008


## Building the Tutorial (Windows)

The kernel in `no_accessor_aliasing.cpp` contains the same loop as the example
above. The goal is to compile this kernel twice, once without any flags and
once with the `-no-accessor-aliasing` flag. By doing this, a significant increase in
throughput is observed.

**NOTE**: CMake is required to build the design.

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Test correctness with emulation.

```
ninja fpga_emu
args_aliasing.fpga_emu.exe 
```

You are expected to see the following output in the console:

> `RESULT: The results are correct`

3. Generate the HTML optimization report:

```
ninja report
```

The reports are available in the `args_aliasing_alias_report.prj/` and
`args_aliasing_noalias_report.prj/` directories. You can compare how the two designs
differ in various optimization metrics.

If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the reports in `../src/args_aliasing_alias_restrict_s10_pac_report.prj/reports/report.html` and `../src/args_aliasing_noalias_restrict_s10_pac_report.prj/reports/report.html`.

```
ninja report_s10_pac
```

4. **Not supported yet:** Compile and run on FPGA hardware:

## Building the Tutorial in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
