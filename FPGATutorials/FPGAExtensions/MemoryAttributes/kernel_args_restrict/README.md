# FPGA Tutorial: Memory Attributes
This tutorial presents the usage of the `kernel_args_restrict` attribute for
your SYCL™ FPGA program and its effect on the performance of your kernel.

You will learn the following:
1. An introduction to the problem of *pointer aliasing*
2. The behaviour of the `kernel_args_restrict` attribute and when to use it on your kernel
3. The effect this attribute can have on your kernel's performance 

| Optimized for   | Description
---               |---
| OS              | Linux Ubuntu 18.04; Windows* 10 or Windows* Server 2016
| Hardware        | Intel(R) Programmable Acceleration Card (PAC) with Intel(R) Arria(R) 10 GX FPGA
| Software        | Intel(R) oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## What Is Pointer Aliasing?
[Pointer aliasing](https://en.wikipedia.org/wiki/Pointer_aliasing) is a situation where the same memory location can be accessed using different *names* (i.e. variables). For example, consider the code below. Here, the variable `pi` can be changed one of three ways: `pi=3.14159`, `*a=3.14159` or `*b=3.14159`. In general, the compiler has to be conservative about which accesses may alias to each other and avoid making optimizations that reorder and/or parallelize operations.

```c++
float pi = 3.14;
float *a = &pi;
float *b = a;
```

The code in this tutorial performs the function illustrated below. Without more information from the developer, the compiler cannot make any optimizations which overlap, vectorize or reorder the assignment operations since it cannot guarantee that `in` does not alias with `out`. For example, imagine a degenerate case where the function was called: like this `myCopy(ptr, ptr+1, 10)`. This would cause `in[i]` and `out[i+1]` to alias to the same address, for all `i` from 0 to 9.

```c++
void myCopy(int *in, int *out, unsigned int size) {
  for(unsigned int i = 0; i < size; i++) {
    out[i] = in[i];
  }
}
```

This problem requires the compiler to be conservative about pointer aliasing. For the FPGA, this results in an II increase since the next iteration of the loop cannot begin until the previous iteration has completed. However, developers can often guarantee that pointers will never alias to each other and inform the compiler that it is safe to make optimizations using this assumption (C/C++ programmers may recognize this as the [restrict keyword](https://en.wikipedia.org/wiki/Restrict)).

In your SYCL™ program, you can use the `[[intel::kernel_args_restrict]]` attribute to tell the compiler that none of the kernel arguments can alias to one another and therefore enable more aggressive optimizations. As discussed in a later section, the usage of this attribute can significantly reduce the II of the loop in the example above.

## Tutorial Code Description
In this tutorial, we will show how to use the `kernel_args_restrict` attribute for your kernel and the effect it has on performance. We show two kernels that perform the same function; one with the `[[intel::kernel_args_restrict]]` applied to it and the other without. The function of the kernel is simple: copy the contents of one buffer to another. We will analyze the effect of the `[[intel::kernel_args_restrict]]` attribute on the performance of the kernel by analyzing the loop II in the reports and the latency of the kernel on actual hardware.

## Build And Verify

### Building the Tutorial (Linux)

Install the tutorial into `build` directory from the design directory by running
`cmake`:

```
mkdir build
cd build
cmake ..
```

This will generate a `Makefile`. Compile the tutorial using the generated `Makefile`. The following four targets are provided, matching the recommended development flow:

  - Compile and run for emulation (fast compile time, targets emulated FPGA device). This step only tests the basic functionality of the code.
    ```
    make fpga_emu
    ./kernel_args_restrict.fpga_emu 
    ```
    You should see the following output (times may vary):
    ```
    Kernel throughput without attribute: 2716.29 MB/s
    Kernel throughput with attribute: 2677.04 MB/s
    PASSED
    ```

  - Generate HTML performance reports. This target generates HTML reports for the tutorial design. Find the reports in `kernel_args_restrict.prj/reports/`. You can use the reports to verify that the compiler respected the attributes. For more information, see the section 'Using Reports to Verify the Design' below.
    ```
    make report
    ``` 

  - Compile and run on an FPGA hardware (longer compile time, targets FPGA device).
    ```
    make fpga 
    ./kernel_args_restrict.fpga
    ```
    You should see the following output (times may vary):
    ```
    Kernel throughput without attribute: 8.06761 MB/s
    Kernel throughput with attribute: 766.873 MB/s
    PASSED
    ```

  - Compile and run on CPU hardware (not optimized):
    ```
    make cpu_host
    ./kernel_args_restrict.cpu_host 
    ```
  - Download the design, compiled for FPGA hardware, from this location: [download page](https://www.intel.com/content/www/us/en/programmable/products/design-software/high-level-design/one-api-for-fpga-support.html)

### Building the Tutorial (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

Enter source file directory.

```
cd src
```

Compile the tutorial. The following four targets are provided, matching the recommended
development flow:

  - Compile and run for emulation (fast compile time, targets emulated FPGA device). This step only tests the basic functionality of the code.
    ```
    ninja fpga_emu
    kernel_args_restrict.fpga_emu.exe 
    ```
    You should see the following output (times may vary):
    ```
    Kernel throughput without attribute: 2716.29 MB/s
    Kernel throughput with attribute: 2677.04 MB/s
    PASSED
    ```

  - Generate HTML performance reports. This target generates HTML reports for the tutorial design. Find the reports in `kernel_args_restrict.prj/reports/`. You can use the reports to verify that the compiler respected the attributes. For more information, see the section 'Using Reports to Verify the Design' below.
    ```
    ninja report
    ```

  - **Not supported yet:** Compile and run on an FPGA hardware

  - Compile and run on CPU hardware (not optimized):
    ```
    ninja cpu_host
    kernel_args_restrict.cpu_host.exe 
    ```

## Building the Tutorial in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel(R) oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

## Using Reports to Verify the Design

Open the HTML report (`kernel_args_restrict.prj/reports/report.html`) and go to the *Loop Analysis* (*Throughput Analysis* > *Loop Analysis*). In the *Loop List pane* on the left you should see two kernels: one is the kernel without the attribute applied (*KernelArgsNoRestrict* in the kernel name) and the other with the attribute applied (*KernelArgsRestrict* in the kernel name). The kernels each have a single for-loop, so the *Loop List* pane should show exactly one loop. Click on the loop under each kernel to see its attributes.

Compare the loop II between the two kernels. Notice that the loop in the *KernelArgsNoRestrict* kernel has an II much larger than 1 (~187), while the loop in the *KernelArgsRestrict* kernel has an II of ~1. Why is this? For the *KernelArgsNoRestrict* kernel, since we didn't tell the compiler that the kernel arguments can't alias, it must be conservative about its scheduling of operations. Since the compiler cannot guarantee that `out[i]` and `in[i+1]` won't alias, it cannot overlap the iteration of the loop performing `out[i] = in[i]` with the next iteration of the loop performing `out[i+1] = in[i+1]` (and likewise for iterations `in[i+2]`, `in[i+3]`, ...). This results in an II equal to the latency of reading `in[i]` plus the latency of writing to `out[i]` plus one to ensure the operation is finished.

We can confirm this by looking at the details of the loop. First, click on the *KernelArgsNoRestrict* kernel in the *Loop List* pane and then click on the loop in the *Loop Analysis* pane. Now look at the *Details* pane below. You should see something like:

- *Compiler failed to schedule this loop with smaller II due to memory dependency*
  - *From: Load Operation (kernel_args_restrict.cpp: 74 > accessor.hpp: 945)*
  - *To: Store Operation (kernel_args_restrict.cpp: 74)*
- *Most critical loop feedback path during scheduling:*
  - *144.00 clock cycles Load Operation (kernel_args_restrict.cpp: 74 > accessor.hpp: 945)*
  - *42.00 clock cycles Store Operation (kernel_args_restrict.cpp: 74)*


The first bullet (and its sub-bullets) tell you that there is memory dependency between the load and store operations in the loop. This is the conservative pointer aliasing memory dependency we described earlier. The second bullet shows you the estimated latencies for the load (144 cycles) and store (42 cycles) operations. As we predicted earlier, the sum of these two latencies (plus 1) is the II of the loop.

Now look at the loop details of the *KernelArgsRestrict* kernel. You will notice that the *Details* pane doesn't show a memory dependency. The usage of the `[[intel::kernel_args_restrict]]` attribute allowed the compiler to perform more aggressive optimizations and schedule a new iteration of the for-loop every cycle since it knows that accesses to `in` and `out` will never alias.

Attribute used?  | II | Kernel Throughput (MB/s)
------------- | ------------- | --------
No  | ~187 | 8
Yes  | ~1 | 767

## Summary
Due to pointer aliasing, the compiler must be conservative about optimizations that reorder, parallelize or overlap operations that could alias. This tutorial motivated and demonstrated the usage of the SYCL `[[intel::kernel_args_restrict]]` kernel attribute, which should be applied anytime you can guarantee that kernel arguments do not alias. This allows you to enable more aggressive compiler optimizations and potentially achieve better kernel performance.
