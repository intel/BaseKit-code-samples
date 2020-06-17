# FPGA Tutorial: Memory Attributes

This tutorial demonstrates an example of using memory attributes in your
SYCL™ FPGA program with their potential benefits and trade-offs. You will
learn the following:

1. Basic concepts about memory attributes.
2. How to apply memory attributes in your program.
3. How to confirm that the memory attributes were respected by the compiler.
4. Develop a basic understanding of the Fmax/area trade-off between single-pumped and double-pumped memories. 

| Optimized for   | Description
---               |---
| OS              | Linux Ubuntu 18.04; Windows* 10 or Windows* Server 2016
| Hardware        | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA
| Software        | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

Lets get started!

## What Are Memory Attributes?

For maximizing throughput, the datapath of your design should have stall-free
accesses to all its memory systems. A memory read or write is said to be
stall-free if the compiler can prove that it has contention-free access to a
memory port. A memory system is stall-free if all its accesses have this
property. Wherever possible, the compiler will try to create a minimum-area
stall-free memory system. If a different area performance trade-off is desired,
or if the compiler fails to find the best configuration, you can use the memory
attributes to override the compiler’s decisions and specify the memory
configuration you need.

Memory Attributes provide a mechanism to control the architecture of kernel
memory synthesized by the compiler. They can be applied to any variable or
array defined within the kernel, and to struct data members in struct
declarations. The compiler supports the following memory attributes:

| Memory Attribute                 | Description
---                                |---
| intelfpga::register              | Forces a variable or array to be carried through the pipeline in registers.
| intelfpga::memory("`impl_type`") | Forces a variable or array to be implemented as embedded memory. The optional string parameter `impl_type` can be `BLOCK_RAM` or `MLAB`.
| intelfpga::numbanks(N)           | Specifies that the memory implementing the variable or array must have N memory banks. 
| intelfpga::bankwidth(W)          | Specifies that the memory implementing the variable or array must be W bytes wide.
| intelfpga::singlepump            | Specifies that the memory implementing the variable or array should be clocked at the same rate as the accesses to it.
| intelfpga::doublepump            | Specifies that the memory implementing the variable or array should be clocked at twice the rate as the accesses to it.
| intelfpga::max_replicates(N)     | Specifies that a maximum of N replicates should be created to enable simultaneous reads from the datapath.
| intelfpga::private_copies(N)     | Specifies that a maximum of N private copies should be created to enable concurrent execution of N pipelined threads.
| intelfpga::simple_dual_port      | Specifies that the memory implementing the variable or array should have no port that services both reads and writes.
| intelfpga::merge("`key`", "`type`")  | Merge two or more variables or arrays in the same scope width-wise or depth-wise. All variables with the same `key` string are merged into the same memory system. The string `type` can be either `width` or `depth`. 
| intelfpga::bank_bits(b<sub>0</sub>,b<sub>1</sub>,...,b<sub>n</sub>)  | Specifies that the local memory addresses should use bits (b<sub>0</sub>,b<sub>1</sub>,...,b<sub>n</sub>) for bank-selection, where (b<sub>0</sub>,b<sub>1</sub>,...,b<sub>n</sub>) are indicated in terms of word-addressing. The bits of the local memory address not included in (b<sub>0</sub>,b<sub>1</sub>,...,b<sub>n</sub>) will be used for word-selection in each bank. 

### Example: Applying Memory Attributes to Private Arrays
```c++
device_queue.submit([&](handler &cgh) {
  cgh.single_task<class KernelCompute>([=]() {
    // create a kernel memory 8 bytes wide (2 integers per memory word)
    // and split the contents into 2 banks (each bank will contain 32
    // integers in 16 memory words). 
    [[intelfpga::bankwidth(8), intelfpga::numbanks(2)]] int a[64];
    
    // Force array 'b' to be carried live in the data path using
    // registers. 
    [[intelfpga::register]] int b[64];

    // merge 'mem_A' and 'mem_B' width-wise so that they are mapped
    // to the same kernel memory system
    [[intelfpga::merge("mem", "width")]] unsigned short mem_A[64];
    [[intelfpga::merge("mem", "width")]] unsigned short mem_B[64];
    // ...
  });
});

```

### Example: Applying Memory Attributes to Struct Data Members
```c++
// memory attributes can be specified for struct data members
// with the struct declaration.
struct State {
  [[intelfpga::numbanks(2)]] int mem[64];
  [[intelfpga::register]] int reg[8];
};

device_queue.submit([&](handler &cgh) {
  cgh.single_task<class KernelCompute>([=]() {
    // The compiler will create two memory systems from S1:
    //  - S1.mem[64] implemented in kernel memory that has 2 banks
    //  - S1.reg[8] implemented in registers 
    State S1;
    
    // In this case, we have attributes on struct declaration as
    // well as struct instantiation. When this happpens, the outer
    // level attribute takes precendence. Here, the compiler will
    // generate a single memory system for S2 which will have 4
    // banks.  
    [[intelfpga::numbanks(4)]] State S2;

    // ...
  });
});

```

You can confirm that the compiler respected your attributes by looking at the
reports. To see the potential benefits and trade-offs between the attributes,
let's look at the tutorial design.

## Tutorial Code Description

In this tutorial, we will show the trade-off between choosing a single-pumped
and double-pumped memory system for your kernel. We will apply the
attributes `[[intelfpga::singlepump]]` and `[[intelfpga::doublepump]]`, one at
a time, to a two dimensional array named `dict_offset` (dimensions `kRows x
kVec`). Array `dict_offset` has the following accesses:

 - It is initialized by copying the contents of global memory `dict_offset_init`
   using kVec writes.
 - It is dynamically read (i.e. the locations being read from are not known at
   compile time) kVec\*kVec times.
 - There are kVec dynamic writes updating the values at some indices.

Overall, there are 2\*kVec writes and kVec\*kVec reads for `dict_offset`. Note
that for both single-pumped and double-pumped versions, we apply memory
attributes to request the compiler to implement `dict_offset` in MLABs (as the
size of the array is small) using kVec banks, with each bank having no more
than kVec replicates (see the explanation at the end of the document to know
why kVec banks are requested).

## Build And Verify

### Building the Tutorial (Linux)

Install the tutorial into `build` directory from the design directory by running
`cmake`:

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

This will generate a `Makefile`. Compile the tutorial using the generated
`Makefile`. The following four targets are provided, matching the recommended
development flow:

  - Compile and run for emulation (fast compile time, targets emulated FPGA
    device). This step only tests the basic functionality of the code.
    ```
    make fpga_emu
    ./memory_attributes.fpga_emu 
    ```
    You should see the following output:
    ```
    PASSED: all kernel results are correct.
    ```

  - Generate HTML performance reports. This target generates HTML reports for
    two variants of the tutorial: one variant uses a single-pumped memory system
    for  `dict_offset` and the other uses a double-pumped memory system. Find the
    reports, respectively, in `singlepump_report.prj/reports/` and
    `doublepump_report.prj/reports/`. You can use the reports to
    verify that the compiler respected the attributes. For more information, see
    the section 'Using Reports to Verify the Design' below.
    ```
    make report
    ``` 

  - Compile and run on an FPGA hardware (longer compile time, targets FPGA
    device). This step also produces two variants, one variant uses a
    single-pumped memory system for  `dict_offset` and the other uses a
    double-pumped memory system.
    ```
    make fpga 
    ./memory_attributes_singlepump.fpga
    ./memory_attributes_doublepump.fpga
    ```
    You should see the following output for each executable run:
    ```
    PASSED: all kernel results are correct.
    ```

  - Compile and run on CPU hardware (not optimized):
    ```
    make cpu_host
    ./memory_attributes.cpu_host 
    ```

### Building the Tutorial (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

Enter source file directory.

```
cd src
```

Compile the tutorial. The following four targets are provided, matching the recommended
development flow:

  - Compile and run for emulation (fast compile time, targets emulated FPGA
    device). This step only tests the basic functionality of the code.
    ```
    ninja fpga_emu
    memory_attributes.fpga_emu.exe 
    ```
    You should see the following output:
    ```
    PASSED: all kernel results are correct.
    ```

  - Generate HTML performance reports. This target generates HTML reports for
    two variants of the tutorial: one variant uses a single-pumped memory system
    for  `dict_offset` and the other uses a double-pumped memory system. Find the
    reports, respectively, in `singlepump_report.prj/reports/` and `doublepump_report.prj/reports/`. You can use the reports to verify that the compiler respected the attributes. For more information, see the section 'Using Reports to Verify the Design' below.
    ```
    ninja report
    ``` 

    If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the reports in `../src/singlepump_s10_pac_report.prj/reports/report.html` and `../src/doublepump_s10_pac_report.prj/reports/report.html`.

    ```
    ninja report_s10_pac
    ```


  - **Not supported yet:** Compile and run on an FPGA hardware

  - Compile and run on CPU hardware (not optimized):
    ```
    ninja cpu_host
    memory_attributes.cpu_host.exe 
    ```

## Building the Tutorial in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)

### Using Reports to Verify the Design

Open the reports and go to the Kernel Memory Viewer (System Viewers > Kernel
Memory Viewer). In the Kernel Memory List pane, click `dict_offset` under
function `KernelCompute`. 

For both single-pumped and double-pumped versions, the compiler generates kVec
banks and implements the memory in MLABs, as was requested using attributes.
The difference between both the memory systems becomes clear once you look
into the number of replicates within each bank. To see the number of
replicates per bank, click any bank label (say Bank 0) under `dict_offset`. You
can see that for the single-pumped memory system, the compiler created 4
replicates per bank, whereas for the double-pumped memory system, the compiler
created 2 replicates per bank. This happens because a single-pumped replicate
has 2 physical ports and a double-pumped replicates has 4 physical ports.
Since the number of accesses to a bank are the same for both
versions, the single-pumped version needs twice the number of replicates to
create a stall-free memory system as compared to the double-pumped version. 

This also means that different amount of hardware resources were used to
generate the stall-free memory systems. Open the reports, go to Area Analysis
of System (Area Analysis > Area Analysis of System) and click "Expand All".
For the single-pumped version, you can see that the compiler used 32 MLABs to
implement the memory system for `dict_offset`, whereas for the double-pumped
version, the compiler used 16 MLABs. For the double-pumped version, the
compiler uses ALUTs and FFs to implement the logic required to double-pump a
memory. No ALUTs or FFs were used in single-pumped version.

The use of double-pumped memories might impact the Fmax of your system. Since
double-pumped memories have to be clocked at twice the frequency of your
kernel, the maximum frequency of your kernel is now limited to half the
maximum clock rate of the memory system. For this tutorial, using
double-pumped memories hurts the Fmax a lot: the single-pumped version has a
Fmax of 307.9 MHz, whereas the double-pumped version has a Fmax of 200.0
MHz.

The table that follows summarizes the Fmax (in MHz) and the number of MLABs used
in generating a stall-free memory system for `dict_offset` on
Intel® Programmable Acceleration Card with Intel® Arria® 10 GX FPGA.

Variant  | Fmax (MHz) | \# MLABs used 
------------- | ------------- | --------
memory_attributes_singlepump  | 307.9 | 32 
memory_attributes_doublepump  | 200.0 | 16 

Note that the numbers reported in the table above may vary slightly from
compile to compile.

## Summary

In this tutorial, we explored the trade-offs between implementing a memory
system as a single-pumped or a double-pumped memory:

 - the number of block RAMs or MLABs consumed by a double-pumped memory system
   is usually less then the that of a single-pumped memory system.
 - a double-pumped memory system always uses extra ALUTs and FFs
   to implement the logic required for clocking the memory at twice the rate
   of the kernel.
 - the Fmax of a kernel with a double-pumped memory system is limited to half
   the maximum clock rate of the memory system. There is no such restriction 
   for single-pumped memories.

As we saw in this tutorial, there are often many ways to generate a
stall-free memory system. As a programmer, the implementation you choose
depends on the constraints you need to follow:

 - if your design is consuming more memory resources (such as block RAMs or
   MLABs) than provided in the FPGA, using double-pumped memory systems might
   make your design fit.
 - if the Fmax of your design is limited because you have double-pumped memory
   systems in your kernel, forcing all memory systems to be single-pumped
   might increase the Fmax.

Finally, feel free to experiment with the tutorial code. You can:
 
 - see the memory system generated by the compiler if you do not specify any
   attributes.
 - change the memory implementation type to block RAMs (using
   `[[intelfpga::memory("BLOCK_RAM")]]`) or registers (using
   `[[intelfpga::register]]`) to see how it affects the area and Fmax of the
   tutorial design.
 - vary kRows and/or kVec (both in powers of 2) to see how it effects the
   trade-off between single-pumped and double-pumped memories.

## Aside: Why do we create kVec banks with a maximum of kVec replicates per bank?

If you list the accesses to `dict_offset` in the code, you will get the
following:

 - kVec writes to `dict_offset[i][k]`, with k=0,..,kVec-1
 - kVec reads from `dict_offset[hash[0]][k]`, with k=0,..,kVec-1
 - kVec reads from `dict_offset[hash[1]][k]`, with k=0,..,kVec-1
 - kVec reads from `dict_offset[hash[2]][k]`, with k=0,..,kVec-1
 - kVec reads from `dict_offset[hash[3]][k]`, with k=0,..,kVec-1
 - kVec writes to `dict_offset[hash[k]][k]`, with k=0,..,kVec-1

Note that whenever `dict_offset` is accessed, the second dimension is always
known at compile time. Hence, if we choose a memory system such that array
elements `dict_offset[:][0]` (`:` denotes all indices in range) are contained
in Bank 0, `dict_offset[:][1]` are contained in Bank 1 and so on, each access
(read or write) needs to go to only one bank. This is achieved by requesting
the compiler to generate kVec banks.

Also, from the access pattern, it is clear that there are kVec reads from each
bank. To make these reads stall-free, we request kVec replicates per bank so that
(if needed) each read can happen simultaneously from a separate replicate. Note
that since all replicates in a bank should contain identical data at all times,
a write to a bank has to go to all replicates. For single-pumped memories, each
replicate has 2 physical ports. In the tutorial code, one of these ports is used
for writing. Hence, the compiler generates kVec replicates per bank to create
stall-free accesses for kVec reads. For double-pumped memories, each replicate
effectively has 4 ports, three of which are available for reads. Hence, the
compiler needs fewer replicates per bank to create stall-free reads.
