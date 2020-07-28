# FPGA Tutorial: Data Transfers Using Pipe Arrays

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel Arria® 10 GX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates how a single kernel in a SYCL* FPGA program can transfer
data to multiple kernels by using an array of pipes. 

# Key Concepts
This tutorial covers the following concepts:
* Basic definition of pipes
* Determine how to declare pipes

## What Are Arrays of Pipes?
In SYCL*, each pipe defines a unique type with static methods for reading data
(`read`) and writing data (`write`). Since pipes are just types (as opposed to
objects), defining a collection of pipes is slghtly non-intuitive but yields highly
efficient code.

This tutorial provides a convenient pair of header files defining an
abstraction for an array of pipes. The headers can be used in any SYCL* design
and be expanded as necessary.

### Example 1: Defining an Array of Pipes in SYCL*

Include the top-level header in your design:

```c++
#include "pipe_array.hpp"
```

Just like pipes, an array of pipes needs template parameters for an ID, for the
`min_capacity` and for the data type. The number of pipes is required as an 
extra parameter for arrays. The following code declares an array of four pipes
with `capacity=4` that operate on `int` values, as shown in the following code snippet:

```c++
using ProducerToConsumerPipeMatrix = PipeArray< // Defined in "pipe_array.h".
    class ProducerConsumerPipe,                 // An identifier for the pipe.
    uint64_t,                                   // The type of data in the pipe.
    kDepth,                                      // The capacity of each pipe.
    kNumRows,                                   // array dimension.
    kNumCols                                    // array dimension.
    >;
```

The uniqueness of a pipe array is derived from a combination of all template
parameters.

Indexing inside a pipe array can be done with the `PipeArray::PipeAt` type alias,
as shown in the following code snippet:

```c++
ProducerToConsumerPipeMatrix::PipeAt<0,0>::write(17);
auto x = ProducerToConsumerPipeMatrix::PipeAt<0,0>::read();
```
The pipe being used must be known at compile time. Therefore, the index is
passed to `PipeAt` through a template parameter.

While it is possible to use `PipeAt` to write or read from individual pipes in
the array, all pipes in the array can be written or read from using a static
form of loop unrolling, as shown in the following code snippet:

```c++
Unroller<0, kNumRows>::step([&input_idx, input_accessor](auto i_idx) {
  constexpr int i = i_idx.value;

  Unroller<0, kNumCols>::step([&input_idx, input_accessor](auto j_idx) {
    constexpr int j = j_idx.value;
    ProducerToConsumerPipeMatrix::PipeAt<i,j>::write(17);
  }
}
```

There are many powerful C++ libraries capable of doing this. This design 
includes a simple header file `unroller.hpp`, which implements the `Unroller`
functionality.

### Example 2: Using a Pipe Array in SYCL*

This example defines a `Producer` kernel that reads data from host
memory and fowards this data into a two dimensional pipe matrix, 
as shown in the following code snippet:

```c++
cgh.single_task<ProducerTutorial>([=]() {
  int input_idx = 0;
  for (int pass = 0; pass < num_passes; pass++) {
    Unroller<0, kNumRows>::step([&input_idx, input_accessor](auto i_idx) {
      constexpr int i = i_idx.value;

      Unroller<0, kNumCols>::step([&input_idx, input_accessor](auto j_idx) {
        constexpr int j = j_idx.value;

        ProducerToConsumerPipeMatrix::PipeAt<i, j>::write(
            input_accessor[input_idx++]);
      });
    });
  }
});
```

This example also defines `Consumer` kernels that read from a unique pipe in
`ProducerToConsumerPipeMatrix` and process the data, writing the result to the host
memory, as shown in the following code snippet:

```c++
uint64_t ConsumerWork(uint64_t i) { return i * i; }

// ...
// Inside Consumer function:

cgh.single_task<ConsumerTutorial<ConsumerID>>([=]() {
  constexpr int consumer_x = ConsumerID / kNumCols;
  constexpr int consumer_y = ConsumerID % kNumCols;
  for (int i = 0; i < num_elements; ++i) {
    auto input = ProducerToConsumerPipeMatrix::PipeAt<consumer_x,
                                                       consumer_y>::read();
    auto answer = ConsumerWork(input);
    output_accessor[i] = answer;
  }
});
```

The host is responsible for instantiating all consumers, as shown in the 
following code snippet:

```c++
{
// Inside main:
    queue device_queue(device_selector);

    buffer<uint64_t, 1> producer_buffer(producer_input.data(), array_size);
    Producer(device_queue, producer_buffer);

    std::vector<buffer<uint64_t, 1>> consumer_buffers;
    Unroller<0, kNumberOfConsumers>::step([&](auto idx) {
      constexpr int consumer_id = idx.value;
      consumer_buffers.emplace_back(consumer_output[consumer_id].data(),
                                    items_per_consumer);
      Consumer<consumer_id>(device_queue, consumer_buffers.back());
    });

    device_queue.wait_and_throw();
}
```

## Building the `pipe_array` Design (Linux)

**NOTE:** CMake is necessary to build the design.

1. Generate the `Makefile` using:

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

2. Compile the design through the generated `Makefile`. The following three build targets are provided that matches the recommended development flow:

   * Compile and run on the FPGA emulator using:

     ```
     make fpga_emu
     ./pipe_array.fpga_emu
     ```

     If the compilation is successful, you see the following output in the console:
     > `PASSED: The results are correct`

   * Generate the HTML optimization report using:

     ```
     make report
     ```

     Locate the reports available in the `pipe_array_report.prj/reports/report.html` directory.

   * Compile and run on the FPGA hardware using:

     ```
     make fpga
     ./pipe_array.fpga
     ```

     If the compilation is successful, you see the following output in the console:
     > `PASSED: The results are correct`

(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://software.intel.com/content/dam/develop/external/us/en/documents/pipe_array.fpga.tar.gz" download>here</a>.


## Building the `pipe_array` Design (Windows)

**NOTE:** CMake is necessary to build the design.

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following three build targets are provided that matches the recommended development flow:

   * Compile and run on the FPGA emulator using:

     ```
     ninja fpga_emu
     pipe_array.fpga_emu.exe
     ```

     If the compilation is successful, you see the following output in the console:
     > `PASSED: The results are correct`

   * Generate the HTML optimization report.

     ```
     ninja report
     ```

     Locate the reports available in the `../src/pipe_array_report.prj/reports/report.html` directory.

     If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/pipe_array_s10_pac_report.prj/reports/report.html`.

     ```
     ninja report_s10_pac
     ```

   * **Not supported yet:** Compile and run on the FPGA hardware.

## Building the `pipe_array` Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
