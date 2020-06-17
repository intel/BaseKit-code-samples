# FPGA Tutorial: Data Transfers Using Pipes

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel® Programmable Acceleration Card (PAC) with Intel®Arria® 10 GX FPGA
| Software                          | Intel® oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates how a kernel in a SYCL* FPGA program transfers
data to or from another kernel (or the host) using the pipe abstraction.

## Key Concepts
* Basic definition of pipes
* Determine how to declare pipes

## Definition of a Pipe

The primary goal of pipes is to allow concurrent execution of kernels that need
to exchange data.

A pipe is a FIFO data structure connecting two endpoints that communicate
using the pipe's `read` and `write` operations. An endpoint can be either a kernel
or the host. Therefore, there are three types of pipes: 
* kernel-kernel
* kernel-host 
* host-host 

This tutorial focuses on kernel-kernel pipes, but
the concepts discussed here apply to other kinds of pipes as well.

The `read` and `write` operations have two variants: 
* Blocking variant: Blocking operations may not return immediately, but are always successful.
* Non-blocking variant: Non-blocking operations take an extra boolean parameter
that is set to `true` if the operation happened successfuly. 

Data flows in a single direction inside pipes. In other words, for a pipe `P`
and two kernels using `P`, one of the kernels is exclusively going to perform
`write` to `P` while the other kernel is exclusively going to perform `read` from
`P`. Bydirectional communication can be achieved using two pipes.

Each pipe has a configurable `capacity` parameter describing the number of `write`
operations that may be performed without any `read` operations being performed. For example,
consider a pipe `P` with capacity 3, and two kernels `K1` and `K2` using
`P`. Assume that `K1` performed the following sequence of operations:

 `write(1)`, `write(2)`, `write(3)`

In this situation, the pipe is full, because three (the `capacity` of
`P`) `write` operations were performed without any `read` operation. In this
situation, a `read` must occur before any other `write` is allowed.

If a `write` is attempted to a full pipe, one of two behaviors occur:

  * If the operation is non-blocking, it returns immediately and its
  boolean parameter is set to `false`. The `write` does not have any effects.
  * If the operation is blocking, it does not return until a `read` is
  performed by the other endpoint. Once the `read` is performed, the `write`
  takes place.

The blocking and non-blocking `read` operations have analogous behaviors when
the pipe is empty.

### Example: Defining a Pipe in SYCL*

In SYCL*, pipes are defined as a class with static members. To declare a pipe that
works on integers with `capacity=4`, use a type alias:

```c++
using ProducerToConsumerPipe = pipe<  // Defined in the SYCL* headers.
  class ProducerConsumerPipe,         // An identifier for the pipe.
  int,                                // The type of data in the pipe.
  4>;                                 // The capacity of the pipe.
```

The `class ProducerToConsumerPipe` template parameter is important to the
uniqueness of the pipe. This class need not be defined, but must be distinct
for each pipe. Consider another type alias with the exact same parameters:

```c++
using ProducerToConsumerPipe2 = pipe<  // Defined in the SYCL* headers.
  class ProducerConsumerPipe,          // An identifier for the pipe.
  int,                                 // The type of data in the pipe.
  4>;                                  // The capacity of the pipe.
```

The uniqueness of a pipe is derived from a combination of all three template
parameters. Since `ProducerToConsumerPipe` and `ProducerToConsumerPipe2` have
the same template parameters, they define the same pipe.

### Example: Using a Pipe in SYCL*

This example defines a `Consumer` and a `Producer` kernel connected
by the pipe `ProducerToConsumerPipe`. Kernels use the
`ProducerToConsumerPipe::write` and `ProducerToConsumerPipe::read` methods for
communication.

The `Producer` kernel reads integers from the global memory and writes those integers
into `ProducerToConsumerPipe`, as shown in the following code snippet:

```c++
void Producer(const std::vector<int> &input) {
  auto property_list =
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
  queue device_queue(no_host{}, property_list);

  auto num_elements = input.size();
  buffer<int, 1> input_buffer(input.data(), num_elements);

  event queue_event;
  queue_event = device_queue.submit([&](handler &cgh) {
    auto input_accessor= input_buffer.get_access<sycl_read>(cgh);

    cgh.single_task<class ProducerTutorial>([=]() {
      for (int i = 0; i < num_elements; ++i) {
	ProducerToConsumerPipe::write(input_accessor[i]);
      }
    });

  });

  device_queue.wait_and_throw();
}
```

The `Consumer` kernel reads integers from `ProducerToConsumerPipe`, processses
the integers (`ConsumerWork(i)`), and writes the result into the global memory.

```c++
void Consumer(std::vector<int> &output) {
  auto property_list =
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
  queue device_queue(no_host{}, property_list);

  auto num_elements = output.size();
  buffer<int, 1> output_buffer(output.data(), num_elements);

  event queue_event;
  queue_event = device_queue.submit([&](handler &cgh) {
    auto output_accessor = output_buffer.get_access<sycl_write>(cgh);

    cgh.single_task<class ConsumerTutorial>([=]() {
      for (int i = 0; i < num_elements; ++i) {
	auto input = ProducerToConsumerPipe::read();
	auto answer = ConsumerWork(input);
	output_accessor[i] = answer;
      }
    });

  });

  device_queue.wait_and_throw();
}
```

**NOTE:** The `read` and `write` operations used are blocking. If
`ConsumerWork` is an expensive operation, then `Producer` might fill
`ProducerToConsumerPipe` faster than `Consumer` can read from it, causing
`Producer` to block occasionally.

## Building the Example (Linux)

**NOTE:** `CMake` is required to build the design.

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

2. Compile the design through the generated `Makefile`. The following three build targets are provided that matches the recommended development flow:

   * Compile and run on the FPGA emulator using:

      ```
      make fpga_emu
      ./pipes.fpga_emu
      ```

      If nothing goes wrong, you are expected to see the following output in the console:
      > `PASSED: The results are correct`

   * Generate the HTML optimization report using:

     ```
     make report
     ```

     The reports are  available in the `pipes_report.prj/reports/report.html` directory.

   * Compile and run on the FPGA hardware using:

     ```
     make fpga
     ./pipes.fpga
     ```

     If nothing goes wrong, you are expected to see the following output in the console:
     > `PASSED: The results are correct`


(Optional) As the above hardware compile may take several hours to complete, an Intel® PAC with Intel Arria® 10 GX FPGA precompiled binary can be downloaded <a href="https://www.intel.com/content/dam/altera-www/global/en_US/others/support/examples/download/pipes.fpga" download>here</a>.

## Building the Example (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following three build targets are provided that matches the recommended development flow:
   * Compile and run on the FPGA emulator using:

      ```
      ninja fpga_emu
      pipes.fpga_emu.exe
      ```

      If nothing goes wrong, you are expected to see the following output in the console:
      > `PASSED: The results are correct`

   * Generate the HTML optimization report.

     ```
     ninja report
     ```

     The reports are  available in the `../src/pipes_report.prj/reports/report.html` directory.

     If you are targeting the Intel® PAC with Intel Stratix® 10 SX FPGA, please use the following target and find the report in `../src/pipes_s10_pac_report.prj/reports/report.html`.

     ```
     ninja report_s10_pac
     ```

   * **Not supported yet:** Compile and run on the FPGA hardware.

## Building the Example in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel® oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
