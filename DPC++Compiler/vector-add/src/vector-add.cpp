//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/intel/fpga_extensions.hpp>
#endif
#include <array>
#include <iostream>

using namespace cl::sycl;

// Convenience data access definitions
constexpr access::mode dp_read = access::mode::read;
constexpr access::mode dp_write = access::mode::write;

// ARRAY type & data size for use in this example
constexpr size_t array_size = 10000;
typedef std::array<int, array_size> IntArray;

// output message for runtime exceptions
#define EXCEPTION_MSG \
  "    If you are targeting FPGA hardware, please ensure that an FPGA board is plugged to the system, \n\
        set up correctly and compile with -DFPGA  \n\
    If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR.\n"

//************************************
// Function description: initialize the array from 0 to array_size-1
//************************************
void initialize_array(IntArray &a) {
  for (size_t i = 0; i < a.size(); i++) a[i] = i;
}

//************************************
// Function description: create a device queue with the default selector or
// explicit FPGA selector when FPGA macro is defined
//    return: DPC++ queue object
//************************************
queue create_device_queue() {
  // create device selector for the device of your interest
#ifdef FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card
  intel::fpga_emulator_selector dselector;
#elif defined(FPGA)
  // DPC++ extension: FPGA selector on systems with FPGA card
  intel::fpga_selector dselector;
#else
  // the default device selector: it will select the most performant device
  // available at runtime.
  default_selector dselector;
#endif

  // create an async exception handler so the program fails more gracefully.
  auto ehandler = [](cl::sycl::exception_list exceptionList) {
    for (std::exception_ptr const &e : exceptionList) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const &e) {
        std::cout << "Caught an asynchronous DPC++ exception, terminating the "
                     "program."
                  << std::endl;
        std::cout << EXCEPTION_MSG;
        std::terminate();
      }
    }
  };

  try {
    // create the devices queue with the selector above and the exception
    // handler to catch async runtime errors the device queue is used to enqueue
    // the kernels and encapsulates all the states needed for execution
    queue q(dselector, ehandler);

    return q;
  } catch (cl::sycl::exception const &e) {
    // catch the exception from devices that are not supported.
    std::cout << "An exception is caught when creating a device queue."
              << std::endl;
    std::cout << EXCEPTION_MSG;
    std::terminate();
  }
}

//************************************
// Compute vector addition in DPC++ on device: sum of the data is returned in
// 3rd parameter "sum_parallel"
//************************************
void VectorAddInDPCPP(const IntArray &addend_1, const IntArray &addend_2,
                      IntArray &sum_parallel) {
  queue q = create_device_queue();

  // print out the device information used for the kernel code
  std::cout << "Device: " << q.get_device().get_info<info::device::name>()
            << std::endl;

  // create the range object for the arrays managed by the buffer
  range<1> num_items{array_size};

  // create buffers that hold the data shared between the host and the devices.
  //    1st parameter: pointer of the data;
  //    2nd parameter: size of the data
  // the buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer<int, 1> addend_1_buf(addend_1.data(), num_items);
  buffer<int, 1> addend_2_buf(addend_2.data(), num_items);
  buffer<int, 1> sum_buf(sum_parallel.data(), num_items);

  // submit a command group to the queue by a lambda function that
  // contains the data access permission and device computation (kernel)
  q.submit([&](handler &h) {
    // create an accessor for each buffer with access permission: read, write or
    // read/write. The accessor is the only means to access the memory in the
    // buffer.
    auto addend_1_accessor = addend_1_buf.get_access<dp_read>(h);
    auto addend_2_accessor = addend_2_buf.get_access<dp_read>(h);

    // the sum_accessor is used to store (with write permision) the sum data
    auto sum_accessor = sum_buf.get_access<dp_write>(h);

    // Use parallel_for to run array addition in parallel on device. This
    // executes the kernel.
    //    1st parameter is the number of work items to use
    //    2nd parameter is the kernel, a lambda that specifies what to do per
    //    work item. the parameter of the lambda is the work item id of the
    //    current item.
    // DPC++ supports unnamed lambda kernel by default.
    h.parallel_for(num_items, [=](id<1> i) {
      sum_accessor[i] = addend_1_accessor[i] + addend_2_accessor[i];
    });
  });

  // q.submit() is an asynchronous call. DPC++ runtime enqueues and runs the
  // kernel asynchronously. At the end of the DPC++ scope the buffer's data is
  // copied back to the host.
}

//************************************
// Demonstrate summation of arrays both in scalar on CPU and parallel on device
//************************************
int main() {
  // create int array objects with "array_size" to store the input and output
  // data
  IntArray addend_1, addend_2, sum_scalar, sum_parallel;

  // Initialize input arrays with values from 0 to array_size-1
  initialize_array(addend_1);
  initialize_array(addend_2);

  // Compute vector addition in DPC++
  VectorAddInDPCPP(addend_1, addend_2, sum_parallel);

  // Computes the sum of two arrays in scalar for validation
  for (size_t i = 0; i < sum_scalar.size(); i++)
    sum_scalar[i] = addend_1[i] + addend_2[i];

  // Verify that the two sum arrays are equal
  for (size_t i = 0; i < sum_parallel.size(); i++) {
    if (sum_parallel[i] != sum_scalar[i]) {
      std::cout << "fail" << std::endl;
      return -1;
    }
  }
  std::cout << "success" << std::endl;

  return 0;
}
