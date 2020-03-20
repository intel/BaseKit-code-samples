//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <iostream>
#include <vector>
using namespace cl::sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

#define TOL (0.001)      // tolerance used in floating point comparisons
#define ARRAY_SIZE (32)  // ARRAY_SIZE of vectors a, b and c

class SimpleAdd;

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
      std::terminate();
    }
  }
};

int main() {
  std::vector<float> vec_a(ARRAY_SIZE);
  std::vector<float> vec_b(ARRAY_SIZE);
  std::vector<float> vec_r(ARRAY_SIZE);
  // Fill vectors a and b with random float values
  int count = ARRAY_SIZE;
  for (int i = 0; i < count; i++) {
    vec_a[i] = rand() / (float)RAND_MAX;
    vec_b[i] = rand() / (float)RAND_MAX;
  }

  try {
    // Device buffers
    buffer<float, 1> device_a(vec_a.data(), ARRAY_SIZE);
    buffer<float, 1> device_b(vec_b.data(), ARRAY_SIZE);
    buffer<float, 1> device_r(vec_r.data(), ARRAY_SIZE);

#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
    host_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    std::unique_ptr<queue> deviceQueue;

    // Catch device seletor runtime error
    try {
      deviceQueue.reset(new queue(device_selector, exception_handler));
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught a synchronous SYCL exception:" << std::endl
                << e.what() << std::endl;
      std::cout << "If you are targeting an FPGA hardware, please "
                   "ensure that your system is plugged to an FPGA board that "
                   "is set up correctly."
                << std::endl;
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR."
                << std::endl;
      std::cout
          << "If you are targeting a CPU host device, compile with -DCPU_HOST."
          << std::endl;
      return 1;
    }

    deviceQueue->submit([&](handler& cgh) {
      // Data accessors
      auto a = device_a.get_access<sycl_read>(cgh);
      auto b = device_b.get_access<sycl_read>(cgh);
      auto r = device_r.get_access<sycl_write>(cgh);
      // Kernel
      cgh.single_task<class SimpleAdd>([=]() {
        for (int k = 0; k < ARRAY_SIZE; ++k) {
          r[k] = a[k] + b[k];
        }
      });
    });

    deviceQueue->throw_asynchronous();

  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
    std::terminate();
  }

  // Test the results
  int correct = 0;
  float tmp;
  for (int i = 0; i < count; i++) {
    tmp = vec_a[i] + vec_b[i];
    tmp -= vec_r[i];
    if (tmp * tmp < TOL * TOL) {
      correct++;
    } else {
      std::cout << "FAILED: results are incorrect\n";
    }
  }
  // summarize results
  if (correct == count) {
    std::cout << "PASSED: results are correct\n";
  }

  return !(correct == count);
}
