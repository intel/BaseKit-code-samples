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

// tolerance used in floating point comparisons
constexpr float kTol = 0.001;

// array size of vectors a, b and c
constexpr unsigned int kArraySize = 32;

class SimpleAdd;

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
      std::terminate();
    }
  }
};

int main() {
  std::vector<float> vec_a(kArraySize);
  std::vector<float> vec_b(kArraySize);
  std::vector<float> vec_r(kArraySize);
  // Fill vectors a and b with random float values
  int count = kArraySize;
  for (int i = 0; i < count; i++) {
    vec_a[i] = rand() / (float)RAND_MAX;
    vec_b[i] = rand() / (float)RAND_MAX;
  }

  try {
    // Device buffers
    buffer<float, 1> device_a(vec_a.data(), kArraySize);
    buffer<float, 1> device_b(vec_b.data(), kArraySize);
    buffer<float, 1> device_r(vec_r.data(), kArraySize);

#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
    host_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    std::unique_ptr<queue> q;

    // Catch device seletor runtime error
    try {
      q.reset(new queue(device_selector, exception_handler));
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught a synchronous SYCL exception:\n" << e.what() << "\n";
      std::cout << "If you are targeting an FPGA hardware, please "
                   "ensure that your system is plugged to an FPGA board that "
                   "is set up correctly.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
      std::cout << "If you are targeting a CPU host device, compile with "
                   "-DCPU_HOST.\n";
      return 1;
    }

    q->submit([&](handler& h) {
      // Data accessors
      auto a = device_a.get_access<sycl_read>(h);
      auto b = device_b.get_access<sycl_read>(h);
      auto r = device_r.get_access<sycl_write>(h);
      // Kernel
      h.single_task<class SimpleAdd>([=]() {
        for (int i = 0; i < kArraySize; ++i) {
          r[i] = a[i] + b[i];
        }
      });
    });

    q->throw_asynchronous();

  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n" << e.what() << "\n";
    std::terminate();
  }

  // Test the results
  int correct = 0;
  for (int i = 0; i < count; i++) {
    float tmp = vec_a[i] + vec_b[i] - vec_r[i];
    if (tmp * tmp < kTol * kTol) {
      correct++;
    }
  }
  // summarize results
  if (correct == count) {
    std::cout << "PASSED: results are correct\n";
  } else {
    std::cout << "FAILED: results are incorrect\n";
  }

  return !(correct == count);
}
