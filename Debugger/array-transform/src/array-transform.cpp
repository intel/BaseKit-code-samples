//==============================================================
// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// This is a simple DPC++ program that accompanies the Getting Started
// Guide of the debugger.  The kernel does not compute anything
// particularly interesting; it is designed to illustrate the most
// essential features of the debugger when the target device is CPU or
// GPU.

#include <CL/sycl.hpp>
#include <iostream>

// A device function, called from inside the kernel.
static int GetDim(cl::sycl::id<1> wi, int dim) {
  return wi[dim];
}

int main() {
  using namespace cl::sycl;

  constexpr size_t length = 64;
  int input[length];
  int output[length];

  // Initialize the input
  for (unsigned int i = 0; i < length; i++)
    input[i] = i + 100;

  auto exception_handler = [](sycl::exception_list exceptionList) {
    for (std::exception_ptr const& e : exceptionList) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& e) {
        std::terminate();
      }
    }
  };

  try {
    queue q(exception_handler);
    std::cout << "[SYCL] Using device: ["
              << q.get_device().get_info<info::device::name>()
              << "]\n";

    range<1> data_range{length};
    buffer<int> buffer_in{&input[0], data_range};
    buffer<int> buffer_out{&output[0], data_range};

    q.submit([&](handler& h) {
      auto in = buffer_in.get_access<access::mode::read>(h);
      auto out = buffer_out.get_access<access::mode::write>(h);

      // kernel-start
      h.parallel_for(data_range, [=](id<1> index) {
        int id0 = GetDim(index, 0);
        int element = in[index];  // breakpoint-here
        int result = element + 50;
        if (id0 % 2 == 0) {
          result = result + 50;  // then-branch
        } else {
          result = -1;  // else-branch
        }
        out[index] = result;
      });
      // kernel-end
    });

    q.wait_and_throw();
  } catch (sycl::exception const& e) {
    std::cout << "fail; synchronous exception occurred: " << e.what() << "\n";
    return -1;
  }

  // Verify the output
  for (unsigned int i = 0; i < length; i++) {
    int result = (i % 2 == 0) ? (input[i] + 100) : -1;
    if (output[i] != result) {
      std::cout << "fail; element " << i << " is " << output[i] << "\n";
      return -1;
    }
  }

  std::cout << "success; result is correct.\n";
  return 0;
}
