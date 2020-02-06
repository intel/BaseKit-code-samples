//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

constexpr int values[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34};
constexpr size_t num_values = sizeof(values) / sizeof(int);

// ############################################################
// bootstrap_function initializes the result buffer

void bootstrap_function(int* result) {
  for (size_t i = 0; i < num_values; ++i) {
    result[i] = values[i];
  }
}

// ############################################################
// Body of the application

void work(queue& q, int* result) {
  buffer<int, 1> my_buffer(result, range<1>(num_values));
  q.submit([&](handler& h) {
    auto my_accessor = my_buffer.get_access<access::mode::read_write>(h);
    h.single_task<class bootstrap>(
        [=]() { bootstrap_function(&my_accessor[0]); });
  });
  q.wait_and_throw();
}

// ############################################################
// entry point for the program

int main(int argc, char** argv) {
  int result[num_values];

  auto exception_handler = [](cl::sycl::exception_list exceptionList) {
    for (std::exception_ptr const& e : exceptionList) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::cout << "ASYNCHRONOUS SYCL exception:\n" << e.what() << std::endl;
        std::terminate();  // exit the process immediately.
      }
    }
  };

  try {
    queue q(default_selector{}, exception_handler);

    std::cout << "Device : " << q.get_device().get_info<info::device::name>()
              << std::endl;

    work(q, result);

    std::cout << "Number of values: " << num_values << std::endl;
    for (auto i = 0; i < num_values; ++i) {
      std::cout << result[i] << " ";
      // Signal an error if the result does not match the expected values
      if (result[i] != values[i]) throw -1;
    }
    std::cout << std::endl;

  } catch (...) {
    std::cout << "Failure" << std::endl;
    std::terminate();
  }

  std::cout << "Success" << std::endl;
  return 0;
}
