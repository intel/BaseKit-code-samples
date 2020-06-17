//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// located in $ONEAPI_ROOT/compiler/latest/linux/include/sycl/
#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>

int main() {
  // create GPU device selector
  cl::sycl::gpu_selector device_selector;

  // print output message
  std::cout << "Hello World!" << std::endl;

  return (EXIT_SUCCESS);
}
