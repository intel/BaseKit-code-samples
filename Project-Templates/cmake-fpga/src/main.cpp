//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// located in $ONEAPI_ROOT/compiler/latest/linux/include/sycl/
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <cstdlib>
#include <iostream>

using namespace cl::sycl;
int main() {
  // create device selector for the device of your interest
  // FPGA_EMULATOR defined in cmake-fpga/src/CMakeLists.txt
#if defined(FPGA_EMULATOR)
  // DPC++ extension: FPGA emulator selector on systems without FPGA card
  intel::fpga_emulator_selector device_selector;
#else
  // DPC++ extension: FPGA selector on systems with FPGA card
  intel::fpga_selector device_selector;
#endif

  // print output message
  std::cout << "Hello World!" << std::endl;

  // note this template does NOT contain an FPGA kernel
  
  return (EXIT_SUCCESS);
}
