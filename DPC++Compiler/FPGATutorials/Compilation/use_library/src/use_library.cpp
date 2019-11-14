//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include "lib.h"

using namespace cl::sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
class kernelCompute;

constexpr unsigned N = 5;

using IntArray = std::array<cl_int, N>;

int main() {
  IntArray A = {1, 2, 3, 4, 5};
  IntArray B;

  //all device related code that can throw exception
  try {
    range<1> numOfItems{N};
    buffer<cl_int, 1> bufferA(A.data(), numOfItems);
    buffer<cl_int, 1> bufferB(B.data(), numOfItems);

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
      deviceQueue.reset( new queue(device_selector, async_handler{}) );
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught a synchronous SYCL exception: " << std::endl << e.what() << std::endl;
      std::cout << "If you are targeting an FPGA hardware, please "
                   "ensure that your system is plugged to an FPGA board that is set up correctly." << std::endl;
      std::cout << "If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR." << std::endl;
      std::cout << "If you are targeting a CPU host device, compile with -DCPU_HOST." << std::endl;
      return 1;
    }

    deviceQueue->submit([&](handler& cgh) {
      auto accessorA = bufferA.get_access<sycl_read>(cgh);
      auto accessorB = bufferB.get_access<sycl_write>(cgh);

      cgh.single_task<class kernelCompute>([=]() {
        for (int i = 0; i < N; ++i) {
          accessorB[i] = my_func(accessorA[i]);
        }
      });
    });

    deviceQueue->throw_asynchronous();
  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught a SYCL exception: " << std::endl << e.what() << std::endl;
    return 1;
  }

  bool passed = true;
  for (int i = 0; i < N; ++i) {
    passed &= (B[i] == A[i]*A[i]);
  }

  if (passed) {
    printf("PASSED: results are correct\n");
  } else {
    printf("FAILED: results are incorrect\n");
    return 1;
  }

  return 0;
}
