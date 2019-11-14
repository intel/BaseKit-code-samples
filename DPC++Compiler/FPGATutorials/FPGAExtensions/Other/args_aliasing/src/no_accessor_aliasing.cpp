//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#ifndef FLAG
#define FLAG 0
#endif

using namespace cl::sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

class SimpleVadd;

void vec_add(const std::vector<float> &VA, const std::vector<float> &VB,
             std::vector<float> &VC, int n) {
  auto property_list =
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
  event queue_event;
  
  // Initialize queue with device selector and enabling profiling
  #if defined(FPGA_EMULATOR)
  intel::fpga_emulator_selector device_selector;
  #elif defined(CPU_HOST)
  host_selector device_selector;
  #else
  intel::fpga_selector device_selector;
  #endif

  queue deviceQueue(device_selector, property_list);

  buffer<float, 1> bufferA(VA.data(), n);
  buffer<float, 1> bufferB(VB.data(), n);
  buffer<float, 1> bufferC(VC.data(), n);
  queue_event = deviceQueue.submit([&](handler &cgh) {
    auto accessorA = bufferA.get_access<sycl_read>(cgh);
    auto accessorB = bufferB.get_access<sycl_read>(cgh);
    auto accessorC = bufferC.get_access<sycl_write>(cgh);
    auto n_items = n;
    cgh.single_task<SimpleVadd>([=]() {
      for (int i = 0; i < n_items; i += 4) {
        accessorC[i    ] = accessorA[i    ] + accessorB[i    ];
        accessorC[i + 1] = accessorA[i + 1] + accessorB[i + 1];
        accessorC[i + 2] = accessorA[i + 2] + accessorB[i + 2];
        accessorC[i + 3] = accessorA[i + 3] + accessorB[i + 3];
      }
    });
  });

  deviceQueue.wait_and_throw();
  cl_ulong startk = queue_event.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();
  cl_ulong endk = queue_event.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();
  double kernel_time = (double)(endk - startk) * 1e-6f;

  std::cout << "kernel time : " << kernel_time << " ms\n";
#if FLAG
  std::cout << "Throughput for kernel with no-accessor-aliasing: ";
#else
  std::cout << "Throughput for kernel with accessor-aliasing: ";
#endif
  std::cout << std::fixed << std::setprecision(3)
            << ((double)(sizeof(float) * n) / kernel_time) / 1e6f << "GB/s\n";
}

int main(int argc, char *argv[]) {

  int array_size = (1 << 20);

  if (argc > 1) {
    std::string option(argv[1]);
    if (option == "-h" || option == "--help") {
      std::cout << "Usage: \n";
      std::cout << "<executable> <data size>\n";
      std::cout << "\n";
      return 0;
    } else {
      array_size = std::stoi(option);
    }
  }

  std::vector<float> InputA(array_size);
  std::vector<float> InputB(array_size);
  std::vector<float> Output(array_size);

  for (int i = 0; i < array_size; i++) {
    InputA[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    InputB[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  std::cout << "Input Array Size:  " << array_size << "\n";

  try {
    vec_add(InputA, InputB, Output, array_size);
  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cout << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cout << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;
    std::cout
        << "   If you are targeting a CPU host device, compile with -DCPU_HOST"
        << std::endl;
    return 1;
  }

  // Verify result
  for (unsigned int i = 0; i < array_size; i++) {
    if (Output[i] != InputA[i] + InputB[i]) {
      std::cout << "FAILED: The results are incorrect" << std::endl;
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct" << std::endl;
  return 0;
}
