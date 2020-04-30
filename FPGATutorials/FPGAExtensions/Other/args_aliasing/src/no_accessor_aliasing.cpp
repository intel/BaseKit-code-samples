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
constexpr access::mode kSyclRead = access::mode::read;
constexpr access::mode kSyclWrite = access::mode::write;

class SimpleVadd;

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
      std::terminate();
    }
  }
};

void VecAdd(const std::vector<float> &V_A, const std::vector<float> &V_B,
            std::vector<float> &V_C, int n) {
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

  queue deviceQueue(device_selector, exception_handler, property_list);

  buffer<float, 1> buffer_A(V_A.data(), n);
  buffer<float, 1> buffer_B(V_B.data(), n);
  buffer<float, 1> buffer_C(V_C.data(), n);
  queue_event = deviceQueue.submit([&](handler &cgh) {
    auto accessor_A = buffer_A.get_access<kSyclRead>(cgh);
    auto accessor_B = buffer_B.get_access<kSyclRead>(cgh);
    auto accessor_C = buffer_C.get_access<kSyclWrite>(cgh);
    auto n_items = n;
    cgh.single_task<SimpleVadd>([=]() {
      for (int i = 0; i < n_items; i += 4) {
        accessor_C[i] = accessor_A[i] + accessor_B[i];
        accessor_C[i + 1] = accessor_A[i + 1] + accessor_B[i + 1];
        accessor_C[i + 2] = accessor_A[i + 2] + accessor_B[i + 2];
        accessor_C[i + 3] = accessor_A[i + 3] + accessor_B[i + 3];
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

  std::vector<float> input_A(array_size);
  std::vector<float> input_B(array_size);
  std::vector<float> output(array_size);

  for (int i = 0; i < array_size; i++) {
    input_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    input_B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  std::cout << "Input Array Size:  " << array_size << "\n";

  try {
    VecAdd(input_A, input_B, output, array_size);
  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::cout << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly\n";
    std::cout << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR\n";
    std::cout << "   If you are targeting a CPU host device, compile with "
                 "-DCPU_HOST\n";
    return 1;
  }

  // Verify result
  for (unsigned int i = 0; i < array_size; i++) {
    if (output[i] != input_A[i] + input_B[i]) {
      std::cout << "FAILED: The results are incorrect\n";
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct\n";
  return 0;
}
