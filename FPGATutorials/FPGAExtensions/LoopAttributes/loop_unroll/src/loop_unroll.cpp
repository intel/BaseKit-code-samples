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
using namespace cl::sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

template <int N>
class SimpleVadd;

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

template <int UNROLL_FACTOR>
void vec_add(const std::vector<float>& VA, const std::vector<float>& VB,
             std::vector<float>& VC, int n) {
  auto property_list =
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
  event queue_event;
  try {
// Initialize queue with device selector and enabling profiling
#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
    host_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    std::unique_ptr<queue> deviceQueue;

    try {
      deviceQueue.reset(
          new queue(device_selector, exception_handler, property_list));
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
      return;
    }

    buffer<float, 1> bufferA(VA.data(), n);
    buffer<float, 1> bufferB(VB.data(), n);
    buffer<float, 1> bufferC(VC.data(), n);

    queue_event = deviceQueue->submit([&](handler& cgh) {
      auto accessorA = bufferA.get_access<sycl_read>(cgh);
      auto accessorB = bufferB.get_access<sycl_read>(cgh);
      auto accessorC = bufferC.get_access<sycl_write>(cgh);
      auto n_items = n;
      cgh.single_task<SimpleVadd<UNROLL_FACTOR> >([=]() {
#pragma unroll UNROLL_FACTOR
        for (int k = 0; k < n_items; k++) {
          accessorC[k] = accessorA[k] + accessorB[k];
        }
      });
    });

    deviceQueue->wait_and_throw();
    cl_ulong startk = queue_event.template get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    cl_ulong endk = queue_event.template get_profiling_info<
        cl::sycl::info::event_profiling::command_end>();
    // unit of startk and endk is nano second, convert to ms
    double kernel_time = (double)(endk - startk) * 1e-6f;

    std::cout << "UNROLL_FACTOR " << UNROLL_FACTOR
              << "kernel time : " << kernel_time << " ms\n";
    std::cout << "Throughput for kernel with UNROLL_FACTOR " << UNROLL_FACTOR
              << ": ";
    std::cout << std::fixed << std::setprecision(3)
              << ((double)n / kernel_time) / 1e6f << " GFlops\n";
  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
    std::terminate();
  }
}

int main(int argc, char* argv[]) {
  int array_size = 67108864;

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

  std::vector<float> A(array_size);
  std::vector<float> B(array_size);

  std::vector<float> C1(array_size);
  std::vector<float> C2(array_size);
  std::vector<float> C3(array_size);
  std::vector<float> C4(array_size);
  std::vector<float> C5(array_size);

  for (int i = 0; i < array_size; i++) {
    A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  std::cout << "Input Array Size:  " << array_size << "\n";

  // Instantiate kernel logic with unroll factor 1, 2, 4, 8
  vec_add<1>(A, B, C1, array_size);
  vec_add<2>(A, B, C2, array_size);
  vec_add<4>(A, B, C3, array_size);
  vec_add<8>(A, B, C4, array_size);
  vec_add<16>(A, B, C5, array_size);

  // Verify result
  for (unsigned int i = 0; i < array_size; i++) {
    if (C1[i] != A[i] + B[i] || C1[i] != C2[i] || C1[i] != C3[i] ||
        C1[i] != C4[i] || C1[i] != C5[i]) {
      std::cout << "FAILED: The results are incorrect" << std::endl;
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct" << std::endl;
  return 0;
}
