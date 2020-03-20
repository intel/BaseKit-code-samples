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
constexpr access::mode sycl_discard_write = access::mode::discard_write;

#define VECTOR_SIZE (64)  // VECTOR_SIZE of vectors a and r

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

class SimpleMath;

int getGoldenResult(int input) {
  constexpr int preadd[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
  static int coeff[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
  int acc = 0;
  int mul = input;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    acc += (coeff[i] * (mul + preadd[i]));
  }

  int tmp = coeff[0];
  for (int i = 0; i < VECTOR_SIZE - 1; i++) {
    coeff[i] = coeff[i + 1];
  }
  coeff[VECTOR_SIZE - 1] = tmp;

  return acc;
}

int main(int argc, char* argv[]) {
  int data_size = 1000000;
  if (argc > 1) {
    std::string option(argv[1]);
    if (option == "-h" || option == "--help") {
      std::cout << "Usage: \n";
      std::cout << "<executable> <data size>\n";
      std::cout << "\n";
      return 0;
    } else {
      data_size = std::stoi(option);
    }
  }
  std::vector<int> vec_a(data_size);
  std::vector<int> vec_r(data_size);
  // Fill vectors a and b with random float values
  for (int i = 0; i < data_size; i++) {
    vec_a[i] = rand() % 128;
  }

  try {
    // Device buffers
    buffer<int, 1> device_a(vec_a.data(), data_size);
    buffer<int, 1> device_r(vec_r.data(), data_size);

    auto property_list =
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};

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
      return 1;
    }

    event queue_event = deviceQueue->submit([&](handler& cgh) {
      // Data accessors
      auto a = device_a.get_access<sycl_read>(cgh);
      auto r = device_r.get_access<sycl_discard_write>(cgh);

      // Kernel
      // Task version
      cgh.single_task<class SimpleMath>([=]() {
        constexpr int preadd[] = {
            1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
        [[intelfpga::register]] int coeff[] = {
            1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
        for (int k = 0; k < data_size; ++k) {
          int acc = 0;
          int mul = a[k];
#pragma unroll
          for (int i = 0; i < VECTOR_SIZE; i++) {
#ifdef USE_FPGA_REG
            mul = intel::fpga_reg(mul);
            acc = intel::fpga_reg(acc) + (coeff[i] * (mul + preadd[i]));
#else
            acc += (coeff[i] * (mul + preadd[i]));
#endif
          }

          int tmp = coeff[0];
#pragma unroll
          for (int i = 0; i < VECTOR_SIZE - 1; i++) {
            coeff[i] = coeff[i + 1];
          }
          coeff[VECTOR_SIZE - 1] = tmp;

          r[k] = acc;
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

    // Two nested loops with 2 Fadd and 1FMul in the innermost loop
    const int numOpsPerKernel = data_size * VECTOR_SIZE * 3;
    std::cout << "Throughput for kernel with data size " << data_size
              << " and VECTOR_SIZE " << VECTOR_SIZE << ": ";
    std::cout << std::fixed << std::setprecision(6)
              << ((double)numOpsPerKernel / kernel_time) / 1e6f << " GFlops\n";
  } catch (cl::sycl::exception const& e) {
    std::cout << "caught a sycl exception:" << std::endl
              << e.what() << std::endl;
  }

  // Test the results
  bool correct = true;

  for (int i = 0; i < data_size; i++) {
    if (getGoldenResult(vec_a[i]) != vec_r[i]) {
      std::cout << "Found mismatch at " << i << ", "
                << getGoldenResult(vec_a[i]) << " != " << vec_r[i] << std::endl;
      correct = false;
    }
  }

  if (correct) {
    std::cout << "PASSED: Results are correct.\n";
  } else {
    std::cout << "FAILED: Results are incorrect.\n";
    return 1;
  }

  return 0;
}
