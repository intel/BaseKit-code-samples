//==============================================================
// Copyright © 2019,2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <array>
#include <iomanip>
#include <iostream>

using namespace cl::sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

constexpr unsigned SIZE = 8096;
constexpr unsigned MAX_ITER = 50000;
constexpr unsigned N = MAX_ITER * SIZE;
constexpr unsigned MAXVAL = 128;

using floatArray = std::array<cl_float, SIZE>;
using floatScalar = std::array<cl_float, 1>;

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
      std::terminate();
    }
  }
};

template <int N>
class kernelCompute;

template <int CONCURRENCY>
void partial_sum_with_shift(const device_selector &selector,
                            const floatArray &A, const float shift,
                            floatScalar &R) {
  double kernel_time = 0.0;
  try {
    auto property_list =
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    std::unique_ptr<queue> deviceQueue;
    event kernel_event;
    deviceQueue.reset(new queue(selector, property_list));

    range<1> numOfItems{SIZE};
    buffer<cl_float, 1> bufferA(A.data(), numOfItems);
    range<1> one{1};
    buffer<cl_float, 1> bufferR(R.data(), one);
    kernel_event = deviceQueue->submit([&](handler &cgh) {
      auto accessorA = bufferA.get_access<sycl_read>(cgh);
      auto accessorR = bufferR.get_access<sycl_write>(cgh);
      cgh.single_task<kernelCompute<CONCURRENCY>>([=]() {
        float result = 0;
        [[intelfpga::max_concurrency(CONCURRENCY)]] for (unsigned i = 0;
                                                         i < MAX_ITER; i++) {
          float a1[SIZE];
          for (int j = 0; j < SIZE; j++)
            a1[j] = accessorA[(i * 4 + j) % SIZE] * shift;
          for (int j = 0; j < SIZE; j++) result += a1[j];
        }
        accessorR[0] = result;
      });
    });
    deviceQueue->wait_and_throw();
    cl_ulong startk = kernel_event.template get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    cl_ulong endk = kernel_event.template get_profiling_info<
        cl::sycl::info::event_profiling::command_end>();
    /* unit is nano second, convert to ms */
    kernel_time = (double)(endk - startk) * 1e-6f;
  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught a SYCL exception:" << std::endl
              << e.what() << std::endl;
    return;
  }

  std::cout << "MAX CONCURRENCY " << CONCURRENCY
            << "kernel time : " << kernel_time << " ms\n";
  std::cout << "Throughput for kernel with MAX_CONCURRENCY " << CONCURRENCY
            << ": ";
  std::cout << std::fixed << std::setprecision(3)
            << ((double)(N * 2) / kernel_time) / 1e6f << " GFlops\n";
}

float golden_result(const floatArray &A, const float shift) {
  float gr = 0;
  for (unsigned i = 0; i < MAX_ITER; i++) {
    float a1[SIZE];
    for (int j = 0; j < SIZE; j++) a1[j] = A[(i * 4 + j) % SIZE] * shift;
    for (int j = 0; j < SIZE; j++) gr += a1[j];
  }
  return gr;
}

int main(int argc, char **argv) {
  bool success = true;

  floatArray A;
  floatScalar R0, R1, R2, R3, R4, R5;

  const float shift = (float)(rand() % MAXVAL);

  for (unsigned int i = 0; i < SIZE; i++) A[i] = rand() % MAXVAL;

#if defined(FPGA_EMULATOR)
  const device_selector &selector = intel::fpga_emulator_selector{};
#elif defined(CPU_HOST)
  const device_selector &selector = host_selector{};
#else
  const device_selector &selector = intel::fpga_selector{};
#endif

  partial_sum_with_shift<0>(selector, A, shift, R0);
  partial_sum_with_shift<1>(selector, A, shift, R1);
  partial_sum_with_shift<2>(selector, A, shift, R2);
  partial_sum_with_shift<4>(selector, A, shift, R3);
  partial_sum_with_shift<8>(selector, A, shift, R4);
  partial_sum_with_shift<16>(selector, A, shift, R5);

  // compute the actual result here
  float gr = golden_result(A, shift);

  if (gr != R0[0]) {
    std::cout << "Max Concurrency 0: mismatch: " << R0[0] << " != " << gr
              << " (kernel != expected)" << std::endl;
    success = false;
  }

  if (gr != R1[0]) {
    std::cout << "Max Concurrency 1: mismatch: " << R1[0] << " != " << gr
              << " (kernel != expected)" << std::endl;
    success = false;
  }

  if (gr != R2[0]) {
    std::cout << "Max Concurrency 2: mismatch: " << R2[0] << " != " << gr
              << " (kernel != expected)" << std::endl;
    success = false;
  }

  if (gr != R3[0]) {
    std::cout << "Max Concurrency 4: mismatch: " << R3[0] << " != " << gr
              << " (kernel != expected)" << std::endl;
    success = false;
  }

  if (gr != R4[0]) {
    std::cout << "Max Concurrency 8: mismatch: " << R4[0] << " != " << gr
              << " (kernel != expected)" << std::endl;
    success = false;
  }

  if (gr != R5[0]) {
    std::cout << "Max Concurrency 16: mismatch: " << R5[0] << " != " << gr
              << " (kernel != expected)" << std::endl;
    success = false;
  }

  if (success) {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  }

  return 1;
}
