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

constexpr unsigned kSize = 8192;
constexpr unsigned kMaxIter = 50000;
constexpr unsigned kTotalOps = 2 * kMaxIter * kSize;
constexpr unsigned kMaxValue = 128;

using FloatArray = std::array<cl_float, kSize>;
using FloatScalar = std::array<cl_float, 1>;

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << '\n';
      std::terminate();
    }
  }
};

template <int N>
class KernelCompute;

template <int CONCURRENCY>
void PartialSumWithShift(const device_selector &selector,
                         const FloatArray &array, const float shift,
                         FloatScalar &result) {
  double kernel_time = 0.0;

  try {
    auto property_list =
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    std::unique_ptr<queue> device_queue;

    event kernel_event;
    device_queue.reset(new queue(selector, exception_handler, property_list));

    range<1> number_of_items{kSize};
    buffer<cl_float, 1> buffer_array(array.data(), number_of_items);
    range<1> one{1};
    buffer<cl_float, 1> buffer_result(result.data(), one);

    kernel_event = device_queue->submit([&](handler &cgh) {
      auto accessor_array = buffer_array.get_access<sycl_read>(cgh);
      auto accessor_result = buffer_result.get_access<sycl_write>(cgh);
      cgh.single_task<KernelCompute<CONCURRENCY>>([=
      ]() [[intel::kernel_args_restrict]] {
        float r = 0;

        [[intelfpga::max_concurrency(CONCURRENCY)]] for (unsigned i = 0;
                                                         i < kMaxIter; i++) {
          float a1[kSize];
          for (int j = 0; j < kSize; j++)
            a1[j] = accessor_array[(i * 4 + j) % kSize] * shift;
          for (int j = 0; j < kSize; j++) r += a1[j];
        }
        accessor_result[0] = r;
      });
    });

    device_queue->wait_and_throw();

    cl_ulong startk = kernel_event.template get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    cl_ulong endk = kernel_event.template get_profiling_info<
        cl::sycl::info::event_profiling::command_end>();
    /* unit is nano second, convert to ms */
    kernel_time = (double)(endk - startk) * 1e-6f;

  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught a SYCL exception:" << '\n' << e.what() << '\n';
    return;
  }

  std::cout << "MAX CONCURRENCY " << CONCURRENCY << " "
            << "kernel time : " << kernel_time << " ms\n";
  std::cout << "Throughput for kernel with MAX_CONCURRENCY " << CONCURRENCY
            << ": ";
  std::cout << std::fixed << std::setprecision(3)
            << ((double)(kTotalOps) / kernel_time) / 1e6f << " GFlops\n";
}

float GoldenResult(const FloatArray &A, const float shift) {
  float gr = 0;
  for (unsigned i = 0; i < kMaxIter; i++) {
    float a1[kSize];
    for (int j = 0; j < kSize; j++) a1[j] = A[(i * 4 + j) % kSize] * shift;
    for (int j = 0; j < kSize; j++) gr += a1[j];
  }
  return gr;
}

int main(int argc, char **argv) {
  bool success = true;

  FloatArray A;
  FloatScalar R0, R1, R2, R3, R4, R5;

  const float shift = (float)(rand() % kMaxValue);

  for (unsigned int i = 0; i < kSize; i++) A[i] = rand() % kMaxValue;

#if defined(FPGA_EMULATOR)
  const device_selector &selector = intel::fpga_emulator_selector{};
#else
  const device_selector &selector = intel::fpga_selector{};
#endif

  PartialSumWithShift<0>(selector, A, shift, R0);
  PartialSumWithShift<1>(selector, A, shift, R1);
  PartialSumWithShift<2>(selector, A, shift, R2);
  PartialSumWithShift<4>(selector, A, shift, R3);
  PartialSumWithShift<8>(selector, A, shift, R4);
  PartialSumWithShift<16>(selector, A, shift, R5);

  // compute the actual result here
  float gr = GoldenResult(A, shift);

  if (gr != R0[0]) {
    std::cout << "Max Concurrency 0: mismatch: " << R0[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R1[0]) {
    std::cout << "Max Concurrency 1: mismatch: " << R1[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R2[0]) {
    std::cout << "Max Concurrency 2: mismatch: " << R2[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R3[0]) {
    std::cout << "Max Concurrency 4: mismatch: " << R3[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R4[0]) {
    std::cout << "Max Concurrency 8: mismatch: " << R4[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (gr != R5[0]) {
    std::cout << "Max Concurrency 16: mismatch: " << R5[0] << " != " << gr
              << " (kernel != expected)" << '\n';
    success = false;
  }

  if (success) {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  }

  return 1;
}
