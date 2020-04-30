//==============================================================
// Copyright © 2019,2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>

constexpr int kRowLength = 128;
constexpr int kMinSafelen = 1;
constexpr int kMaxSafelen = kRowLength;
constexpr int kMatrixSize = kRowLength * kRowLength;

using namespace cl::sycl;

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

template <int SAFELEN>
void TransposeAndFold(const device_selector &selector,
                      const std::array<float, kMatrixSize> &m_input,
                      std::array<float, kMatrixSize> &m_output) {
  double kernel_time = 0;
  try {
    auto property_list =
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    event kernel_event;
    queue device_queue(selector, exception_handler, property_list);
    buffer<float, 1> buffer_input(m_input.data(), kMatrixSize);
    buffer<float, 1> buffer_output(m_output.data(), kMatrixSize);
    /* Run the base version without the ivdep attribute for performance
     * evaluation */
    kernel_event = device_queue.submit([&](handler &cgh) {
      auto accessor_input = buffer_input.get_access<access::mode::read>(cgh);
      auto accessor_output = buffer_output.get_access<access::mode::write>(cgh);
      cgh.single_task<KernelCompute<SAFELEN>>([=]() [[intel::kernel_args_restrict]] {
        float in_buffer[kRowLength][kRowLength];
        float temp_buffer[kRowLength][kRowLength];
        /* Initialize local buffers */
        for (int i = 0; i < kMatrixSize; i++) {
          in_buffer[i / kRowLength][i % kRowLength] = accessor_input[i];
          temp_buffer[i / kRowLength][i % kRowLength] = 0;
        }
        /* No iterations of the following loop store data into the same memory
         * location */
        /* that are less than kRowLength iterations apart. */
        [[intelfpga::ivdep(SAFELEN)]] for (int j = 0;
                                           j < kMatrixSize * kRowLength; j++) {
#pragma unroll
          for (int i = 0; i < kRowLength; i++) {
            temp_buffer[j % kRowLength][i] += in_buffer[i][j % kRowLength];
          }
        }
        /* Write result to output */
        for (int i = 0; i < kMatrixSize; i++) {
          accessor_output[i] = temp_buffer[i / kRowLength][i % kRowLength];
        }
      });
    });

    device_queue.wait_and_throw();
    cl_ulong start_k = kernel_event.template get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    cl_ulong end_k = kernel_event.template get_profiling_info<
        cl::sycl::info::event_profiling::command_end>();
    /* unit is nano second, convert to ms */
    kernel_time = (double)(end_k - start_k) * 1e-6f;
  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught synchronous SYCL exception:\n" << e.what() << '\n';
    std::terminate();
  }

  std::cout << "SAFELEN: " << SAFELEN << " -- kernel time : " << kernel_time
            << " ms\n";
  std::cout << "Throughput for kernel with SAFELEN " << SAFELEN << ": ";
  std::cout << std::fixed << std::setprecision(0)
            << (((double)kMatrixSize * sizeof(float) * 1e-3f) /
                (kernel_time * 1e-3f))
            << "KB/s\n";
}

int main(int argc, char **argv) {
  std::array<float, kMatrixSize> A, B, C;
  // Initialize input with random data
  for (int i = 0; i < kMatrixSize; i++) {
    A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

#if defined(FPGA_EMULATOR)
  const device_selector &selector = intel::fpga_emulator_selector{};
#elif defined(CPU_HOST)
  const device_selector &selector = host_selector{};
#else
  const device_selector &selector = intel::fpga_selector{};
#endif

  // Instantiate kernel logic with the min and max correct safelen parameter.
  TransposeAndFold<kMinSafelen>(selector, A, B);
  TransposeAndFold<kMaxSafelen>(selector, A, C);

  // Verify result
  for (int i = 0; i < kMatrixSize; i++) {
    if (B[i] != C[i]) {
      std::cout << "FAILED: The results are incorrect" << '\n';
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct" << '\n';
  return 0;
}
