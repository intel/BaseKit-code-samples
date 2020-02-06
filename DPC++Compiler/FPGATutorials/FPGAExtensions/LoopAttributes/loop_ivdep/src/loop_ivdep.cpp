//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>

constexpr int rowLength = 128;
constexpr int minSafelen = 1;
constexpr int maxSafelen = rowLength;
constexpr int matrixSize = rowLength * rowLength;

using namespace cl::sycl;

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
class TransposeAndFold;

template <int SAFELEN>
void transpose_and_fold(const device_selector &selector,
                        const std::array<float, matrixSize> &MInput,
                        std::array<float, matrixSize> &MOutput) {
  double kernelTime = 0;
  try {
    auto propertyList =
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    event kernelEvent;
    queue deviceQueue(selector, exception_handler, propertyList);
    buffer<float, 1> bufferInput(MInput.data(), matrixSize);
    buffer<float, 1> bufferOutput(MOutput.data(), matrixSize);
    /* Run the base version without the ivdep attribute for performance
     * evaluation */
    kernelEvent = deviceQueue.submit([&](handler &cgh) {
      auto accessorInput = bufferInput.get_access<access::mode::read>(cgh);
      auto accessorOutput = bufferOutput.get_access<access::mode::write>(cgh);
      cgh.single_task<TransposeAndFold<SAFELEN>>([=]() {
        float inBuff[rowLength][rowLength];
        float tmpBuff[rowLength][rowLength];
        /* Initialize local buffers */
        for (int i = 0; i < matrixSize; i++) {
          inBuff[i / rowLength][i % rowLength] = accessorInput[i];
          tmpBuff[i / rowLength][i % rowLength] = 0;
        }
        /* No iterations of the following loop store data into the same memory
         * location */
        /* that are less than rowLength iterations apart. */
        [[intelfpga::ivdep(SAFELEN)]]
        for (int j = 0; j < matrixSize * rowLength; j++) {
          #pragma unroll
          for (int i = 0; i < rowLength; i++) {
            tmpBuff[j % rowLength][i] += inBuff[i][j % rowLength];
          }
        }
        /* Write result to output */
        for (int i = 0; i < matrixSize; i++) {
          accessorOutput[i] = tmpBuff[i / rowLength][i % rowLength];
        }
      });
    });

    deviceQueue.wait_and_throw();
    cl_ulong startk = kernelEvent.template get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    cl_ulong endk = kernelEvent.template get_profiling_info<
        cl::sycl::info::event_profiling::command_end>();
    /* unit is nano second, convert to ms */
    kernelTime = (double)(endk - startk) * 1e-6f;
  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
    std::terminate();
  }

  std::cout << "SAFELEN: " << SAFELEN << " -- kernel time : " << kernelTime
            << " ms\n";
  std::cout << "Throughput for kernel with SAFELEN " << SAFELEN << ": ";
  std::cout << std::fixed << std::setprecision(0)
            << (((double)matrixSize * sizeof(float) * 1e-3f) /
                (kernelTime * 1e-3f))
            << "KB/s\n";
}

int main(int argc, char **argv) {
  std::array<float, matrixSize> A, B, C;
  // Initialize input with random data
  for (int i = 0; i < matrixSize; i++) {
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
  transpose_and_fold<minSafelen>(selector, A, B);
  transpose_and_fold<maxSafelen>(selector, A, C);

  // Verify result
  for (int i = 0; i < matrixSize; i++) {
    if (B[i] != C[i]) {
      std::cout << "FAILED: The results are incorrect" << std::endl;
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct" << std::endl;
  return 0;
}
