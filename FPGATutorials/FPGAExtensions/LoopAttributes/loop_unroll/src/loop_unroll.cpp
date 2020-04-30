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

auto exception_handler = [](exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (exception const& e) {
      std::cout << "Caught asynchronous SYCL exception:\n";
      std::terminate();
    }
  }
};

// This function instantiates the VecAdd kernel, which contains a loop that adds
// up the two summand arrays, and stores the result into sum This loop will be
// unrolled by the specified UnrollFactor macro
template <int UnrollFactor>
void VecAdd(const std::vector<float>& summands1,
            const std::vector<float>& summands2, std::vector<float>& sum,
            int array_size) {
  auto prop_list = property_list{property::queue::enable_profiling()};
  event e;
  try {
// Initialize queue with device selector and enabling profiling
#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
    host_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    std::unique_ptr<queue> q;

    try {
      q.reset(new queue(device_selector, exception_handler, prop_list));
    } catch (exception const& e) {
      std::cout << "Caught a synchronous SYCL exception:\n" << e.what() << "\n";
      std::cout << "If you are targeting an FPGA hardware, please "
                   "ensure that your system is plugged to an FPGA board that "
                   "is set up correctly.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
      std::cout << "If you are targeting a CPU host device, compile with "
                   "-DCPU_HOST.\n";
      return;
    }

    buffer<float, 1> buffer_summands1(summands1.data(), array_size);
    buffer<float, 1> buffer_summands2(summands2.data(), array_size);
    buffer<float, 1> buffer_sum(sum.data(), array_size);

    e = q->submit([&](handler& h) {
      auto acc_summands1 = buffer_summands1.get_access<sycl_read>(h);
      auto acc_summands2 = buffer_summands2.get_access<sycl_read>(h);
      auto acc_sum = buffer_sum.get_access<sycl_write>(h);
      auto size = array_size;
      h.single_task<SimpleVadd<UnrollFactor> >([=
      ]() [[intel::kernel_args_restrict]] {
#pragma unroll UnrollFactor
        for (int k = 0; k < size; k++) {
          acc_sum[k] = acc_summands1[k] + acc_summands2[k];
        }
      });
    });

    q->wait_and_throw();
    cl_ulong startk =
        e.get_profiling_info<info::event_profiling::command_start>();
    cl_ulong endk = e.get_profiling_info<info::event_profiling::command_end>();
    // unit of startk and endk is nano second, convert to ms
    double kernel_time = (double)(endk - startk) * 1e-6f;

    std::cout << "UnrollFactor " << UnrollFactor
              << " kernel time : " << kernel_time << " ms\n";
    std::cout << "Throughput for kernel with UnrollFactor " << UnrollFactor
              << ": ";
    std::cout << std::fixed << std::setprecision(3)
              << ((double)array_size / kernel_time) / 1e6f << " GFlops\n";
  } catch (exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n";
    std::terminate();
  }
}

int main(int argc, char* argv[]) {
  int array_size = 67108864;

  if (argc > 1) {
    std::string option(argv[1]);
    if (option == "-h" || option == "--help") {
      std::cout << "Usage: \n";
      std::cout << "<executable> <data size>\n\n";
      return 0;
    } else {
      array_size = std::stoi(option);
    }
  }

  std::vector<float> summands1(array_size);
  std::vector<float> summands2(array_size);

  std::vector<float> sum_unrollx1(array_size);
  std::vector<float> sum_unrollx2(array_size);
  std::vector<float> sum_unrollx4(array_size);
  std::vector<float> sum_unrollx8(array_size);
  std::vector<float> sum_unrollx16(array_size);

  // Initialize the two summand arrays (arrays to be added to each other) to
  // 1:N and N:1, so that the sum of all elements is N + 1
  for (int i = 0; i < array_size; i++) {
    summands1[i] = static_cast<float>(i + 1);
    summands2[i] = static_cast<float>(array_size - i);
  }

  std::cout << "Input Array Size:  " << array_size << "\n";

  // Instantiate VecAdd kernel with different unroll factors: 1, 2, 4, 8, 16
  // The VecAdd kernel contains a loop that adds up the two summand arrays
  // This loop will be unrolled by the specified unroll factor
  // The sum array is expected to be identical, regardless of the unroll factor
  VecAdd<1>(summands1, summands2, sum_unrollx1, array_size);
  VecAdd<2>(summands1, summands2, sum_unrollx2, array_size);
  VecAdd<4>(summands1, summands2, sum_unrollx4, array_size);
  VecAdd<8>(summands1, summands2, sum_unrollx8, array_size);
  VecAdd<16>(summands1, summands2, sum_unrollx16, array_size);

  // Verify that each sum array is identical to each other, for every unroll
  // factor
  for (unsigned int i = 0; i < array_size; i++) {
    if (sum_unrollx1[i] != summands1[i] + summands2[i] ||
        sum_unrollx1[i] != sum_unrollx2[i] ||
        sum_unrollx1[i] != sum_unrollx4[i] ||
        sum_unrollx1[i] != sum_unrollx8[i] ||
        sum_unrollx1[i] != sum_unrollx16[i]) {
      std::cout << "FAILED: The results are incorrect\n";
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct\n";
  return 0;
}
