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
using namespace std;

constexpr access::mode kSyclRead = access::mode::read;
constexpr access::mode kSyclDiscardWrite = access::mode::discard_write;

#define VECTOR_SIZE (64)  // VECTOR_SIZE of vectors a and r

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (exception_ptr const& e : exceptions) {
    try {
      rethrow_exception(e);
    } catch (cl::sycl::exception const& e) {
      cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
      terminate();
    }
  }
};

class SimpleMath;

int GetGoldenResult(int input) {
  constexpr int kRead[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
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
    acc += (coeff[i] * (mul + kRead[i]));
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
    string option(argv[1]);
    if (option == "-h" || option == "--help") {
      cout << "Usage: \n";
      cout << "<executable> <data size>\n";
      cout << "\n";
      return 1;
    } else {
      data_size = stoi(option);
    }
  }
  vector<int> vec_a(data_size);
  vector<int> vec_r(data_size);
  for (int i = 0; i < data_size; i++) {
    vec_a[i] = rand() % 128;
  }

  try {
    // Device buffers
    buffer<int, 1> device_a(vec_a.data(), data_size);
    buffer<int, 1> device_r(vec_r.data(), data_size);

    auto list_properties = property_list{property::queue::enable_profiling()};

#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    unique_ptr<queue> q;

    try {
      q.reset(new queue(device_selector, exception_handler, list_properties));
    } catch (cl::sycl::exception const& e) {
      cout << "Caught a synchronous SYCL exception:"
           << "\n"
           << e.what() << "\n";
      cout << "If you are targeting an FPGA hardware, please "
              "ensure that your system is plugged to an FPGA board that "
              "is set up correctly."
           << "\n";
      cout << "If you are targeting the FPGA emulator, compile with "
              "-DFPGA_EMULATOR."
           << "\n";
      return 1;
    }

    event e = q->submit([&](handler& h) {
      auto a = device_a.get_access<kSyclRead>(h);
      auto r = device_r.get_access<kSyclDiscardWrite>(h);

      // Kernel
      // Task version
      h.single_task<class SimpleMath>([=]() [[intel::kernel_args_restrict]] {
        constexpr int kRead[] = {
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
            acc = intel::fpga_reg(acc) + (coeff[i] * (mul + kRead[i]));
#else
            acc += (coeff[i] * (mul + kRead[i]));
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

    q->wait_and_throw();
    cl_ulong start_k =
        e.get_profiling_info<info::event_profiling::command_start>();
    cl_ulong end_k = e.get_profiling_info<info::event_profiling::command_end>();

    // Convert the units of start_k and end_k to milliseconds from nanoseconds.
    double kernel_time = (double)(end_k - start_k) * 1e-6f;

    // Kernel consists of two nested loops with 3 operations in the innermost
    // loop: 2 additions and 1 multiplication operation.
    const int kNumOpsPerKernel = data_size * VECTOR_SIZE * 3;
    cout << "Throughput for kernel with data size " << data_size
         << " and VECTOR_SIZE " << VECTOR_SIZE << ": ";
    cout << std::fixed << std::setprecision(6)
         << ((double)kNumOpsPerKernel / kernel_time) / 1e6f << " GFlops\n";
  } catch (cl::sycl::exception const& e) {
    cout << "caught a sycl exception:"
         << "\n"
         << e.what() << "\n";
  }

  // Test the results.
  bool correct = true;

  for (int i = 0; i < data_size; i++) {
    if (GetGoldenResult(vec_a[i]) != vec_r[i]) {
      cout << "Found mismatch at " << i << ", " << GetGoldenResult(vec_a[i])
           << " != " << vec_r[i] << "\n";
      correct = false;
    }
  }

  if (correct) {
    cout << "PASSED: Results are correct.\n";
  } else {
    cout << "FAILED: Results are incorrect.\n";
    return 1;
  }

  return 0;
}
