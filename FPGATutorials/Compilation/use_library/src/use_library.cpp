//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include "lib.hpp"

using namespace cl::sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
class kernelCompute;

auto exception_handler = [](exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (exception const& e) {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << "\n";
      std::terminate();
    }
  }
};

int main() {
  cl_uint result = 0;
  const float a = 2.0f;
  const float b = 3.0f;

  // all device related code that can throw exception
  try {
    buffer<cl_float, 1> buffer_a(&a, 1);
    buffer<cl_float, 1> buffer_b(&b, 1);
    buffer<cl_uint, 1> buffer_c(&result, 1);

#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
    host_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    std::unique_ptr<queue> q;

    // Catch device seletor runtime error
    try {
      q.reset(new queue(device_selector, exception_handler));
    } catch (exception const& e) {
      std::cout << "Caught a synchronous SYCL exception: \n"
                << e.what() << "\n";
      std::cout << "If you are targeting an FPGA hardware, please "
                   "ensure that your system is plugged to an FPGA board that "
                   "is set up correctly.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
      std::cout
          << "If you are targeting a CPU host device, compile with -DCPU_HOST.\n";
      return -1;
    }

    q->submit([&](handler& h) {
      auto const accessor_a = buffer_a.get_access<sycl_read>(h);
      auto const accessor_b = buffer_b.get_access<sycl_read>(h);
      auto accessor_c = buffer_c.get_access<sycl_write>(h);
      h.single_task<class kernelCompute>([=]() {
        float a_sq = OclSquare(accessor_a[0]);
        float a_sq_sqrt = HlsSqrtf(a_sq);
        float b_sq = SyclSquare(accessor_b[0]);
        accessor_c[0] = RtlByteswap((cl_uint)(a_sq_sqrt + b_sq));
      });
    });

    q->throw_asynchronous();
  } catch (exception const& e) {
    std::cout << "Caught synchronous SYCL exception: \n"
              << e.what() << "\n";
    std::terminate();
  }

  cl_uint gold = sqrt(a * a) + (b * b);
  gold = gold << 16 | gold >> 16;

  if (result != gold) {
    std::cout << "FAILED: result is incorrect!\n";
    return -1;
  }
  std::cout << "PASSED: result is correct!\n";
  return 0;
}
