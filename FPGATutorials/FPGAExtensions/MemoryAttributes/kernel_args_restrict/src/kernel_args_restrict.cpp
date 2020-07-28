//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <vector>

using namespace cl::sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

// problem input size
constexpr int INSIZE = 1000000;

// kernel names, global scope to avoid excessive name mangling
class KernelArgsRestrict;
class KernelArgsNoRestrict;

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

void runKernels(int size, std::vector<int>& in,
                std::vector<int>& norestrict_out,
                std::vector<int>& restrict_out) {
  // device selector
#if defined(FPGA_EMULATOR)
  intel::fpga_emulator_selector device_selector;
#else
  intel::fpga_selector device_selector;
#endif

  // queue properties to enable kernel profiling
  auto property_list =
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};

  // create the SYCL device queue
  cl::sycl::queue device_queue(device_selector, exception_handler,
                               property_list);

  // set up the input/output buffers
  buffer<int, 1> in_buf(in.data(), size);
  buffer<int, 1> norestrict_out_buf(norestrict_out.data(), size);
  buffer<int, 1> restrict_out_buf(restrict_out.data(), size);

  // submit the task that DOES NOT apply the kernel_args_restrict attribute
  event event_norestrict = device_queue.submit([&](handler& cgh) {
    // create accessors from global memory
    auto in_accessor = in_buf.template get_access<sycl_read>(cgh);
    auto out_accessor = norestrict_out_buf.template get_access<sycl_write>(cgh);

    // run the task
    cgh.single_task<KernelArgsNoRestrict>([=]() {
      for (unsigned i = 0; i < size; i++) {
        out_accessor[i] = in_accessor[i];
      }
    });
  });

  // submit the task that DOES apply the kernel_args_restrict attribute
  event event_restrict = device_queue.submit([&](handler& cgh) {
    // create accessors from global memory
    auto in_accessor = in_buf.template get_access<sycl_read>(cgh);
    auto out_accessor = restrict_out_buf.template get_access<sycl_write>(cgh);

    // run the task (note the use of the attribute here)
    cgh.single_task<KernelArgsRestrict>([=]() [[intel::kernel_args_restrict]] {
      for (unsigned i = 0; i < size; i++) {
        out_accessor[i] = in_accessor[i];
      }
    });
  });

  // wait for kernels to finish
  event_norestrict.wait();
  event_restrict.wait();

  // gather profiling info
  // auto norestrict_submit_time =
  // event_norestrict.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto norestrict_start_time =
      event_norestrict
          .get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto norestrict_end_time =
      event_norestrict
          .get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  auto norestrict_execution_time =
      (norestrict_end_time - norestrict_start_time) / 1000000000.0f;  // ns to s

  // auto restrict_submit_time =
  // event_restrict.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
  auto restrict_start_time =
      event_restrict
          .get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  auto restrict_end_time =
      event_restrict
          .get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  auto restrict_execution_time =
      (restrict_end_time - restrict_start_time) / 1000000000.0f;  // ns to s

  const double input_megabytes = (INSIZE * sizeof(int)) / (1024 * 1024);

  // print some output
  std::cout << "Kernel throughput without attribute: "
            << (input_megabytes / norestrict_execution_time) << " MB/s"
            << std::endl;
  std::cout << "Kernel throughput with attribute: "
            << (input_megabytes / restrict_execution_time) << " MB/s"
            << std::endl;
}

//// main driver program
int main() {
  // seed the random number generator
  srand(0);

  // input/output data
  std::vector<int> in(INSIZE);
  std::vector<int> norestrict_out(INSIZE), restrict_out(INSIZE);

  try {
    // generate some random input data
    for (int i = 0; i < INSIZE; i++) {
      in[i] = rand() % 7777;
    }

    // Run the kernels
    runKernels(INSIZE, in, norestrict_out, restrict_out);

    // validate the restrict outputs
    for (int i = 0; i < INSIZE; i++) {
      if (in[i] != restrict_out[i]) {
        std::cout << "FAILED: mismatch at entry " << i
                  << " of 'KernelArgsNoRestrict' kernel output" << std::endl;
        return 1;
      }
    }
    for (int i = 0; i < INSIZE; i++) {
      if (in[i] != restrict_out[i]) {
        std::cout << "FAILED: mismatch at entry " << i
                  << " of 'KernelArgsRestrict' kernel output" << std::endl;
        return 1;
      }
    }

  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cout << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that "
                 "is set up correctly"
              << std::endl;
    std::cout << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;
    return 1;
  }

  printf("PASSED\n");

  return 0;
}
