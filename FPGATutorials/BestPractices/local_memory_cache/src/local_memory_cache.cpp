//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>

using namespace cl::sycl;

constexpr int kInitNumInputs = 16 * 1024 * 1024;  // Default number of inputs.
constexpr int kNumOutputs = 64;                   // Number of outputs
constexpr int kInitSeed = 42;         // Seed for randomizing data inputs
constexpr int kCacheDepth = 5;        // Depth of the cache.
constexpr int kNumRuns = 2;           // runs twice to show the impact of cache
constexpr double kNs = 1000000000.0;  // number of nanoseconds in a second

class Task;

// This kernel function implements two data paths: with and without caching.
// use_cache specifies which path to take.
void Histogram(std::unique_ptr<queue>& q, buffer<uint32_t>& input_buf,
               buffer<uint32_t>& output_buf, event& e, bool use_cache) {
  // Enqueue  kernel
  e = q->submit([&](handler& h) {
    // Get accessors to the SYCL buffers
    auto _input = input_buf.get_access<access::mode::read>(h);
    auto _output = output_buf.get_access<access::mode::discard_write>(h);

    h.single_task<Task>([=]() [[intel::kernel_args_restrict]] {
      const bool cache = use_cache;

      // Local memory for Histogram
      uint32_t local_output[kNumOutputs];
      uint32_t local_output_with_cache[kNumOutputs];

      // Register-based cache of recently-accessed memory locations
      uint32_t last_sum[kCacheDepth + 1];
      uint32_t last_sum_index[kCacheDepth + 1];

      // Initialize Histogram to zero
      for (uint32_t b = 0; b < kNumOutputs; ++b) {
        local_output[b] = 0;
        local_output_with_cache[b] = 0;
      }

      // Compute the Histogram
      if (!cache) {  // Without cache
        for (uint32_t n = 0; n < kInitNumInputs; ++n) {
          // Compute the Histogram index to increment
          uint32_t b = _input[n] % kNumOutputs;
          local_output[b]++;
        }
      } else {  // With cache

        // Specify that the minimum dependence-distance of
        // loop carried variables is kCacheDepth.
        [[intelfpga::ivdep(kCacheDepth)]] for (uint32_t n = 0;
                                               n < kInitNumInputs; ++n) {
          // Compute the Histogram index to increment
          uint32_t b = _input[n] % kNumOutputs;

          // Get the value from the local mem at this index.
          uint32_t val = local_output_with_cache[b];

          // However, if this location in local mem was recently
          // written to, take the value from the cache.
#pragma unroll
          for (int i = 0; i < kCacheDepth + 1; i++) {
            if (last_sum_index[i] == b) val = last_sum[i];
          }

          // Write the new value to both the cache and the local mem.
          last_sum[kCacheDepth] = local_output_with_cache[b] = val + 1;
          last_sum_index[kCacheDepth] = b;

// Cache is just a shift register, so shift the shift reg. Pushing into the back
// of the shift reg is done above.
#pragma unroll
          for (int i = 0; i < kCacheDepth; i++) {
            last_sum[i] = last_sum[i + 1];
            last_sum_index[i] = last_sum_index[i + 1];
          }
        }
      }

      // Write output to global memory
      for (uint32_t b = 0; b < kNumOutputs; ++b) {
        if (!cache) {
          _output[b] = local_output[b];
        } else {
          _output[b] = local_output_with_cache[b];
        }
      }
    });
  });

  q->throw_asynchronous();
}

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

int main() {
  try {
    // Host and kernel profiling
    event e;
    cl_ulong t1_kernel, t2_kernel;
    double time_kernel;

// Create queue, get platform and device
#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
    std::cout << "\nEmulator output does not demonstrate true hardware "
                 "performance. The design may need to run on actual hardware "
                 "to observe the performance benefit of the optimization "
                 "exemplified in this tutorial.\n\n";
#else
    intel::fpga_selector device_selector;
#endif

    auto property_list =
        cl::sycl::property_list{property::queue::enable_profiling()};
    std::unique_ptr<queue> q;

    try {
      // queue q(device_selector, property_list);
      q.reset(new queue(device_selector, exception_handler, property_list));
    } catch (exception const& e) {
      std::cout << "Caught a synchronous SYCL exception:\n" << e.what() << "\n";
      std::cout << "If you are targeting an FPGA hardware, please "
                   "ensure that your system is plugged to an FPGA board that "
                   "is set up correctly.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
      return 1;
    }

    platform platform = q->get_context().get_platform();
    device device = q->get_device();
    std::cout << "Platform name: "
              << platform.get_info<info::platform::name>().c_str() << "\n";
    std::cout << "Device name: "
              << device.get_info<info::device::name>().c_str() << "\n\n\n";

    std::cout << "\nNumber of inputs: " << kInitNumInputs << "\n";
    std::cout << "Number of outputs: " << kNumOutputs << "\n\n";

    // Create input and output buffers
    auto input_buf = buffer<uint32_t>(range<1>(kInitNumInputs));
    auto output_buf = buffer<uint32_t>(range<1>(kNumOutputs));

    srand(kInitSeed);

    // Compute the reference solution
    uint32_t gold[kNumOutputs];

    {
      // Get host-side accessors to the SYCL buffers
      auto _input_host = input_buf.get_access<access::mode::write>();
      // Initialize random input
      for (int i = 0; i < kInitNumInputs; ++i) {
        _input_host[i] = rand();
      }

      for (int b = 0; b < kNumOutputs; ++b) {
        gold[b] = 0;
      }
      for (int i = 0; i < kInitNumInputs; ++i) {
        int b = _input_host[i] % kNumOutputs;
        gold[b]++;
      }
    }

    // Host accessor is now out-of-scope and is destructed. This is required
    // in order to unblock the kernel's subsequent accessor to the same buffer.

    for (int i = 0; i < kNumRuns; i++) {
      switch (i) {
        case 0: {
          std::cout << "Beginning run without local memory caching.\n\n";
          Histogram(q, input_buf, output_buf, e, false);
          break;
        }
        case 1: {
          std::cout << "Beginning run with local memory caching.\n\n";
          Histogram(q, input_buf, output_buf, e, true);
          break;
        }
        default: {
          Histogram(q, input_buf, output_buf, e, false);
        }
      }

      // Wait for kernels to finish
      q->wait();

      // Compute kernel execution time
      t1_kernel = e.get_profiling_info<info::event_profiling::command_start>();
      t2_kernel = e.get_profiling_info<info::event_profiling::command_end>();
      time_kernel = (t2_kernel - t1_kernel) / kNs;

      // Get accessor to output buffer. Accessing the buffer at this point in
      // the code will block on kernel completion.
      auto _output_host = output_buf.get_access<access::mode::read>();

      // Verify output and print pass/fail
      bool passed = true;
      int num_errors = 0;
      for (int b = 0; b < kNumOutputs; b++) {
        if (num_errors < 10 && _output_host[b] != gold[b]) {
          passed = false;
          std::cout << " (mismatch, expected " << gold[b] << ")\n";
          num_errors++;
        }
      }

      if (passed) {
        std::cout << "Verification PASSED\n\n";

        // Report host execution time and throughput
        std::cout.setf(std::ios::fixed);
        double N_MB = (kInitNumInputs * sizeof(uint32_t)) /
                      (1024 * 1024);  // Input size in MB

        // Report kernel execution time and throughput
        std::cout << "Kernel execution time: " << time_kernel << " seconds\n";
        std::cout << "Kernel throughput " << (i == 0 ? "without" : "with")
                  << " caching: " << N_MB / time_kernel << " MB/s\n\n";
      } else {
        std::cout << "Verification FAILED\n";
        return 1;
      }
    }
  } catch (exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n" << e.what() << "\n";
    return 1;
  }

  return 0;
}
