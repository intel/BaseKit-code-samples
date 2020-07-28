//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>

using namespace cl::sycl;

constexpr access::mode kSyclRead = access::mode::read;
constexpr access::mode kSyclWrite = access::mode::write;

constexpr unsigned kRows = 8;
constexpr unsigned kVec = 4;
constexpr unsigned kMaxVal = 512;
constexpr unsigned kNumTests = 64;
constexpr int kMaxIter = 8;

// forward declare the class name used in lamda for defining the kernel
class KernelCompute;

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
      std::terminate();
    }
  }
};

// The shared compute function for host and device code
unsigned Compute(unsigned init, unsigned int dict_offset[][kVec]) {
  using UintArray = cl_uint[kVec];
  using Uint2DArray = cl_uint[kVec][kVec];

  // We do not provide any attributes for compare_offset and hash;
  // we let the compiler decide what's best based on the access pattern
  // and their size.
  Uint2DArray compare_offset;
  UintArray hash;

#pragma unroll
  for (unsigned char i = 0; i < kVec; i++) {
    hash[i] = (++init) & (kRows - 1);
  }

  int count = 0, iter = 0;
  do {
    // After unrolling both loops, we have kVec*kVec reads from dict_offset
#pragma unroll
    for (int i = 0; i < kVec; i++) {
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        compare_offset[k][i] = dict_offset[hash[i]][k];
      }
    }

    // After unrolling, we have kVec writes to dict_offset
#pragma unroll
    for (unsigned char k = 0; k < kVec; ++k) {
      dict_offset[hash[k]][k] = (init << k);
    }
    init++;

#pragma unroll
    for (int i = 0; i < kVec; i++) {
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        count += compare_offset[i][k];
      }
    }
  } while (++iter < kMaxIter);
  return count;
}

unsigned RunKernel(unsigned init, unsigned int const dict_offset_init[]) {
  cl_uint result = 0;
  // Include all the SYCL work in a {} block to ensure all
  // SYCL tasks are completed before exiting the block.
  {
#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    queue device_queue(device_selector, exception_handler);

    // set up the buffers.
    buffer<cl_uint, 1> buffer_dict_offset_init(dict_offset_init,
                                               range<1>(kRows * kVec));
    buffer<cl_uint, 1> buffer_R(&result, range<1>(1));

    event queue_event;
    queue_event = device_queue.submit([&](handler& cgh) {
      // create accessors from global memory buffers to read input data and
      // to write the results.
      auto accessor_D = buffer_dict_offset_init.get_access<kSyclRead>(cgh);
      auto accessor_R = buffer_R.get_access<kSyclWrite>(cgh);

      cgh.single_task<KernelCompute>([=]() [[intel::kernel_args_restrict]] {
#if defined(SINGLEPUMP)
        [[intelfpga::singlepump, intelfpga::memory("MLAB"),
          intelfpga::numbanks(kVec), intelfpga::max_replicates(kVec)]]
#elif defined(DOUBLEPUMP)
        [[intelfpga::doublepump, intelfpga::memory("MLAB"),
          intelfpga::numbanks(kVec), intelfpga::max_replicates(kVec)]]
#endif
        unsigned int dict_offset[kRows][kVec];

        // Initialize 'dict_offset' with values from global memory.
        for (int i = 0; i < kRows; ++i) {
#pragma unroll
          for (unsigned char k = 0; k < kVec; ++k) {
            // After unrolling, we end up with kVec writes to dict_offset.
            dict_offset[i][k] = accessor_D[i * kVec + k];
          }
        }
        accessor_R[0] = Compute(init, dict_offset);
      });
    });
  }
  return result;
}

// This host side function performs the same computation as the device side
// kernel, and is used to verify functional correctness.
unsigned GoldenRun(unsigned init, unsigned int const dict_offset_init[]) {
  unsigned int dict_offset[kRows][kVec];
  for (int i = 0; i < kRows; ++i) {
    for (unsigned char k = 0; k < kVec; ++k) {
      dict_offset[i][k] = dict_offset_init[i * kVec + k];
    }
  }
  return Compute(init, dict_offset);
}

int main() {
  srand(0);

#if defined(SINGLEPUMP)
  printf("Testing Kernel with Single-pumped memories\n");
#elif defined(DOUBLEPUMP)
  printf("Testing kernel with Double-pumped memories\n");
#else
  printf("Testing kernel with no attributes applied to memories\n");
#endif

  bool passed = true;
  for (unsigned j = 0; j < kNumTests; j++) {
    // initialize input data with random values
    const unsigned init = rand() % kMaxVal;
    unsigned int dict_offset_init[kRows * kVec];
    for (int i = 0; i < kRows; ++i) {
      for (char k = 0; k < kVec; ++k) {
        dict_offset_init[i * kVec + k] = rand() % kMaxVal;
      }
    }

    try {
      // run the device side kernel, the result in retunrned in R, the
      // time taken for the kernel to execute is return in KernelRunTimeNs.
      auto kernel_result = RunKernel(init, dict_offset_init);

      // compute the golden result
      auto golden_result = GoldenRun(init, dict_offset_init);

      // kernel run is functionally correct only if its result matches the
      // golden result.
      passed = (kernel_result == golden_result);

      if (!passed) {
        printf(
            "  Test#%u: mismatch: %d != %d (kernel_result != golden_result)\n",
            j, kernel_result, golden_result);
      }
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
      std::cout << "   If you are targeting an FPGA hardware, "
                   "ensure that your system is plugged to an FPGA board that "
                   "is set up correctly\n";
      std::cout << "   If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR\n";
      return 1;
    }
  }

  if (passed) {
    printf("PASSED: all kernel results are correct.\n");
  } else {
    printf("FAILED\n");
    return 1;
  }

  return 0;
}
