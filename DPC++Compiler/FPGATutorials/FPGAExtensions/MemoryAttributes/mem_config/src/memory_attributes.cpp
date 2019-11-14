//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>

using namespace cl::sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

constexpr unsigned ROWS = 8;
constexpr unsigned VEC = 4;
constexpr unsigned MAXVAL = 512;
constexpr unsigned NUMTESTS = 64;
constexpr int MAXITER = 8;

// forward declare the class name used in lamda for defining the kernel
class KernelCompute;

// The shared compute function for host and device code
unsigned compute(unsigned Init, unsigned int DictOffset[][VEC]) {
  using UintArray = cl_uint[VEC];
  using Uint2DArray = cl_uint[VEC][VEC];

  // We do not provide any attributes for CompareOffset and Hash;
  // we let the compiler decide what's best based on the access pattern
  // and their size.
  Uint2DArray CompareOffset;
  UintArray Hash;

#pragma unroll
  for (unsigned char i = 0; i < VEC; i++) {
    Hash[i] = (++Init) & (ROWS - 1);
  }

  int Count = 0, Iter = 0;
  do {
    // After unrolling both loops, we have VEC*VEC reads from DictOffset
#pragma unroll
    for (int i = 0; i < VEC; i++) {
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        CompareOffset[k][i] = DictOffset[Hash[i]][k];
      }
    }

    // After unrolling, we have VEC writes to DictOffset
#pragma unroll
    for (unsigned char k = 0; k < VEC; ++k) {
      DictOffset[Hash[k]][k] = (Init << k);
    }
    Init++;

#pragma unroll
    for (int i = 0; i < VEC; i++) {
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        Count += CompareOffset[i][k];
      }
    }
  } while (++Iter < MAXITER);
  return Count;
}

unsigned runKernel(unsigned Init, unsigned int const DictOffsetInit[]) {
  cl_uint result = 0;
  // Include all the SYCL work in a {} block to ensure all
  // SYCL tasks are completed before exiting the block.
  {

#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
    host_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    queue DeviceQueue(device_selector);

    // set up the buffers.
    buffer<cl_uint, 1> bufferDictOffsetInit(DictOffsetInit,
                                            range<1>(ROWS * VEC));
    buffer<cl_uint, 1> bufferR(&result, range<1>(1));

    event QueueEvent;
    QueueEvent = DeviceQueue.submit([&](handler &cgh) {
      // create accessors from global memory buffers to read input data and
      // to write the results.
      auto accessorD = bufferDictOffsetInit.get_access<sycl_read>(cgh);
      auto accessorR = bufferR.get_access<sycl_write>(cgh);

      cgh.single_task<class KernelCompute>([=]() {
#if defined(SINGLEPUMP)
        [[intelfpga::singlepump, intelfpga::memory("MLAB"),
          intelfpga::numbanks(VEC), intelfpga::max_replicates(VEC)]]
#elif defined(DOUBLEPUMP)
        [[intelfpga::doublepump, intelfpga::memory("MLAB"),
          intelfpga::numbanks(VEC), intelfpga::max_replicates(VEC)]]
#endif
        unsigned int DictOffset[ROWS][VEC];

        // Initialize 'DictOffset' with values from global memory.
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
          for (unsigned char k = 0; k < VEC; ++k) {
            // After unrolling, we end up with VEC writes to DictOffset.
            DictOffset[i][k] = accessorD[i * VEC + k];
          }
        }
        accessorR[0] = compute(Init, DictOffset);
      });
    });
  }
  return result;
}

// This host side function performs the same computation as the device side
// kernel, and is used to verify functional correctness.
unsigned goldenRun(unsigned Init, unsigned int const DictOffsetInit[]) {
  unsigned int DictOffset[ROWS][VEC];
  for (int i = 0; i < ROWS; ++i) {
    for (unsigned char k = 0; k < VEC; ++k) {
      DictOffset[i][k] = DictOffsetInit[i * VEC + k];
    }
  }
  return compute(Init, DictOffset);
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

  bool Passed = true;
  for (unsigned j = 0; j < NUMTESTS; j++) {
    // initialize input data with random values
    const unsigned Init = rand() % MAXVAL;
    unsigned int DictOffsetInit[ROWS * VEC];
    for (int i = 0; i < ROWS; ++i) {
      for (char k = 0; k < VEC; ++k) {
        DictOffsetInit[i * VEC + k] = rand() % MAXVAL;
      }
    }

    try {
      // run the device side kernel, the result in retunrned in R, the
      // time taken for the kernel to execute is return in KernelRunTimeNs.
      auto KernelResult = runKernel(Init, DictOffsetInit);

      // compute the golden result
      auto GoldenResult = goldenRun(Init, DictOffsetInit);

      // kernel run is functionally correct only if its result matches the
      // golden result.
      Passed = (KernelResult == GoldenResult);

      if (!Passed) {
        printf("  Test#%u: mismatch: %d != %d (KernelResult != GoldenResult)\n",
               j, KernelResult, GoldenResult);
      }
    } catch (cl::sycl::exception const &e) {
      std::cout << "Caught a synchronous SYCL exception: " << e.what()
                << std::endl;
      std::cout << "   If you are targeting an FPGA hardware, "
                   "ensure that your system is plugged to an FPGA board that "
                   "is set up correctly"
                << std::endl;
      std::cout << "   If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR"
                << std::endl;
      std::cout << "   If you are targeting a CPU host device, compile with "
                   "-DCPU_HOST"
                << std::endl;
      return 1;
    }
  }

  if (Passed) {
    printf("PASSED: all kernel results are correct.\n");
  } else {
    printf("FAILED\n");
    return 1;
  }

  return 0;
}
