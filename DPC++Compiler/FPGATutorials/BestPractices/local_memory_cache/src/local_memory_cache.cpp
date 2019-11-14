//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>

using namespace cl::sycl;

#define INIT_NUM_INPUTS (16 * 1024 * 1024)  // Default number of inputs.
#define NUM_OUTPUTS     (64)                // Number of outputs
#define INIT_SEED       (42)                // Seed for randomizing data inputs
#define CACHE_DEPTH     (5)                 // Depth of the cache.
#define NUM_RUNS        (2)                 // This tutorial runs twice to show the impact with and without the cache.
#define NS              (1000000000.0)      // number of nanoseconds in a second

class Task;

// This kernel function implements two data paths: with and without caching. use_cache specifies which path to take.
void histogram  ( std::unique_ptr<queue> &device_queue,
                  buffer<uint32_t> &input_buf,
                  buffer<uint32_t> &output_buf,
                  event &queue_event,
                  bool use_cache
                ) {
  // Enqueue  kernel
  queue_event = device_queue->submit([&](handler& cgh) {
    
    // Get accessors to the SYCL buffers
    auto _input = input_buf.get_access<access::mode::read>(cgh);
    auto _output = output_buf.get_access<access::mode::discard_write>(cgh);

    cgh.single_task<Task>([=]() {

      const bool cache = use_cache;
      
      // Local memory for histogram
      uint32_t local_output[NUM_OUTPUTS];
      uint32_t local_output_with_cache[NUM_OUTPUTS];
      // Register-based cache of recently-accessed memory locations
      uint32_t lastSum[CACHE_DEPTH + 1];
      uint32_t lastSumIndex[CACHE_DEPTH + 1];

      // Initialize histogram to zero
      for (uint32_t b = 0; b < NUM_OUTPUTS; ++b) {
        local_output[b] = 0;
        local_output_with_cache[b] = 0;
      }

      // Compute the histogram
      if (!cache) { // Without cache
        for (uint32_t n = 0; n < INIT_NUM_INPUTS; ++n) {
          uint32_t b = _input[n] % NUM_OUTPUTS; // Compute the histogram index to increment
          local_output[b]++;
        }
      } else { // With cache
        [[intelfpga::ivdep(CACHE_DEPTH)]] // Specify that the minimum dependence-distance of loop carried variables is CACHE_DEPTH.
        for (uint32_t n = 0; n < INIT_NUM_INPUTS; ++n) {
          
          uint32_t b = _input[n] % NUM_OUTPUTS; // Compute the histogram index to increment
          
          uint32_t val = local_output_with_cache[b]; // Get the value from the local mem at this index.
          
          // However, if this location in local mem was recently written to, take the value from the cache.
          #pragma unroll
          for (int i = 0; i < CACHE_DEPTH + 1; i++) {
             if (lastSumIndex[i] == b) val = lastSum[i];
          }

          // Write the new value to both the cache and the local mem.
          lastSum[CACHE_DEPTH] = local_output_with_cache[b] = val + 1;
          lastSumIndex[CACHE_DEPTH] = b;
          
          // Cache is just a shift register, so shift the shift reg. Pushing into the back of the shift reg is done above.
          #pragma unroll
          for (int i = 0; i < CACHE_DEPTH; i++) {
             lastSum[i] = lastSum[i + 1];
             lastSumIndex[i] = lastSumIndex[i + 1];
          }           
        }
      }

      // Write output to global memory
      for (uint32_t b = 0; b < NUM_OUTPUTS; ++b) {
        if (!cache) {
          _output[b] = local_output[b];
        } else {
          _output[b] = local_output_with_cache[b];
        }
      }
    });
  });

  device_queue->throw_asynchronous();

}

int main() {
  try {
    // Host and kernel profiling
    event queue_event;
    cl_ulong t1_kernel, t2_kernel;
    double time_kernel;

    // Create queue, get platform and device
    #if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
    std::cout << std::endl << "Emulator output does not demonstrate true hardware performance. The design may need to run on actual hardware to observe the performance benefit of the optimization exemplified in this tutorial." << std::endl << std::endl;
    #elif defined(CPU_HOST)
    host_selector device_selector;
    std::cout << std::endl << "CPU Host target does not accurately measure kernel execution time. The design must run on actual hardware to observe the benefit of the optimization exemplified in this tutorial." << std::endl << std::endl;
    #else
    intel::fpga_selector device_selector;
    #endif
    
    auto property_list = cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    std::unique_ptr<queue> device_queue;

    try {
      //queue device_queue(device_selector, property_list);
      device_queue.reset( new queue(device_selector, async_handler{}, property_list) );
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught a synchronous SYCL exception:" << std::endl << e.what() << std::endl;
      std::cout << "If you are targeting an FPGA hardware, please "
                    "ensure that your system is plugged to an FPGA board that is set up correctly." << std::endl;
      std::cout << "If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR." << std::endl;
      std::cout << "If you are targeting a CPU host device, compile with -DCPU_HOST." << std::endl;
      return 1;
    }

    platform platform = device_queue->get_context().get_platform();
    device device = device_queue->get_device();
    std::cout << "Platform name: " <<  platform.get_info<info::platform::name>().c_str() << std::endl;
    std::cout << "Device name: " <<  device.get_info<info::device::name>().c_str() << std::endl << std::endl << std::endl;

    std::cout << std::endl << "Number of inputs: " << INIT_NUM_INPUTS << std::endl;
    std::cout << "Number of outputs: " << NUM_OUTPUTS << std::endl << std::endl;

    // Create input and output buffers
    auto input_buf = buffer<uint32_t>(range<1>(INIT_NUM_INPUTS));
    auto output_buf = buffer<uint32_t>(range<1>(NUM_OUTPUTS));

    // Get host-side accessors to the SYCL buffers
    auto _input_host = input_buf.template get_access<access::mode::write>();
    // Initialize random input
    srand(INIT_SEED);
    for (int i = 0; i < INIT_NUM_INPUTS; ++i) {
      _input_host[i] = rand();
    }  

    // Compute the reference solution
    uint32_t gold[NUM_OUTPUTS];
    for (int b = 0; b < NUM_OUTPUTS; ++b) {
      gold[b] = 0;
    }
    for (int i = 0; i < INIT_NUM_INPUTS; ++i) {
      int b = _input_host[i] % NUM_OUTPUTS;
      gold[b]++;
    }

    for (int i=0;i<NUM_RUNS;i++) {
      
      switch (i)
      {
          case 0: {
                    std::cout << "Beginning run without local memory caching." << std::endl << std::endl;
                    histogram (device_queue, input_buf, output_buf, queue_event, false);
                    break;
                  }
          case 1: {
                    std::cout << "Beginning run with local memory caching." << std::endl << std::endl;
                    histogram (device_queue, input_buf, output_buf, queue_event, true);
                    break;
                  }
          default: { histogram (device_queue, input_buf, output_buf, queue_event, false); }
      }    

      // Wait for kernels to finish
      device_queue->wait();

      // Compute kernel execution time
      t1_kernel = queue_event.template get_profiling_info<info::event_profiling::command_start>();
      t2_kernel = queue_event.template get_profiling_info<info::event_profiling::command_end>();
      time_kernel = (t2_kernel - t1_kernel) / NS;

      // Get accessor to output buffer. Accessing the buffer at this point in the code will block on kernel completion.
      auto _output_host = output_buf.template get_access<access::mode::read>();

      // Verify output and print pass/fail
      bool passed = true;
      int num_errors=0;
      for (int b = 0; b < NUM_OUTPUTS; b++) {
        if (num_errors<10 && _output_host[b] != gold[b]) {
          passed = false;
          std::cout << " (mismatch, expected " << gold[b] << ")" << std::endl;
          num_errors++;
        }
      }

      if (passed) {
        std::cout << "Verification PASSED" << std::endl << std::endl;

        // Report host execution time and throughput
        std::cout.setf(std::ios::fixed);
        double N_MB = (INIT_NUM_INPUTS * sizeof(uint32_t))/(1024 * 1024); // Input size in MB

        // Report kernel execution time and throughput
        std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;
        std::cout << "Kernel throughput "<< (i==0? "without" : "with") << " caching: " << N_MB/time_kernel << " MB/s" << std::endl << std::endl;
      } else {
        std::cout << "Verification FAILED" << std::endl;
        return 1;
      }
    }
  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught a SYCL exception:" << std::endl << e.what() << std::endl;
    return 1;
  }

  return 0;
}
