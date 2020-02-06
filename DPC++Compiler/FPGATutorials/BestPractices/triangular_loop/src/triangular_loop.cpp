//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>

using namespace cl::sycl;

#define INIT_SEED       (42)                // Seed for randomizing data inputs
#define NUM_RUNS        (2)                 // This tutorial runs twice to show the impact with and without the optimization.
#define NS              (1000000000.0)      // number of nanoseconds in a second
#define SIZE            (8*1024)            // Number of inputs. Don't set this too large, otherwise computation of the reference solution will take a long time on the host (the time is proportional to SIZE^2)
#define M               (30)                // >=1. Minimum number of iterations of the inner loop that must be executed in the optimized implementation. Set this approximately equal to the ii of inner loop in the unoptimized implementation.

// do not use with unary operators, e.g., MIN(x++, y++)
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

class Task;

// This method represents the operation you perform on the loop-carried variable in the triangular loop (i.e. a dot product or something
// that may take many cycles to complete).
int something_complicated(int x) { return (int)cl::sycl::sqrt((float)x); }

// This kernel function implements two data paths: with and without the optimization. 'optimize' specifies which path to take.
void triangular_loop  ( std::unique_ptr<queue> &device_queue,
                  buffer<uint32_t> &input_buf,
                  buffer<uint32_t> &output_buf,
                  uint32_t n,
                  event &queue_event,
                  bool optimize
                ) {
  // Enqueue kernel
  queue_event = device_queue->submit([&](cl::sycl::handler& cgh) {
    
    // Get accessors to the SYCL buffers
    auto _input = input_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto _output = output_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);

    cgh.single_task<Task>([=]() {

      // See README for description of the loop_bound calculation.
      const int loop_bound = (n*(n+1)/2 - 1) + // real iterations
                             (M-2)*(M-1)/2;    // extra dummy iterations      

      // Local memory for the buffer to be operated on
      uint32_t local_buf[SIZE];

      // Read the input_buf from global mem and load it into the local mem
      for (uint32_t i = 0; i < SIZE; i++) {
        local_buf[i] = _input[i];
      }

      // Perform the triangular loop computation

      if (!optimize) { // Unoptimized loop.

        for (int x = 0; x < n; x++) {
          for (int y = x + 1; y < n; y++) {
            local_buf[y] = local_buf[y] + something_complicated(local_buf[x]);
          }
        }

      } else {  // Optimized loop.

        // Indices to track the execution inside the single, merged loop.
        int x = 0, y = 1;

        // Specify that the minimum dependence-distance of loop-carried variables is M iterations. We ensure this is true
        // by modifying the y index such that a minimum of M iterations are always executed.
        [[intelfpga::ivdep(M)]]
        for (int i = 0; i < loop_bound; i++) {
          // Determine if this iteration is a dummy iteration or a real iteration in which the computation should be performed.
          bool compute = y > x;
          // Perform the computation if needed.
          if (compute) {
            local_buf[y] = local_buf[y] + something_complicated(local_buf[x]);
          }
          // Figure out the next value for the indices.
          y++;
          if (y == n) { // If we've hit the end, set y such that a minimum of M iterations are exected.
            x++;
            y = MIN(n - M, x + 1);
          }
        }

      }

      // Write the output to global mem
      for (uint32_t i = 0; i < SIZE; i++) {
        _output[i] = local_buf[i];
      }      

    });
  });

  device_queue->throw_asynchronous();

}

int main() {
  try {
    // Host and kernel profiling
    cl::sycl::event queue_event;
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

    // Create input and output buffers
    auto input_buf = buffer<uint32_t>(range<1>(SIZE));
    auto output_buf = buffer<uint32_t>(range<1>(SIZE));
  
    srand(INIT_SEED);

    // Compute the reference solution
    uint32_t gold[SIZE];

    {
      // Get host-side accessors to the SYCL buffers.
      auto _input_host = input_buf.template get_access<access::mode::write>();      
      // Initialize random input
      for (int i = 0; i < SIZE; ++i) {
        _input_host[i] = rand() % 256;
      }  

      for (int i = 0; i < SIZE; ++i) {
        gold[i] = _input_host[i];
      }
    }  // Host accessor goes out-of-scope and is destructed. This is required in order to unblock the kernel's subsequent accessor to the same buffer.

    for (int x = 0; x < SIZE; x++) {
      for (int y = x + 1; y < SIZE; y++) {
        gold[y] += something_complicated(gold[x]);
      }
    }  

    std::cout << "Length of input array: " << SIZE << std::endl << std::endl;

    for (int i=0;i<NUM_RUNS;i++) {
      
      switch (i)
      {
          case 0: {
                    std::cout << "Beginning run without triangular loop optimization." << std::endl << std::endl;
                    triangular_loop (device_queue, input_buf, output_buf, SIZE, queue_event, false);
                    break;
                  }
          case 1: {
                    std::cout << "Beginning run with triangular loop optimization." << std::endl << std::endl;
                    triangular_loop (device_queue, input_buf, output_buf, SIZE, queue_event, true);
                    break;
                  }
          default: { triangular_loop (device_queue, input_buf, output_buf, SIZE, queue_event, false); }
      }    

      // Wait for kernels to finish
      device_queue->wait();

      t1_kernel = queue_event.template get_profiling_info<info::event_profiling::command_start>();
      t2_kernel = queue_event.template get_profiling_info<info::event_profiling::command_end>();
      time_kernel = (t2_kernel - t1_kernel) / NS;

      // Get accessor to output buffer. Accessing the buffer at this point in the code will block on kernel completion.
      auto _output_host = output_buf.template get_access<access::mode::read>();

      // Verify output and print pass/fail
      bool passed = true;
      int num_errors=0;
      for (int b = 0; b < SIZE; b++) {
        if (num_errors<10 && _output_host[b] != gold[b]) {
          passed = false;
          std::cout << " Mismatch at element " << b << ". expected " << gold[b] << ")" << std::endl;
          num_errors++;
        }
      }

      if (passed) {
        std::cout << "Verification PASSED" << std::endl << std::endl;

        // Report host execution time and throughput
        std::cout.setf(std::ios::fixed);
        std::cout << "Execution time: " << time_kernel << " seconds" << std::endl;
        int num_iterations = SIZE*(SIZE+1)/2 - 1; // One piece of data is processed on each iteration. This formula is taken from the loop_bound calculation.
        double N_MB = (sizeof(uint32_t) * num_iterations)/(1024*1024); // Amount of data processed, in mB
        std::cout << "Throughput "<< (i==0? "without" : "with") << " optimization: " << N_MB/time_kernel << " MB/s" << std::endl << std::endl;
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
