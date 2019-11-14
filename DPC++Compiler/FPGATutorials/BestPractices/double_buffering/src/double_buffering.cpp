//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <random>
#include <iomanip>

using namespace std::chrono;
using namespace cl::sycl;

#if defined(FPGA_EMULATOR) || defined(CPU_HOST)
  #define TIMES     (20)        // # times to execute the kernel. TIMES must be >= 2
  #define SIZE      (4096)     
#else 
  #define TIMES     (100)      // # times to execute the kernel. TIMES must be >= 2
  #define SIZE      (2621440)  // # of floats to process on each kernel execution. ~10MB
#endif

#define POW       (20)        // Kernel executes a power function (base^POW). Must be >= 2. Can increase this to increase kernel execution time, but process_output() time will also increase.
#define NUM_RUNS  (2)         // Number of iterations through the main loop

bool pass = true;

class SimpleVpow; 

/*  Kernel function.
    Performs bufferB[i] = bufferA[i] ** pow
    Only supports pow >= 2.
    This kernel is not meant to be an optimal implementation of the power operation -- it's just a sample kernel for this tutorial whose execution time is easily controlled via the pow parameter.
    SYCL buffers are created externally and passed in by reference to control (external to this function) when the buffers are destructed. 
    The destructor causes a blocking buffer transfer from device to host and double buffering requires us to not block here (because we need to another kernel).
    So we only want this transfer to occur at the end of overall execution, not at the end of each individual kernel execution.
*/
void simple_pow ( std::unique_ptr<queue> &deviceQueue,
                  buffer<cl_float, 1> &bufferA,
                  buffer<cl_float, 1> &bufferB,
                  event &queue_event
                ) {

  // Submit to the queue and execute the kernel
  queue_event = deviceQueue->submit([&](handler& cgh) {
    // Get kernel access to the buffers
    auto accessorA = bufferA.template get_access<access::mode::read>(cgh);
    auto accessorB = bufferB.template get_access<access::mode::discard_read_write>(cgh);

    const int num = SIZE;
    assert (POW >= 2);
    const int p = POW-1; // Assumes pow >= 2;

    /*
      Explicitly instruct the SYCL runtime to copy the kernel's output buffer back to the host upon kernel completion.
      This is not required for functionality since the buffer access in process_output() also implicitly instructs the runtime to copy the data back. But it should be noted that this buffer access 
      blocks process_output() until the kernel is complete and the data is copied.
      In contrast, update_host() instructs the runtime to perform the copy earlier. This allows process_output() to optionally perform more useful work *before* making the blocking buffer access.
      Said another way, this allows process_output() to potentially perform more work in parallel with the runtime's copy operation.
    */
    cgh.update_host(accessorB); 

    cgh.single_task<class SimpleVpow>([=]() {
      for (int j = 0; j < p; j++) {
        if (j==0) {
          for (int i = 0; i < num; i++) {
             accessorB[i] = accessorA[i] * accessorA[i];
          }
        } else {
          for (int i = 0; i < num; i++) {
             accessorB[i] = accessorB[i] * accessorA[i];
          }          
        }
      }
    });
  });

  deviceQueue->throw_asynchronous();

}

// Returns kernel execution time for a given SYCL event from a queue.
cl_ulong sycl_get_exec_time_ns(event queue_event){

  cl_ulong start_time = queue_event.template get_profiling_info<info::event_profiling::command_start>();
  cl_ulong end_time = queue_event.template get_profiling_info<info::event_profiling::command_end>();
  return (end_time-start_time);

}

// Local pow function for verifying results
cl_float my_pow (cl_float input, int pow) {

  return (pow == 0) ? 1 : input * my_pow(input, pow - 1);

}

/*  Compares kernel output against expected output. Only compares part of the output so that this method completes quickly. This is done intentionally/artificially keep host-processing time shorter than kernel execution time.
    Grabs kernel output data from its SYCL buffer. Reading from this buffer is a blocking operation that will block on the kernel completing.
    Queries and records execution time of the kernel that just completed. This is a natural place to do this because process_output() is blocked on kernel completion.
*/
void process_output (buffer<cl_float, 1> &input_buf, buffer<cl_float, 1> &output_buf, int exec_number, event queue_event, cl_ulong &total_kernel_time_per_slot){
  auto input_buf_acc = input_buf.template get_access<access::mode::read>();
  auto output_buf_acc = output_buf.template get_access<access::mode::read>();
  int num_errors = 0;
  int num_errors_to_print = 10;
  /*  The use of update_host() in the kernel function allows for additional host-side operations to be performed here, in parallel with the buffer copy operation from device to host, 
      before the blocking access to the output buffer is made via output_buf_acc[]. To be clear, no real operations are done here and this is just a note that this is the place
      where you *could* do it. */
  for (int i=0; i<SIZE/8; i++) {
    if ((num_errors<num_errors_to_print) && (my_pow(input_buf_acc[i], POW) != output_buf_acc[i])) {
      if (num_errors==0) {
        pass = false;
        std::cout << "Verification failed on kernel execution # " << exec_number << ". Showing up to " << num_errors_to_print << " mismatches." << std::endl;
      }
      std::cout << "Verification failed on kernel execution # " << exec_number << ", at element " << i << ". Expected " << std::fixed << std::setprecision(16) << my_pow(input_buf_acc[i],POW) << " but got " << output_buf_acc[i] << std::endl;
      num_errors++;
    }
  }
  
  // At this point we know the kernel has completed, so can query the profiling data.
  total_kernel_time_per_slot += sycl_get_exec_time_ns(queue_event);  

}

/*
    Generates input data for the next kernel execution. Only fills part of the buffer so that this method completes quickly. This is done intentionally/artificially keep host-processing time shorter than kernel execution time.
    Writes the data into the associated SYCL buffer. The write will block until the previous kernel execution, that is using this buffer, completes.
*/
void process_input (buffer<cl_float, 1> &buf){
  auto buf_acc = buf.template get_access<access::mode::discard_write>(); // We are generating completely new input data, so can use discard_write() here to indicate we don't care about the SYCL buffer's current contents.
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();//seed
  std::default_random_engine dre(seed);//engine
  std::uniform_real_distribution<float> di(1.0f, 2.0f); //Values between 1 and 2

  float start_val = di(dre); // Randomly generate a start value and increment from there. Compared to randomly generating every value, this is done to speed up this function a bit.

  for (int i=0;i<SIZE/8;i++) {
    buf_acc[i] = start_val;
    start_val++;
  } 

}

int main() {

  try {
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

    std::cout << "Executing kernel " <<  TIMES << " times in each round." << std::endl << std::endl;

    // Create a vector to store the input/output SYCL buffers
    std::vector<buffer<cl_float, 1>> input_buf;
    std::vector<buffer<cl_float, 1>> output_buf;

    event sycl_events[2];             // SYCL events for each kernel launch.
    cl_ulong total_kernel_time_per_slot[2];     // In nanoseconds. Total execution time of kernels in a given slot.
    cl_ulong total_kernel_time = 0;             // Total execution time of all kernels.
   

    // Allocate vectors to store the host-side copies of the input data
    // Create and allocate the SYCL buffers
    for (int i = 0; i < 2; i++) {
      input_buf.push_back (buffer<cl_float, 1>(range<1>(SIZE)));
      output_buf.push_back (buffer<cl_float, 1>(range<1>(SIZE)));    
    }

    /*
      Main loop. This loop runs twice to show the performance difference without and with double buffering.
    */
    for (int i=0;i<NUM_RUNS;i++) {

      for (int i = 0; i < 2; i++) {
        total_kernel_time_per_slot[i] = 0; // Initialize timers to zero.
      }

      switch (i)
      {
          case 0: { 
                    std::cout << "*** Beginning execution, without double buffering" << std::endl;
                    break;
                  }
          case 1: { 
                    std::cout << "*** Beginning execution, with double buffering." << std::endl;
                    break;
                  }  
          default:  {
                      std::cout << "*** Beginning execution." << std::endl;
                    }
      }

      high_resolution_clock::time_point t1 = high_resolution_clock::now(); // Start the timer. This will include the time to process the input data for the first 2 kernel executions.

      if (i==0) { // Single buffering
        for (int i=0; i < TIMES; i++) {
          if (i%10 == 0) { std::cout << "Launching kernel #" << i << std::endl; } // Only print every few iterations, just to limit the prints.
          process_input(input_buf[0]);
          simple_pow(device_queue, input_buf[0], output_buf[0], sycl_events[0]);
          process_output (input_buf[0],output_buf[0], i, sycl_events[0], total_kernel_time_per_slot[0]);
        }
      } else { // Double buffering
        // Process input for first 2 kernel launches and queue them. Then block on processing the output of the first kernel.
        process_input(input_buf[0]);
        process_input(input_buf[1]);
        simple_pow(device_queue, input_buf[0], output_buf[0], sycl_events[0]);
        for (int i=1; i < TIMES; i++) {
          if (i%10 == 0) { std::cout << "Launching kernel #" << i << std::endl; } // Only print every few iterations, just to limit the prints.
          simple_pow(device_queue, input_buf[i%2], output_buf[i%2], sycl_events[i%2]); // Launch the next kernel
          // Process output from previous kernel. This will block on kernel completion.
          process_output (input_buf[(i-1)%2], output_buf[(i-1)%2], i, sycl_events[(i-1)%2], total_kernel_time_per_slot[(i-1)%2]);
          // Generate input for the next kernel.
          process_input (input_buf[(i-1)%2]);
        }
        // Process output of the final kernel
        process_output (input_buf[(TIMES-1)%2], output_buf[(TIMES-1)%2], i, sycl_events[(TIMES-1)%2], total_kernel_time_per_slot[(TIMES-1)%2]);        
      }
     
      // Add up the overall kernel execution time.
      total_kernel_time = 0;
      for (int i=0; i<2;i++) {
        total_kernel_time += total_kernel_time_per_slot[i];
      }

      high_resolution_clock::time_point t2 = high_resolution_clock::now();  // Stop the timer.

      duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

      std::cout << std::endl << "Overall execution time " << ((i==0)? "without" : "with") << " double buffering = " << (unsigned) (time_span.count() * 1000) << " ms" << std::endl;
      std::cout << "Total kernel-only execution time " << ((i==0)? "without" : "with") << " double buffering = " << (unsigned) (total_kernel_time / 1000000) << " ms" << std::endl;
      std::cout << "Throughput = " << std::setprecision(8) << (float)SIZE * (float)TIMES * (float)sizeof(cl_float) / (float)time_span.count() / 1000000  << " MB/s" << std::endl << std::endl << std::endl;
    }
    if (pass) {
      std::cout << "Verification PASSED" << std::endl;
    } else {
      std::cout << "Verification FAILED" << std::endl;
      return 1;
    }
  } catch (cl::sycl::exception const& e) {
    std::cout << "Caught a SYCL exception:" << std::endl << e.what() << std::endl;
    return 1;
  }  

  return 0;
}

