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
constexpr access::mode kSyclRead = access::mode::read;
constexpr access::mode kSyclWrite = access::mode::write;

using ProducerToConsumerPipe = intel::pipe<  // Defined in the SYCL headers.
    class ProducerConsumerPipe,              // An identifier for the pipe.
    int,                                     // The type of data in the pipe.
    4>;                                      // The capacity of the pipe.

class ProducerTutorial;

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
      std::terminate();
    }
  }
};

void Producer(queue &device_queue, buffer<int, 1> &input_buffer) {
  std::cout << "Enqueuing producer...\n";

  event queue_event;
  queue_event = device_queue.submit([&](handler &cgh) {
    auto input_accessor = input_buffer.get_access<kSyclRead>(cgh);
    auto num_elements = input_buffer.get_count();

    cgh.single_task<ProducerTutorial>([=]() {
      for (int i = 0; i < num_elements; ++i) {
        ProducerToConsumerPipe::write(input_accessor[i]);
      }
    });
  });
}

int ConsumerWork(int i) { return i * i; }

class ConsumerTutorial;
void Consumer(queue &device_queue, buffer<int, 1> &output_buffer) {
  std::cout << "Enqueuing consumer...\n";

  event queue_event;
  queue_event = device_queue.submit([&](handler &cgh) {
    auto output_accessor = output_buffer.get_access<kSyclWrite>(cgh);
    auto num_elements = output_buffer.get_count();

    cgh.single_task<ConsumerTutorial>([=]() {
      for (int i = 0; i < num_elements; ++i) {
        auto input = ProducerToConsumerPipe::read();
        auto answer = ConsumerWork(input);
        output_accessor[i] = answer;
      }
    });
  });
}

int main(int argc, char *argv[]) {
  int array_size = (1 << 10);

  if (argc > 1) {
    std::string option(argv[1]);
    if (option == "-h" || option == "--help") {
      std::cout << "Usage: \n"
                << "<executable> <data size>\n"
                << "\n";
      return 0;
    } else {
      array_size = std::stoi(option);
    }
  }

  std::cout << "Input Array Size:  " << array_size << "\n";

  std::vector<int> producer_input(array_size, -1);
  std::vector<int> consumer_output(array_size, -1);

  for (int i = 0; i < array_size; i++) producer_input[i] = i;

  try {
    buffer<int, 1> producer_buffer(producer_input.data(), array_size);
    buffer<int, 1> consumer_buffer(consumer_output.data(), array_size);

#ifdef FPGA_EMULATOR
    intel::fpga_emulator_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    queue device_queue(device_selector, exception_handler);

    Producer(device_queue, producer_buffer);
    Consumer(device_queue, consumer_buffer);
    device_queue.wait_and_throw();
  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::cout << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly\n";
    std::cout << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR\n";
    std::cout << "   This design is not supported on CPU targets.\n";
    return 1;
  }

  // Verify result
  for (unsigned int i = 0; i < array_size; i++) {
    if (consumer_output[i] != ConsumerWork(producer_input[i])) {
      std::cout << "input = " << producer_input[i]
                << " expected: " << ConsumerWork(producer_input[i])
                << " got: " << consumer_output[i] << "\n";
      std::cout << "FAILED: The results are incorrect\n";
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct\n";
  return 0;
}
