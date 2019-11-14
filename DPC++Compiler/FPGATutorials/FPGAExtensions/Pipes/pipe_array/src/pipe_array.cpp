//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "pipe_array.h"
#include "unroller.h"
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace cl::sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

constexpr int NUM_ROWS = 2;
constexpr int NUM_COLS = 2;
constexpr int NUMBER_OF_CONSUMERS = NUM_ROWS * NUM_COLS;
constexpr int DEPTH = 2;

using ProducerToConsumerPipeMatrix = PipeArray< // Defined in "pipe_array.h".
    class ProducerConsumerPipe,                 // An identifier for the pipe.
    uint64_t,                                   // The type of data in the pipe.
    DEPTH,                                      // The capacity of each pipe.
    NUM_ROWS,                                   // array dimension.
    NUM_COLS                                    // array dimension.
    >;

class ProducerTutorial;
void producer(queue &deviceQueue, buffer<uint64_t, 1> &input_buffer) {
  std::cout << "Enqueuing producer...\n";

  event queue_event;
  queue_event = deviceQueue.submit([&](handler &cgh) {
    auto input_accessor = input_buffer.get_access<sycl_read>(cgh);
    auto num_elements = input_buffer.get_count();
    auto num_passes = num_elements / NUMBER_OF_CONSUMERS;

    cgh.single_task<ProducerTutorial>([=]() {
      int input_idx = 0;
      for (int pass = 0; pass < num_passes; pass++) {
        unroller<0, NUM_ROWS>::step([&input_idx, input_accessor](auto i_idx) {
          constexpr int i = i_idx.value;

          unroller<0, NUM_COLS>::step([&input_idx, input_accessor](auto j_idx) {
            constexpr int j = j_idx.value;

            ProducerToConsumerPipeMatrix::pipe_at<i, j>::write(
                input_accessor[input_idx++]);
          });
        });
      }
    });
  });
}

uint64_t consumer_work(uint64_t i) { return i * i; }

template <int ConsumerID> class ConsumerTutorial;
template <int ConsumerID>
void consumer(queue &deviceQueue, buffer<uint64_t, 1> &output_buffer) {
  std::cout << "Enqueuing consumer " << ConsumerID << "...\n";

  event queue_event;
  queue_event = deviceQueue.submit([&](handler &cgh) {
    auto output_accessor = output_buffer.get_access<sycl_write>(cgh);
    auto num_elements = output_buffer.get_count();

    cgh.single_task<ConsumerTutorial<ConsumerID>>([=]() {
      constexpr int consumer_x = ConsumerID / NUM_COLS;
      constexpr int consumer_y = ConsumerID % NUM_COLS;
      for (int i = 0; i < num_elements; ++i) {
        auto input = ProducerToConsumerPipeMatrix::pipe_at<consumer_x,
                                                           consumer_y>::read();
        auto answer = consumer_work(input);
        output_accessor[i] = answer;
      }
    });
  });
}

int main(int argc, char *argv[]) {

  uint64_t array_size = 1;
  array_size <<= 10;

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

  if (array_size % NUMBER_OF_CONSUMERS != 0) {
    std::cout << "Array size must be a multiple of the number of consumers! "
                 "Exiting...\n";
    return 0;
  }

  uint64_t items_per_consumer = array_size / NUMBER_OF_CONSUMERS;
  std::vector<uint64_t> producer_input(array_size, -1);
  std::array<std::vector<uint64_t>, NUMBER_OF_CONSUMERS> consumer_output;

  for (auto &output : consumer_output)
    output.resize(items_per_consumer, -1);

  for (int i = 0; i < array_size; i++)
    producer_input[i] = i;

  try {
#ifdef FPGA_EMULATOR
    intel::fpga_emulator_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif
    queue deviceQueue(device_selector);

    buffer<uint64_t, 1> producer_buffer(producer_input.data(), array_size);
    producer(deviceQueue, producer_buffer);

    std::vector<buffer<uint64_t, 1>> consumer_buffers;
    unroller<0, NUMBER_OF_CONSUMERS>::step([&](auto idx) {
      constexpr int consumer_id = idx.value;
      consumer_buffers.emplace_back(consumer_output[consumer_id].data(),
                                    items_per_consumer);
      consumer<consumer_id>(deviceQueue, consumer_buffers.back());
    });

    deviceQueue.wait_and_throw();
  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cout << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cout << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;
    std::cout
        << "   This design is not supported on CPU targets."
        << std::endl;
    return 1;
  }

  // Verify result
  for (int i = 0; i < items_per_consumer; ++i) {
    for (int consumer = 0; consumer < NUMBER_OF_CONSUMERS; ++consumer) {
      auto fpga_result = consumer_output[consumer][i];
      auto expected_result = consumer_work(NUMBER_OF_CONSUMERS * i + consumer);
      if (fpga_result != expected_result) {
        std::cout << "FAILED: The results are incorrect" << std::endl;
        std::cout << "On Input: " << NUMBER_OF_CONSUMERS * i + consumer
                  << " Expected: " << expected_result << " Got: " << fpga_result
                  << "\n";
        return 1;
      }
    }
  }
  std::cout << "PASSED: The results are correct" << std::endl;
  return 0;
}
