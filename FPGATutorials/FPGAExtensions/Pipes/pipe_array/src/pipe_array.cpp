//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "pipe_array.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#include "unroller.hpp"
using namespace cl::sycl;
constexpr access::mode kSyclRead = access::mode::read;
constexpr access::mode kSyclWrite = access::mode::write;

constexpr int kNumRows = 2;
constexpr int kNumCols = 2;
constexpr int kNumberOfConsumers = kNumRows * kNumCols;
constexpr int kDepth = 2;

using ProducerToConsumerPipeMatrix = PipeArray<  // Defined in "pipe_array.h".
    class ProducerConsumerPipe,                  // An identifier for the pipe.
    uint64_t,  // The type of data in the pipe.
    kDepth,    // The capacity of each pipe.
    kNumRows,  // array dimension.
    kNumCols   // array dimension.
    >;

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

void Producer(queue &device_queue, buffer<uint64_t, 1> &input_buffer) {
  std::cout << "Enqueuing producer...\n";

  event queue_event;
  queue_event = device_queue.submit([&](handler &cgh) {
    auto input_accessor = input_buffer.get_access<kSyclRead>(cgh);
    auto num_elements = input_buffer.get_count();
    auto num_passes = num_elements / kNumberOfConsumers;

    cgh.single_task<ProducerTutorial>([=]() {
      int input_idx = 0;
      for (int pass = 0; pass < num_passes; pass++) {
        Unroller<0, kNumRows>::step([&input_idx, input_accessor](auto i_idx) {
          constexpr int i = i_idx.value;

          Unroller<0, kNumCols>::step([&input_idx, input_accessor](auto j_idx) {
            constexpr int j = j_idx.value;

            ProducerToConsumerPipeMatrix::PipeAt<i, j>::write(
                input_accessor[input_idx++]);
          });
        });
      }
    });
  });
}

uint64_t ConsumerWork(uint64_t i) { return i * i; }

template <int ConsumerID>
class ConsumerTutorial;
template <int ConsumerID>
void Consumer(queue &device_queue, buffer<uint64_t, 1> &output_buffer) {
  std::cout << "Enqueuing consumer " << ConsumerID << "...\n";

  event queue_event;
  queue_event = device_queue.submit([&](handler &cgh) {
    auto output_accessor = output_buffer.get_access<kSyclWrite>(cgh);
    auto num_elements = output_buffer.get_count();

    cgh.single_task<ConsumerTutorial<ConsumerID>>([=]() {
      constexpr int consumer_x = ConsumerID / kNumCols;
      constexpr int consumer_y = ConsumerID % kNumCols;
      for (int i = 0; i < num_elements; ++i) {
        auto input = ProducerToConsumerPipeMatrix::PipeAt<consumer_x,
                                                          consumer_y>::read();
        auto answer = ConsumerWork(input);
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

  if (array_size % kNumberOfConsumers != 0) {
    std::cout << "Array size must be a multiple of the number of consumers! "
                 "Exiting...\n";
    return 0;
  }

  uint64_t items_per_consumer = array_size / kNumberOfConsumers;
  std::vector<uint64_t> producer_input(array_size, -1);
  std::array<std::vector<uint64_t>, kNumberOfConsumers> consumer_output;

  for (auto &output : consumer_output) output.resize(items_per_consumer, -1);

  for (int i = 0; i < array_size; i++) producer_input[i] = i;

  try {
#ifdef FPGA_EMULATOR
    intel::fpga_emulator_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif
    queue device_queue(device_selector, exception_handler);

    buffer<uint64_t, 1> producer_buffer(producer_input.data(), array_size);
    Producer(device_queue, producer_buffer);

    std::vector<buffer<uint64_t, 1>> consumer_buffers;
    Unroller<0, kNumberOfConsumers>::step([&](auto idx) {
      constexpr int consumer_id = idx.value;
      consumer_buffers.emplace_back(consumer_output[consumer_id].data(),
                                    items_per_consumer);
      Consumer<consumer_id>(device_queue, consumer_buffers.back());
    });

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
  for (int i = 0; i < items_per_consumer; ++i) {
    for (int consumer = 0; consumer < kNumberOfConsumers; ++consumer) {
      auto fpga_result = consumer_output[consumer][i];
      auto expected_result = ConsumerWork(kNumberOfConsumers * i + consumer);
      if (fpga_result != expected_result) {
        std::cout << "FAILED: The results are incorrect\n";
        std::cout << "On Input: " << kNumberOfConsumers * i + consumer
                  << " Expected: " << expected_result << " Got: " << fpga_result
                  << "\n";
        return 1;
      }
    }
  }
  std::cout << "PASSED: The results are correct\n";
  return 0;
}
