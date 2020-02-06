//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <array>
#include <iostream>

#include <CL/sycl.hpp>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include "tbb/task_group.h"
// exception handler
/*
The exception_list parameter is an iterable list of std::exception_ptr objects.
But those pointers are not always directly readable.
So, we rethrow the pointer, catch it,  and then we have the exception itself.
Note: depending upon the operation there may be several exceptions.
*/
auto exception_handler = [](cl::sycl::exception_list exceptionList) {
  for (std::exception_ptr const& e : exceptionList) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const& e) {
      std::terminate();  // exit the process immediately.
    }
  }
};

#define VERBOSE

float alpha = 0.5;  // coeff for triad calculation

const size_t ARRAY_SIZE = 16;
std::array<float, ARRAY_SIZE> A_array;      // input
std::array<float, ARRAY_SIZE> B_array;      // input
std::array<float, ARRAY_SIZE> C_array;      // output
std::array<float, ARRAY_SIZE> C_array_tbb;  // output

class execute_on_gpu {
  const char* message;

 public:
  execute_on_gpu(const char* str) : message(str) {}
  void operator()() const {
    std::cout << message << std::endl;

    // By including all the SYCL work in a {} block, we ensure
    // all SYCL tasks must complete before exiting the block
    {  // starting SYCL code

      const float coeff = alpha;  // coeff is a local varaible
      cl::sycl::range<1> n_items{ARRAY_SIZE};
      cl::sycl::buffer<cl::sycl::cl_float, 1> A_array_buffer(A_array.data(),
                                                             n_items);
      cl::sycl::buffer<cl::sycl::cl_float, 1> B_array_buffer(B_array.data(),
                                                             n_items);
      cl::sycl::buffer<cl::sycl::cl_float, 1> C_array_buffer(C_array.data(),
                                                             n_items);

      cl::sycl::queue device_queue;
      device_queue
          .submit([&](cl::sycl::handler& cgh) {
            auto A_accessor =
                A_array_buffer.get_access<cl::sycl::access::mode::read>(cgh);
            auto B_accessor =
                B_array_buffer.get_access<cl::sycl::access::mode::read>(cgh);
            auto C_accessor =
                C_array_buffer.get_access<cl::sycl::access::mode::write>(cgh);

            cgh.parallel_for<class Triad>(n_items, [=](cl::sycl::id<1> index) {
              C_accessor[index] = A_accessor[index] + B_accessor[index] * coeff;
            });  // end of the kernel -- parallel for
          })
          .wait_and_throw();  // end of the commands for the SYCL queue

    }  // end of the scope for SYCL code; wait unti queued work completes
  }    // operator
};

class execute_on_cpu {
  const char* message;

 public:
  execute_on_cpu(const char* str) : message(str) {}
  void operator()() const {
    std::cout << message << std::endl;

    tbb::parallel_for(tbb::blocked_range<int>(0, A_array.size()),
                      [&](tbb::blocked_range<int> r) {
                        for (int index = r.begin(); index < r.end(); ++index) {
                          C_array_tbb[index] =
                              A_array[index] + B_array[index] * alpha;
                        }
                      });
  }  // operator()
};

void print_array(const char* text, const std::array<float, ARRAY_SIZE>& array) {
  std::cout << text;
  for (const auto& s : array) std::cout << s << ' ';
  std::cout << '\n';
}

int main() {
  // init input arrays
  for (int i = 0; i < ARRAY_SIZE; i++) {
    A_array[i] = i;
    B_array[i] = i;
  }

  // start tbb task group
  tbb::task_group tg;
  //  tbb::task_scheduler_init init(2);
  int nth = 4;  // number of threads
  auto mp = tbb::global_control::max_allowed_parallelism;
  tbb::global_control gc(mp, nth);

  tg.run(execute_on_gpu("executing on GPU"));  // spawn task and return
  tg.run(execute_on_cpu("executing on CPU"));  // spawn task and return

  tg.wait();  // wait for tasks to complete

  // Serial execution
  std::array<float, ARRAY_SIZE> CGold;

  for (size_t i = 0; i < ARRAY_SIZE; ++i)
    CGold[i] = A_array[i] + alpha * B_array[i];

  // Compare golden triad with heterogeneous triad
  if (!std::equal(std::begin(C_array), std::end(C_array), std::begin(CGold)))
    std::cout << "Heterogenous triad error.\n";
  else
    std::cout << "Heterogenous triad correct.\n";

  // Compare golden triad with TBB triad
  if (!std::equal(std::begin(C_array_tbb), std::end(C_array_tbb),
                  std::begin(CGold)))
    std::cout << "TBB triad error.\n";
  else
    std::cout << "TBB triad correct.\n";

#ifdef VERBOSE
  print_array("input array A_array: ", A_array);
  print_array("input array B_array: ", B_array);
  print_array("output array C_array on GPU: ", C_array);
  print_array("output array C_array_tbb on CPU: ", C_array_tbb);
#endif

}  // main
