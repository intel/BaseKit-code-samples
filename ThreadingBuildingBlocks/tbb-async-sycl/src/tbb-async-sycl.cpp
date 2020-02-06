//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <math.h>  //for ceil
#include <array>
#include <iostream>

#include <CL/sycl.hpp>

//#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/flow_graph.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
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

float ratio = 0.5;  // CPU to GPU offload ratio
float alpha = 0.5;  // coeff for triad calculation

const size_t ARRAY_SIZE = 16;
std::array<float, ARRAY_SIZE> A_array;  // input
std::array<float, ARRAY_SIZE> B_array;  // input
std::array<float, ARRAY_SIZE> C_array;  // output

void print_array(const char* text, const std::array<float, ARRAY_SIZE>& array) {
  std::cout << text;
  for (const auto& s : array) std::cout << s << ' ';
  std::cout << '\n';
}

using async_node_type = tbb::flow::async_node<float, double>;
using gateway_type = async_node_type::gateway_type;

class AsyncActivity {
  tbb::task_arena a;

 public:
  AsyncActivity() { a = tbb::task_arena{1, 0}; }
  void run(float offload_ratio, gateway_type& gateway) {
    gateway.reserve_wait();
    a.enqueue([&, offload_ratio]() {
      // Execute the kernel over a portion of the array range
      size_t array_size_sycl = ceil(ARRAY_SIZE * offload_ratio);
      std::cout << "start index for GPU = 0; end index for GPU = "
                << array_size_sycl << std::endl;
      const float coeff = alpha;  // coeff is a local varaible

      // By including all the SYCL work in a {} block, we ensure
      // all SYCL tasks must complete before exiting the block
      {  // starting SYCL code
        cl::sycl::range<1> n_items{array_size_sycl};
        cl::sycl::buffer<cl::sycl::cl_float, 1> A_array_buffer(A_array.data(),
                                                               n_items);
        cl::sycl::buffer<cl::sycl::cl_float, 1> B_array_buffer(B_array.data(),
                                                               n_items);
        cl::sycl::buffer<cl::sycl::cl_float, 1> C_array_buffer(C_array.data(),
                                                               n_items);

        cl::sycl::queue device_queue;
        device_queue
            .submit([&](cl::sycl::handler& cgh) {
              constexpr auto sycl_read = cl::sycl::access::mode::read;
              constexpr auto sycl_write = cl::sycl::access::mode::write;

              auto A_accessor = A_array_buffer.get_access<sycl_read>(cgh);
              auto B_accessor = B_array_buffer.get_access<sycl_read>(cgh);
              auto C_accessor = C_array_buffer.get_access<sycl_write>(cgh);

              cgh.parallel_for<class Triad>(
                  n_items, [=](cl::sycl::id<1> index) {
                    C_accessor[index] =
                        A_accessor[index] + B_accessor[index] * coeff;
                  });  // end of the kernel -- parallel for
            })
            .wait_and_throw();  // end of the commands for the SYCL queue
      }  // end of the scope for SYCL code; wait unti queued work completes

      double sycl_result = 1.0;  // passing some numerical result/flag
      gateway.try_put(sycl_result);
      gateway.release_wait();
    });  // a.enqueue
  }      // run
};

int main() {
  // init input arrays
  for (int i = 0; i < ARRAY_SIZE; i++) {
    A_array[i] = i;
    B_array[i] = i;
  }

  int nth = 4;  // number of threads
                // tbb::task_scheduler_init init { nth };

  auto mp = tbb::global_control::max_allowed_parallelism;
  tbb::global_control gc(mp, nth + 1);  // One more thread, but sleeping
  tbb::flow::graph g;

  // Source node:
  bool n = false;
  tbb::flow::source_node<float> in_node{g,
                                        [&](float& offload_ratio) {
                                          if (n) return false;
                                          offload_ratio = ratio;
                                          n = true;
                                          return true;
                                        },
                                        false};

  // CPU node
  tbb::flow::function_node<float, double> cpu_node{
      g, tbb::flow::unlimited, [&](float offload_ratio) -> double {
        size_t i_start = static_cast<size_t>(ceil(ARRAY_SIZE * offload_ratio));
        size_t i_end = static_cast<size_t>(ARRAY_SIZE);
        std::cout << "start index for CPU = " << i_start
                  << "; end index for CPU = " << i_end << std::endl;

        tbb::parallel_for(tbb::blocked_range<size_t>{i_start, i_end},
                          [&](const tbb::blocked_range<size_t>& r) {
                            for (size_t i = r.begin(); i < r.end(); ++i)
                              C_array[i] = A_array[i] + alpha * B_array[i];
                          });
        double tbb_result = 1.0;  // passing some numerical result/flag
        return (tbb_result);
      }};

  // async node  -- GPU
  AsyncActivity asyncAct;
  async_node_type a_node{
      g, tbb::flow::unlimited,
      [&asyncAct](const float& offload_ratio, gateway_type& gateway) {
        asyncAct.run(offload_ratio, gateway);
      }};

  // join node
  using join_t =
      tbb::flow::join_node<std::tuple<double, double>, tbb::flow::queueing>;
  join_t node_join{g};

  // out node
  tbb::flow::function_node<join_t::output_type> out_node{
      g, tbb::flow::unlimited, [&](const join_t::output_type& times) {
        // Serial execution
        std::array<float, ARRAY_SIZE> CGold;
        for (size_t i = 0; i < ARRAY_SIZE; ++i)
          CGold[i] = A_array[i] + alpha * B_array[i];

        // Compare golden triad with heterogeneous triad
        if (!std::equal(std::begin(C_array), std::end(C_array),
                        std::begin(CGold)))
          std::cout << "Heterogenous triad error.\n";
        else
          std::cout << "Heterogenous triad correct.\n";

#ifdef VERBOSE
        print_array("C_array: ", C_array);
        print_array("CGold  : ", CGold);
#endif
      }};  // end of out node

  // construct graph
  tbb::flow::make_edge(in_node, a_node);
  tbb::flow::make_edge(in_node, cpu_node);
  tbb::flow::make_edge(a_node, tbb::flow::input_port<0>(node_join));
  tbb::flow::make_edge(cpu_node, tbb::flow::input_port<1>(node_join));
  tbb::flow::make_edge(node_join, out_node);

  in_node.activate();
  g.wait_for_all();

  return 0;
}
