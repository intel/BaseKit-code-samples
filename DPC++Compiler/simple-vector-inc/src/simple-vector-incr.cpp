//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// ### Step 1 - Inspect
// The code presents one input buffer (vector1) for which Sycl buffer memory is allocated.  
// The associated with vector1_accessor set to read/write gets the contents of the buffer.

#include <CL/sycl.hpp>
using namespace cl::sycl;
static const size_t N = 2;
int main() {
        default_selector my_selector;
        queue my_queue(my_selector);
        std::cout << "Device : " << my_queue.get_device().get_info<info::device::name>() << std::endl;
        int vector1[N] = {10,10};
        std::cout << "Input  : " << vector1[0] << ", " << vector1[1] << std::endl;

        // ### Step 2 - Add another input vector - vector2
        // Uncomment the following line to add input vector2
        // int vector2[N] = {20,20};

        // ### Step 3 - Print out for vector2
        // Uncomment the following line
        // std::cout << "Input  : " << vector2[0] << ", " << vector2[1] << std::endl;
        buffer<int, 1> vector1_buffer(vector1, range<1>(N));

        // ### Step 4 - Add another Sycl buffer - vector2_buffer
        // Uncomment the following line
        // buffer<int, 1> vector2_buffer(vector2, range<1>(N));
        my_queue.submit([&] (handler &my_handler){
                auto vector1_accessor = vector1_buffer.get_access<access::mode::read_write>(my_handler);

        // Step 5 - add an accessor for vector2_buffer
        // Look in the source code for the comment
        // auto vector2_accessor = vector2_buffer.template get_access < access::mode::read > (my_handler);

                my_handler.parallel_for<class test>(range<1>(N), [=](id<1> index) {
                        // ### Step 6 - Replace the existing vector1_accessor to accumulate vector2_accessor   
                        // Comment the line: vector1_accessor[index] += 1;
                        vector1_accessor[index] += 1;

                        // Uncomment the following line
                        // vector1_accessor[index] += vector2_accessor[index];
                });
        });
        my_queue.wait_and_throw();
        vector1_buffer.get_access<access::mode::read>();
        std::cout << "Output : " << vector1[0] << ", " << vector1[1] << std::endl;
}
