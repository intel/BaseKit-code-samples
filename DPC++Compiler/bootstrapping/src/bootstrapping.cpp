//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace cl::sycl;
constexpr size_t N { 15 };

// ############################################################
// bootstrap_function returns a char buffer thru an input buffer

void bootstrap_function(char *result) {
    char bootstrap[N] = {'B','o','o','t','s','t','r','a','p','p','i','n','g','!','\0'};
    for( size_t i = 0; i < N; ++i ) { result[i] = bootstrap[i]; }
 }

// ############################################################
// entry point for the program

int main(int argc, char** argv) {
        // default_selector my_selector;
        queue my_queue;
        std::cout << "Device : " << my_queue.get_device().get_info<info::device::name>() << std::endl;

        char result[14];
        buffer<char, 1> my_buffer(result, range<1>(N));
        my_queue.submit([&] (handler &my_handler){
                auto my_accessor = my_buffer.get_access<access::mode::read_write>(my_handler);
                my_handler.single_task<class bootstrap>([=](){
                        bootstrap_function(&my_accessor[0]);
                });
        });
        my_queue.wait_and_throw();
        my_buffer.get_access<access::mode::read>();

        for(auto i = 0; i < N; ++i) { std::cout << result[i]; }
        std::cout << std::endl;

        return 0;
}
