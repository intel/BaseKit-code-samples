//==============================================================
// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// This is a simple DPC++ program that accompanies the Getting Started
// Guide of the debugger.  The kernel does not compute anything
// particularly interesting; it is designed to illustrate the most
// essential features of the debugger when the target device is CPU or
// GPU.

#include <CL/sycl.hpp>
#include <iostream>

// A device function, called from inside the kernel.
static int get_dim(cl::sycl::id<1> wi, int dim) {
    return wi[dim];
}

int main () {
    using namespace cl::sycl;

    constexpr size_t LENGTH = 64;
    int input[LENGTH];
    int output[LENGTH];

    // Initialize the input
    for (unsigned int i = 0; i < LENGTH; i++)
        input[i] = i + 100;

    try {
        queue device_queue;
        std::cout << "[SYCL] Using device: ["
                  << device_queue.get_device().get_info<info::device::name>()
                  << "]" << std::endl;

        range<1> data_range {LENGTH};
        buffer<int, 1> buffer_in {&input[0], data_range};
        buffer<int, 1> buffer_out {&output[0], data_range};

        device_queue.submit ([&] (handler& cgh) {
            auto acc_in = buffer_in.get_access<access::mode::read>(cgh);
            auto acc_out = buffer_out.get_access<access::mode::write>(cgh);

            // kernel-start
            cgh.parallel_for<class kernel>(data_range, [=](id<1> index) {
                int id0 = get_dim(index, 0);
                int element = acc_in[index]; // breakpoint-here
                int result = element + 50;
                if (id0 % 2 == 0) {
                    result = result + 50; // then-branch
                } else {
                    result = -1;          // else-branch
                }
                acc_out[index] = result;
            });
            // kernel-end
        });

        // The scope enforces waiting on the queue, but here we make it
        // explicit.
        device_queue.wait();
    } catch (cl::sycl::exception const& e) {
        std::cout << "fail; synchronous exception occurred: "
                  << e.what() << std::endl;
        return -1;
    }

    // Verify the output
    for (unsigned int i = 0; i < LENGTH; i++) {
        int result = (i % 2 == 0) ? (input[i] + 100) : -1;
        if (output[i] != result) {
            std::cout << "fail; element " << i << " is " << output[i]
                      << std::endl;
            return -1;
        }
    }

    std::cout << "success; result is correct." << std::endl;
    return 0;
}
