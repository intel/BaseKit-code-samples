//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <array>
#include <iostream>

using namespace cl::sycl;

// This is the class used to name the kernel for the runtime.
// This must be done when the kernel is expressed as a lambda.
class ArrayAdd;

// Convience access definitions
constexpr access::mode dp_read = access::mode::read;
constexpr access::mode dp_write = access::mode::write;

// Problem size for this example
constexpr size_t array_size = 10000;

// Define the ARRAY type for use in this example
typedef std::array<cl::sycl::cl_int, array_size> IntArray;

//
// Initialize the @param a of size array_size with consecutive
// elements from 0 to array_size-1
//
void initialize_array(IntArray &a) {
	for (size_t i = 0; i < a.size(); i++) {
		a[i] = i;  // Initializing to the index
	}
}

//
// Computes the sum of two arrays in scalar with a simple loop.
//
void add_arrays_scalar(IntArray &sum, const IntArray &addend_1,
	const IntArray &addend_2) {
	for (size_t i = 0; i < sum.size(); i++) {
		sum[i] = addend_1[i] + addend_2[i];
	}
}

//
// Select the device and allocate the queue
//
std::unique_ptr<queue> initialize_device_queue() {
	// DPC++ model: A host is connected to OpenCL devices
	// Selectors are used to choose devices to be used
	// FPGA emulator and FPGA hardware devices can be targetted explicity
	// The default selector will choose the most performant device
	// Ex: It will use an accelerator if it can find one

	// including an async exception handler
	auto ehandler = [](cl::sycl::exception_list exceptionList) {
		for (std::exception_ptr const &e : exceptionList) {
			try {
				std::rethrow_exception(e);
			}
			catch (cl::sycl::exception const &e) {
				std::cout << "fail" << std::endl;
				// std::terminate() will exit the process, return non-zero, and output a
				// message to the user about the exception
				std::terminate();
			}
		}
	};

	// FPGA device selector:  Emulator or Hardware
#ifdef FPGA_EMULATOR
	intel::fpga_emulator_selector device_selector;
#elif defined(FPGA)
	intel::fpga_selector device_selector;
#else
  // Initializing the devices queue with the default selector
  // The device queue is used to enqueue the kernels and encapsulates
  // all the states needed for execution
	default_selector device_selector;
#endif

	std::unique_ptr<queue> q;

	// Catch device selector runtime error
	try {
		q.reset(new queue(device_selector, ehandler));
	}
	catch (cl::sycl::exception const &e) {
		std::cout << "Caught a synchronous DPC++ exception:" << std::endl
			<< e.what() << std::endl;
		std::cout << "If you are targeting an FPGA hardware, please "
			"ensure that your system is plugged to an FPGA board that is "
			"set up correctly and compile with -DFPGA"
			<< std::endl;
		std::cout << "If you are targeting the FPGA emulator, compile with "
			"-DFPGA_EMULATOR."
			<< std::endl;
		std::terminate();
	}

	std::cout << "Device: " << q->get_device().get_info<info::device::name>()
		<< std::endl;

	return q;
}

//
// Computes the sum of two arrays in parallel using DPC++.
//
void add_arrays_parallel(IntArray &sum, const IntArray &addend_1,
	const IntArray &addend_2) {
	std::unique_ptr<queue> q = initialize_device_queue();

	// The range of the arrays managed by the buffer
	range<1> num_items{ array_size };

	// Buffers are used to tell DPC++ which data will be shared between the host
	// and the devices because they usually don't share physical memory
	// The pointer that's being passed as the first parameter transfers ownership
	// of the data to DPC++ at runtime. The destructor is called when the buffer
	// goes out of scope and the data is given back to the std::arrays.
	// The second parameter specifies the range given to the buffer.
	buffer<cl_int, 1> addend_1_buf(addend_1.data(), num_items);
	buffer<cl_int, 1> addend_2_buf(addend_2.data(), num_items);
	buffer<cl_int, 1> sum_buf(sum.data(), num_items);

	// queue::submit takes in a lambda that is passed in a command group handler
	// constructed at runtime. The lambda also contains a command group, which
	// contains the device-side operation and its dependencies
	q->submit([&](handler &h) {
		// Accessors are the only way to get access to the memory owned
		// by the buffers initialized above. The first get_access template parameter
		// specifies the access mode for the memory and the second template
		// parameter is the type of memory to access the data from; this parameter
		// has a default value
		auto addend_1_accessor = addend_1_buf.template get_access<dp_read>(h);
		auto addend_2_accessor = addend_2_buf.template get_access<dp_read>(h);

		// Note: Can use access::mode::discard_write instead of access::mode::write
		// because we're replacing the contents of the entire buffer.
		auto sum_accessor = sum_buf.template get_access<dp_write>(h);

		// Use parallel_for to run array addition in parallel. This executes the
		// kernel. The first parameter is the number of work items to use and the
		// second is the kernel, a lambda that specifies what to do per work item.
		// The template parameter ArrayAdd is used to name the kernel at runtime.
		// The parameter passed to the lambda is the work item id of the current
		// item.
		//
		// To remove the requirement to specify the kernel name you can enable
		// unnamed lamdba kernels with the option:
		//     dpcpp -fsycl-unnamed-lambda
		h.parallel_for<class ArrayAdd>(num_items, [=](id<1> i) {
			sum_accessor[i] = addend_1_accessor[i] + addend_2_accessor[i];
		});
	});

	// call wait_and_throw to catch async exception
	q->wait_and_throw();

	// DPC++ will enqueue and run the kernel. Recall that the buffer's data is
	// given back to the host at the end of the method's scope.
}

//
// Demonstrate summation of arrays both in scalar and parallel
//
int main() {
	IntArray addend_1, addend_2, sum_scalar, sum_parallel;

	// Initialize arrays with values from 0 to array_size-1
	initialize_array(addend_1);
	initialize_array(addend_2);
	initialize_array(sum_scalar);
	initialize_array(sum_parallel);

	// Add arrays in scalar and in parallel
	add_arrays_scalar(sum_scalar, addend_1, addend_2);
	add_arrays_parallel(sum_parallel, addend_1, addend_2);

	// Verify that the two sum arrays are equal
	for (size_t i = 0; i < sum_parallel.size(); i++) {
		if (sum_parallel[i] != sum_scalar[i]) {
			std::cout << "fail" << std::endl;
			return -1;
		}
	}
	std::cout << "success" << std::endl;

	return 0;
}
