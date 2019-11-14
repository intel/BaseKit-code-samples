
//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include "rtm_stencil.h"

#include <CL/sycl.hpp>
#include <chrono>

#include <iostream>

using namespace cl::sycl;

int main(int argc, char* argv[]) {
	// Initialization
	g_grid3D = new float*[2];
	g_grid3D[0] = new float[nsize];
	g_grid3D[1] = new float[nsize];
	g_vsq = new float[nsize];
	auto current_time=0;
	auto start_time=0;
	printf("Order-%d 3D-Stencil (%d points) with space %dx%dx%d and time %d\n",
		2 * c_distance, c_distance * 2 * 3 + 1, c_num_x, c_num_y, c_num_z, c_time);

	init_variables();
	try {

		/* Create a default_selector to select a device to execute on. */
		default_selector mySelector;

		
		/* Create a queue from the default_selector to create an implicit context
		* and queue to execute with. */
		queue myQueue(mySelector);
		std::cout << "Running on "
			<< myQueue.get_device().get_info<cl::sycl::info::device::name>()
			<< "\n";

		{
			buffer <float, 1> coefbuffer(c_coef, range<1>(c_distance + 1));
			buffer<float, 1> velocitybuffer(g_vsq, range<1>(n1 * n2 *n3));
			buffer<float, 1> gridbuffer1(g_grid3D[0], range<1>(n1 * n2 * n3));
			buffer<float, 1> gridbuffer2(g_grid3D[1], range<1>(n1 * n2 * n3));
			auto global_nd_range = range<3>((n1 - 2 * c_distance), (n2 - 2 * c_distance), (n3 - 2 * c_distance));
			
	
		

			auto start_time = std::chrono::high_resolution_clock::now();
			
			for (int t = 0; t < 40; ++t) {
				myQueue.submit([&](handler &cgh) {



					/* Create accessors for accessing the input and output data within the
					* kernel. */
					auto inputPtrvel = velocitybuffer.get_access<access::mode::read>(cgh);
					auto inputPtrcoef = coefbuffer.get_access<access::mode::read, access::target::constant_buffer>(cgh);
					auto inputPtrgrid1 = gridbuffer1.get_access<access::mode::read_write>(cgh);
					auto inputPtrgrid2 = gridbuffer2.get_access<access::mode::read_write>(cgh);
					cgh.parallel_for<class rtmstencil>(
						global_nd_range,
						[=](nd_item<3> it) {

						/* Retreive the  global id for the current work item. */
						size_t gid = (it.get_global_id(0) + c_distance) + ((it.get_global_id(1) + c_distance) * n1) + ((it.get_global_id(2) + c_distance ) * n1*n2);

						
						float div = inputPtrcoef[0] * inputPtrgrid1[gid];

						for (size_t iter = 1; iter <= c_distance; iter++){
							div = inputPtrcoef[iter] * (
								inputPtrgrid1[gid + iter] + inputPtrgrid1[gid - iter] +
						 		inputPtrgrid1[gid + iter * n1] + inputPtrgrid1[gid - iter * n1] +
								inputPtrgrid1[gid + iter * n1*n2] + inputPtrgrid1[gid - iter * n1*n2]
								);
						}

						 inputPtrgrid2[gid] =  2 * inputPtrgrid1[gid] - inputPtrgrid2[gid] +  inputPtrvel[gid] * div;
						 
					});
				});


			}

			auto current_time = std::chrono::high_resolution_clock::now();
			std::cout << "Program has been running for " << std::chrono::duration<double>(current_time - start_time).count() << " seconds" << std::endl;
		}
	}

	catch (exception e) {

		/* In the case of an exception being throw, print theerror message and
		* return 1. */
		std::cout << e.what();
		return 1;
	}

	//std::cout << "Program has been running for " << std::chrono::duration<double>(current_time - start_time).count() << " seconds" << std::endl;
	print_summary((char*)"stencil_loop");
	print_y((char*)"parallel");
}
