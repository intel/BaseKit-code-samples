//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "../include/iso3dfd.h"

/*
 * Device-Code - Optimized for CPU
 *
 * SYCL implementation for single iteration of iso3dfd kernel
 * without using any shared local memory optimizations
 * 
 * ND-Range kernel is used to spawn work-items in x, y dimension
 * Each work-item then traverses in the z-dimension
 * 
 * z-dimension slicing can be used to vary the total number
 * global work-items.
 * 
 */

void iso_3dfd_iteration_cpu(cl::sycl::nd_item<3> it, float *next,
	float *prev, float *vel, const float *c,
	size_t nx, size_t nxy, size_t bx, size_t by, size_t z_offset, size_t full_end_z) {

	size_t id0 = it.get_local_id(0);
	size_t id1 = it.get_local_id(1);

	size_t begin_z = it.get_global_id(2)* z_offset + HALF_LENGTH;
	size_t end_z = begin_z + z_offset;
	if (end_z > full_end_z) end_z = full_end_z;
	
	size_t gid = (it.get_global_id(0) + bx) + ((it.get_global_id(1) + by) * nx) + (begin_z * nxy);

	for (size_t i = begin_z; i < end_z; i++) {
		float value = c[0] * prev[gid];

#pragma unroll(HALF_LENGTH)
		for (size_t iter = 1; iter <= HALF_LENGTH; iter++)
		{
			value += c[iter] * ( 
					prev[gid + iter] + prev[gid - iter] +
					prev[gid + iter * nx] + prev[gid - iter * nx] +
					prev[gid + iter*nxy] + prev[gid - iter*nxy]
				);
		}	
		next[gid] = 2.0f*prev[gid] - next[gid] + value * vel[gid];
		gid += nxy;
	}
}


/*
 * Host-side SYCL Code
 *
 * Driver function for ISO3DFD SYCL code for CPU
 * Uses ptr_next and ptr_prev as ping-pong buffers to achieve
 * accelerated wave propogation 
 * 
 */
bool iso_3dfd_cpu(cl::sycl::queue& q, float* ptr_next, float* ptr_prev, float* ptr_vel, float* ptr_coeff,
	size_t n1, size_t n2, size_t n3,
	size_t bx, size_t by, size_t begin_z, size_t end_z,
	unsigned int nIterations) {

	size_t nx = n1;
	size_t nxy = n1 * n2;
	
	printTargetInfo(q, DIMX_CPU, DIMY_CPU);
	
	size_t sizeTotal = (size_t)(nxy * n3);
	buffer<float, 1> b_ptr_next(ptr_next, range<1>{sizeTotal});
	buffer<float, 1> b_ptr_prev(ptr_prev, range<1>{sizeTotal});
	buffer<float, 1> b_ptr_vel(ptr_vel, range<1>{sizeTotal});
	buffer<float, 1> b_ptr_coeff(ptr_coeff, range<1>{HALF_LENGTH + 1});


    for(unsigned int k=0;k<nIterations;k+=1)
    {		

		q.submit([&](handler &cgh) {

			auto next = b_ptr_next.get_access<access::mode::read_write>(cgh);
			auto prev = b_ptr_prev.get_access<access::mode::read_write>(cgh);
			auto vel = b_ptr_vel.get_access<access::mode::read>(cgh);
			auto coeff = b_ptr_coeff.get_access<access::mode::read, access::target::constant_buffer>(cgh);


			auto local_nd_range = range<3>(DIMX_CPU, DIMY_CPU, 1);
			auto global_nd_range = range<3>((n1-2*HALF_LENGTH), (n2-2*HALF_LENGTH), (n3 - 2 * HALF_LENGTH) / BLOCKZ_CPU);
	
			if(k % 2 == 0)	
				cgh.parallel_for<class iso_3dfd_kernel_cpu>(
					nd_range<3>{global_nd_range,
								local_nd_range},
				[=](nd_item<3> it) {
				iso_3dfd_iteration_cpu(it, next.get_pointer(), prev.get_pointer(),
					vel.get_pointer(), coeff.get_pointer(),
					nx, nxy, bx, by, BLOCKZ_CPU, end_z);
				});
			else
				cgh.parallel_for<class iso_3dfd_kernel_cpu_2>(
					nd_range<3>{global_nd_range,
							local_nd_range},
				[=](nd_item<3> it) {
				iso_3dfd_iteration_cpu(it, prev.get_pointer(), next.get_pointer(),
					vel.get_pointer(), coeff.get_pointer(),
					nx, nxy, bx, by, BLOCKZ_CPU, end_z);
				});							

		});

	}
	return true;

}
