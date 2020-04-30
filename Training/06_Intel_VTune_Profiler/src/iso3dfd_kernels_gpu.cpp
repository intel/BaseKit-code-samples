//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include "../include/iso3dfd.h"

/*
 * Device-Code - Optimized for GPU
 * SYCL implementation for single iteration of iso3dfd kernel
 * using shared local memory optimizations
 * 
 * ND-Range kernel is used to spawn work-items in x, y dimension
 * Each work-item then traverses in the z-dimension
 * 
 * z-dimension slicing can be used to vary the total number
 * global work-items.
 * 
 * SLM Padding can be used to eliminate SLM bank conflicts if 
 * there are any 
 */
void iso_3dfd_iteration_slm(cl::sycl::nd_item<3> it, float *next,
	float *prev, float *vel, const float *coeff, float *tab,
	size_t nx, size_t nxy, size_t bx, size_t by, size_t z_offset, int full_end_z) {

	size_t id0 = it.get_local_id(0);
	size_t id1 = it.get_local_id(1);

	size_t size0 = it.get_local_range(0) + 2 * HALF_LENGTH + PAD;
	size_t identifiant = (id0 + HALF_LENGTH) + (id1 + HALF_LENGTH)*size0;

	size_t begin_z = it.get_global_id(2)* z_offset + HALF_LENGTH;
	size_t end_z = begin_z + z_offset;
	if (end_z > full_end_z) end_z = full_end_z;

	size_t gid = (it.get_global_id(0) + bx) + ((it.get_global_id(1) + by) * nx) + (begin_z * nxy);


	float front[HALF_LENGTH + 1];
	float back[HALF_LENGTH];
	float c[HALF_LENGTH + 1];

        const unsigned int items_X = it.get_local_range(0);
        const unsigned int items_Y = it.get_local_range(1);

        bool copyHaloY = false, copyHaloX = false;
        if(id1 < HALF_LENGTH) copyHaloY = true;
        if(id0 < HALF_LENGTH) copyHaloX = true;
	
	for (unsigned int iter = 0; iter < HALF_LENGTH; iter++)
	{
		front[iter] = prev[gid + iter * nxy];
	}
	c[0] = coeff[0];
	
	for (unsigned int iter = 1; iter <= HALF_LENGTH; iter++)
	{
		back[iter-1] = prev[gid - iter * nxy];
		c[iter] = coeff[iter];
	}

	for (size_t i = begin_z; i < end_z; i++) {

                if(copyHaloY){
                        tab[identifiant - HALF_LENGTH * size0] = prev[gid - HALF_LENGTH * nx];
                        tab[identifiant + items_Y * size0] = prev[gid + items_Y * nx];
                }
                if(copyHaloX){
                        tab[identifiant - HALF_LENGTH] = prev[gid - HALF_LENGTH];
                        tab[identifiant + items_X] = prev[gid + items_X];
                }
                tab[identifiant] = front[0];
		
		it.barrier(access::fence_space::local_space);

		front[HALF_LENGTH] = prev[gid + HALF_LENGTH * nxy];
		float value = c[0] * front[0];

#pragma unroll(HALF_LENGTH)
		for (unsigned int iter = 1; iter <= HALF_LENGTH; iter++)
		{
			value += c[iter] * (front[iter] + back[iter - 1] + 
							tab[identifiant + iter] + tab[identifiant - iter] +
								tab[identifiant + iter * size0] + tab[identifiant - iter * size0]);
		}

		next[gid] = 2.0f*front[0] - next[gid] + value * vel[gid];

		gid += nxy;

		for (unsigned int iter = HALF_LENGTH-1; iter > 0; iter--)
		{
			back[iter] = back[iter-1];
		}
		back[0] = front[0];

		for (unsigned int iter = 0; iter < HALF_LENGTH; iter++)
		{
			front[iter] = front[iter + 1];
		}
		
		// This is to ensure that SLM buffers are not overwritten by next
		// set of work-items
		it.barrier(access::fence_space::local_space);

	}
}



/*
 * Device-Code - Optimized for GPU
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
void iso_3dfd_iteration_global(cl::sycl::nd_item<3> it, float *next,
	float *prev, float *vel, const float *coeff,
	int nx, int nxy, int bx, int by, int z_offset, int full_end_z) {

	size_t id0 = it.get_local_id(0);
	size_t id1 = it.get_local_id(1);

	size_t begin_z = it.get_global_id(2)* z_offset + HALF_LENGTH;
	size_t end_z = begin_z + z_offset;
	if (end_z > full_end_z) end_z = full_end_z;

	size_t gid = (it.get_global_id(0) + bx) + ((it.get_global_id(1) + by) * nx) + (begin_z * nxy);


	float front[HALF_LENGTH + 1];
	float back[HALF_LENGTH];
	float c[HALF_LENGTH + 1];

	for (unsigned int iter = 0; iter < HALF_LENGTH; iter++)
	{
		front[iter] = prev[gid + iter * nxy];
	}
	c[0] = coeff[0];
	for (unsigned int iter = 1; iter <= HALF_LENGTH; iter++)
	{
		c[iter] = coeff[iter];
		back[iter-1] = prev[gid - iter * nxy];
	}

	for (size_t i = begin_z; i < end_z; i++) {

		front[HALF_LENGTH] = prev[gid + HALF_LENGTH * nxy];
		float value = c[0] * front[0];

#pragma unroll(HALF_LENGTH)
		for (unsigned int iter = 1; iter <= HALF_LENGTH; iter++)
		{
			value += c[iter] * (front[iter] + back[iter - 1] +
				prev[gid + iter] + prev[gid - iter] +
				prev[gid + iter * nx] + prev[gid - iter * nx]);
		}

		next[gid] = 2.0f*front[0] - next[gid] + value * vel[gid];

		gid += nxy;

		for (unsigned int iter = HALF_LENGTH - 1; iter > 0; iter--)
		{
			back[iter] = back[iter - 1];
		}
		back[0] = front[0];

		for (unsigned int iter = 0; iter < HALF_LENGTH; iter++)
		{
			front[iter] = front[iter + 1];
		}
	}
}



/*
 * Host-side SYCL Code
 *
 * Driver function for ISO3DFD SYCL code
 * Uses ptr_next and ptr_prev as ping-pong buffers to achieve
 * accelerated wave propogation 
 * 
 * This function uses SYCL buffers to facilitate host to device
 * buffer copies
 * 
 */

bool iso_3dfd_device(cl::sycl::queue& q, float* ptr_next, float* ptr_prev, float* ptr_vel, float* ptr_coeff,
	size_t n1, size_t n2, size_t n3,
	size_t bx, size_t by, size_t begin_z, size_t end_z,
	unsigned int nIterations) {

	size_t nx = n1;
	size_t nxy = n1 * n2;
	
	printTargetInfo(q, DIMX, DIMY);
	
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


			auto local_nd_range = range<3>(DIMX, DIMY, 1);
			auto global_nd_range = range<3>((n1-2*HALF_LENGTH), (n2-2*HALF_LENGTH), (n3-2*HALF_LENGTH)/BLOCKZ);
	
#ifdef USE_SHARED
			auto localRange_ptr_prev = range<1>((DIMX + (2 * HALF_LENGTH) + PAD) * (DIMY + (2 * HALF_LENGTH)));
			accessor<float, 1, access::mode::read_write, access::target::local> tab(
				localRange_ptr_prev, cgh);


			if(k % 2 == 0)		  
				cgh.parallel_for<class iso_3dfd_kernel>(
					  nd_range<3>{global_nd_range,
								  local_nd_range},
							[=](nd_item<3> it) {
							iso_3dfd_iteration_slm(it, next.get_pointer(), prev.get_pointer(),
									vel.get_pointer(), coeff.get_pointer(),  tab.get_pointer(),
									nx, nxy, bx, by, BLOCKZ, end_z);
				});
			else
				cgh.parallel_for<class iso_3dfd_kernel_2>(
					  nd_range<3>{global_nd_range,
								  local_nd_range},
							[=](nd_item<3> it) {
							iso_3dfd_iteration_slm(it, prev.get_pointer(), next.get_pointer(),
									vel.get_pointer(), coeff.get_pointer(),  tab.get_pointer(),
									nx, nxy, bx, by, BLOCKZ, end_z);
				});
				
#else
			if(k % 2 == 0)	
				cgh.parallel_for<class iso_3dfd_kernel>(
					nd_range<3>{global_nd_range,
								local_nd_range},
				[=](nd_item<3> it) {
				iso_3dfd_iteration_global(it, next.get_pointer(), prev.get_pointer(),
					vel.get_pointer(), coeff.get_pointer(),
					nx, nxy, bx, by, BLOCKZ, end_z);
				});
			else
				cgh.parallel_for<class iso_3dfd_kernel_2>(
					nd_range<3>{global_nd_range,
							local_nd_range},
				[=](nd_item<3> it) {
				iso_3dfd_iteration_global(it, prev.get_pointer(), next.get_pointer(),
					vel.get_pointer(), coeff.get_pointer(),
					nx, nxy, bx, by, BLOCKZ, end_z);
				});							
#endif

		});

	}
	return true;

}

