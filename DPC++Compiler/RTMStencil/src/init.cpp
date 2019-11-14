
//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================



// This code listed utility functions used to initialize the stencil array, print out summary report
// and run test with different parallelization and vectorization methods.

//#pragma once
#include "rtm_stencil.h"
#include <stdio.h>
#include <math.h>

// Points to memory storing the current and new/next space value
// The current and new space value is switching from g_grid3D[0] and g_grid3D[1] through Time
float **g_grid3D;

// Operator array stores values to calculate stencil in each dimension
float *g_vsq;

const int num_xy = n1*n2;

// Description:
// This function computes reference point from 3D space and Time in g_grid3D.
// [in]: t, x, y, z
// [out]: aref
static inline float &aref(int t, int x, int y, int z)
{
	return g_grid3D[t & 1][num_xy * z + n1 * y + x];
}

// Description:
// This function computes reference point from 3D space in g_vsq.
// [in]: x, y, z
// [out]: vsqref
static inline float &vsqref(int x, int y, int z)
{
	return g_vsq[num_xy * z + n1 * y + x];
}

// Description:
// This function computes the reference of one point from one dimension space in g_vsq.
// [in]: point_xyz32w
// [out]: vsqref
static inline float &vsqref(int point_xyz)
{
	return g_vsq[point_xyz];
}

// Description:
// This function initialize stencil array g_grid3D and operator array in g_vsq.
// [in]: 
// [out]: g_grid3D, g_vsq
void init_variables()
{
	for (int z = 0; z < n3; ++z)
		for (int y = 0; y < n2; ++y)
			for (int x = 0; x < n1; ++x) {
				/* set initial values */
				float r = fabs((float)(x - n1 / 2 + y - n2 / 2 + z - n3 / 2) / 30);
				r = max(1 - r, 0.0f) + 1;

				aref(0, x, y, z) = r;
				aref(1, x, y, z) = r;
				vsqref(x, y, z) = 0.001f;
			}
}


// Description:
// This function calculates throughput performance and 
// print out both time and throughput perfomrance result.
// [in]: header, interval
// [out]: 
void print_summary(char *header) {
	/* print timing information */
	long total = n1 * n2 * n3;
	printf("++++++++++ %s ++++++++++\n", header);
	printf("first non-zero numbers\n");
	for (int i = 0; i < total; i++) {
		if (g_grid3D[c_time % 2][i] != 0) {
			printf("%d: %fs\n", i, g_grid3D[c_time % 2][i]);
			break;
		}
	}

	/**double mul = c_num_x - 8;
	mul *= c_num_y - 8;
	mul *= c_num_z - 8;
	mul *= c_time;
	double perf = mul / (interval * 1e6);

	printf("Benchmark time: %f sec\n", interval);**/

	// Mcells/sec means number of Million cells are processed per second
	// M-FAdd/s means number of floating point add operations are processed per second for one million cells
	// M-FMul/s means number of floating point multiply operations are processed per second for one million cells
	/**printf("Perf: %f Mcells/sec (%f M-FAdd/s, %f M-FMul/s)\n",
		perf,
		perf * 26,
		perf * 7);**/
}


// Description:
// This function output value of points in dimension y to file y_points_name.txt file.
// Name is passing to identify the result for different methods.
// [in]: name
// [out]: 
void print_y(char *name) {
	char filename[35];
	sprintf(filename, "y_points_%s.txt", name);
	FILE *fout = fopen(filename, "w");
	int z = c_num_z / 2;
	int x = c_num_x / 2;
	for (int y = 0; y < c_num_y; y++) {
		fprintf(fout, "%f\n", aref(c_time, x, y, z));
	}
	fclose(fout);
	printf("Done writing output\n");
}

