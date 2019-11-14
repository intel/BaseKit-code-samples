
//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#pragma once
#include <algorithm>
#ifndef RTMSTENCIL_H
#define RTMSTENCIL_H
using std::min;
using std::max;

//constants
// Neighborhood distance in each dimension
const int c_distance = 4;
// Number of points in direction x
const int c_num_x = 200;
// Number of points in direction y
const int c_num_y = 200;
// Number of points in direction z
const int c_num_z = 100;
// Time
const int c_time = 40;

const int n1 = c_num_x + (2*c_distance);
const int n2 = c_num_y + (2*c_distance);
const int n3 = c_num_z + (2*c_distance);

const int nsize = n1 * n2 * n3;

// Coefficients for differnt distances in order
const float c_coef[c_distance + 1] = { -1435.0f / 504 * 3, 1.6f, -0.2f, 8.0f / 315, -1.0f / 560.0f };

// Points to array space storing 3D grid values
// The code uses two 3D arrays of coordinates, one for even values of t and the other for odd values.
extern float **g_grid3D;

// Phase velocities
extern float *g_vsq;


void init_variables();


void print_summary(char *header);

void print_y(char *name);
#endif
#pragma once
