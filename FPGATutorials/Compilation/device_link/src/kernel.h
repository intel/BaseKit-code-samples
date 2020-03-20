//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace cl::sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

#define TOL (0.001) // tolerance used in floating point comparisons
#define ARRAY_SIZE (32) // Length of vectors a, b and c

void run_kernel(std::vector<float> &va, std::vector<float> &vb,
                std::vector<float> &vr);