//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace cl::sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

// tolerance used in floating point comparisons
constexpr float kTol = 0.001;

// array size of vectors a, b and c
constexpr unsigned int kArraySize = 32;

void run_kernel(std::vector<float> &va, std::vector<float> &vb,
                std::vector<float> &vr);
