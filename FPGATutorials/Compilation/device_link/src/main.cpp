//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "kernel.h"
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace cl::sycl;

class SimpleAdd;

int main() {
  std::vector<float> vec_a(ARRAY_SIZE);
  std::vector<float> vec_b(ARRAY_SIZE);
  std::vector<float> vec_r(ARRAY_SIZE);
  // Fill vectors a and b with random float values
  int count = ARRAY_SIZE;
  for (int i = 0; i < count; i++) {
    vec_a[i] = rand() / (float)RAND_MAX;
    vec_b[i] = rand() / (float)RAND_MAX;
  }
  run_kernel(vec_a, vec_b, vec_r);
  // Test the results
  int correct = 0;
  float tmp;
  for (int i = 0; i < count; i++) {
    tmp = vec_a[i] + vec_b[i];
    tmp -= vec_r[i];
    if (tmp * tmp < TOL * TOL) {
      correct++;
    }
  }
  // summarize results
  if (correct == count) {
    std::cout << "PASSED: results are correct\n";
  } else {
    std::cout << "FAILED: results are incorrect\n";
  }

  return !(correct == count);
}