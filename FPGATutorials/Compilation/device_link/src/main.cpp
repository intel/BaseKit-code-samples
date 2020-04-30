//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#include "kernel.hpp"
using namespace cl::sycl;

class SimpleAdd;

int main() {
  std::vector<float> vec_a(kArraySize);
  std::vector<float> vec_b(kArraySize);
  std::vector<float> vec_r(kArraySize);
  // Fill vectors a and b with random float values
  int count = kArraySize;
  for (int i = 0; i < count; i++) {
    vec_a[i] = rand() / (float)RAND_MAX;
    vec_b[i] = rand() / (float)RAND_MAX;
  }
  run_kernel(vec_a, vec_b, vec_r);
  // Test the results
  int correct = 0;
  for (int i = 0; i < count; i++) {
    float tmp = vec_a[i] + vec_b[i] - vec_r[i];
    if (tmp * tmp < kTol * kTol) {
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
