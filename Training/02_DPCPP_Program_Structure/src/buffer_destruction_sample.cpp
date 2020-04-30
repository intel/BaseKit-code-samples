//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iostream>
using namespace cl::sycl;

int main() {
  constexpr int N = 100;
  auto R = range<1>(N);
  std::vector<double> v(N, 10);
  queue q;
  //Buffer creation happens within a separate C++ scope.
  {
    //Buffer takes ownership of the data stored in vector
    buffer<double, 1> buf(v.data(), R);
    q.submit([&](handler& h) {
    auto a = buf.get_access<access::mode::read_write>(h);
    h.parallel_for(R, [=](id<1> i) {
      a[i] += 2;
      });
    });
  }
  //When execution advances beyond this scope, buffer destructor is invoked which relinquishes the ownership of data and copies back the data to the host memory.
  for (int i = 0; i < N; i++)
    std::cout << v[i] << "\n";
  return 0;
}