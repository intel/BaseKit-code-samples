//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 256;

int main() {
  /* ordered_queue*/
  ordered_queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;

  int *data = static_cast<int *>(
      malloc_shared(N * sizeof(int), q.get_device(), q.get_context()));
  for (int i = 0; i < N; i++) data[i] = 10;

  q.submit([&](handler &h) {
    h.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });
  });

  q.submit([&](handler &h) {
    h.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
  });

  q.submit([&](handler &h) {
    h.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 5; });
  }).wait();

  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << std::endl;
  free(data, q.get_context());
  return 0;
}
