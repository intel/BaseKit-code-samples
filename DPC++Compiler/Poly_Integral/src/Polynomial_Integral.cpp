//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iomanip>
#include <vector>

using namespace cl::sycl;
using namespace std;
static const int N = 100;

// Defining the slice of the dx as 1000
#define d_x 1000

// An example polynomail function. ax^2+bx+c
float function_x(float x) { return (2 * x * x * +3 * x * -3 * x / 8 + 1 / 4); }

// a and v are the arrays of the lower bound and the upper bounds of the x-axis.
// d is the result of the area after calculating the integral of the above
// Polynomial function
void dpcpp_parallel(queue& q, float* a, float* v, float* d) {
  // Setup buffers for input and output vectors
  buffer<float, 1> bufv1(a, range<1>(N));
  buffer<float, 1> bufv2(v, range<1>(N));
  buffer<float, 1> bufv3(d, range<1>(N));

  std::cout << "Target Device: "
            << q.get_device().get_info<info::device::name>() << "\n";
  // Submit Command group function object to the queue
  q.submit([&](handler& h) {
    auto acc_vect1 = bufv1.get_access<access::mode::read>(h);
    auto acc_vect2 = bufv2.get_access<access::mode::read>(h);
    auto acc_vect3 = bufv3.get_access<access::mode::write>(h);

    h.parallel_for<class CompIntegral>(range<1>(N), [=](id<1> i) {
      float dx = (float)((acc_vect1[i] - acc_vect2[i]) / d_x);
      float area_int = 0;

      for (int j = 0; j < d_x; j++) {
        float xC = acc_vect1[i] + dx * j;
        float yC = function_x(xC);
        area_int = xC * yC;
        area_int += area_int;
      }
      acc_vect3[i] = area_int;
    });
  });
  q.wait_and_throw();
}

int main() {
  float lBound[N], uBound[N], i_area[N], d[N];

  // Initialize the lower bound and upper bound of the x axis arrays. Below we
  // are initializing such that upper bound is always greater than the lower
  // bound.
  for (int i = 0; i < N; i++) {
    lBound[i] = i + 40 + 10;
    uBound[i] = (i + 1) * 40 + 70;
    i_area[i] = 0;
    d[i] = 0;
  }
  // this exception handler with catch async exceptions
  auto exception_handler = [&](cl::sycl::exception_list eList) {
    for (std::exception_ptr const& e : eList) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::cout << "Failure" << std::endl;
        std::terminate();
      }
    }
  };
  try {
    // queue constructor passed exception handler
    queue q(default_selector{}, exception_handler);
    // Call the dpcpp_parallel function with lBound and uBound as inputs and
    // i_area array as the output
    dpcpp_parallel(q, lBound, uBound, i_area);
  } catch (...) {
    // some other exception detected
    std::cout << "Failure" << std::endl;
    std::terminate();
  }
  cout << "****************************************Calculating Integral area "
          "in Parallel********************************************************"
       << std::endl;
  for (int i = 0; i < N; i++) {
    cout << "Area: " << i_area[i] << ' ';
    if (i == N - 1) {
      cout << "\n"
           << "\n";
    }
  }
  return 0;
}
