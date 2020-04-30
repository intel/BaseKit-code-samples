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
static const int num_elements = 100;

// Defining the number of slices on X as 1000
static const int number_of_slices = 1000;

// An example polynomail function. ax^2+bx+c
float CalculateYValue(float x) {
  return (2 * x * x * +3 * x * -3 * x / 8 + 1 / 4);
}

// a and v are the arrays of the lower bound and the upper bounds of the x-axis.
// d is the result of the area after calculating the integral of the above
// Polynomial function
void DpcppParallel(queue& q, float* l_bound, float* u_bound, float* out_area) {
  // Setup buffers for input and output vectors
  buffer<float, 1> buf_l_bound(l_bound, range<1>(num_elements));
  buffer<float, 1> buf_u_bound(u_bound, range<1>(num_elements));
  buffer<float, 1> buf_out_area(out_area, range<1>(num_elements));

  std::cout << "Target Device: "
            << q.get_device().get_info<info::device::name>() << "\n";
  // Submit Command group function object to the queue
  q.submit([&](handler& h) {
    auto A = buf_l_bound.get_access<access::mode::read>(h);
    auto B = buf_u_bound.get_access<access::mode::read>(h);
    auto C = buf_out_area.get_access<access::mode::write>(h);

    h.parallel_for(range<1>(num_elements), [=](id<1> i) {
      float x_val = (float)((A[i] - B[i]) / number_of_slices);
      float area_int = 0;

      for (int j = 0; j < number_of_slices; j++) {
        float x_len = A[i] + x_val * j;
        float y_len = CalculateYValue(x_len);
        area_int = x_len * y_len;
        area_int += area_int;
      }
      C[i] = area_int;
    });
  });
  q.wait_and_throw();
}

int main() {
  float lower_bound[num_elements], upper_bound[num_elements],
      i_area[num_elements];

  // Initialize the lower bound and upper bound of the x axis arrays. Below we
  // are initializing such that upper bound is always greater than the lower
  // bound.
  for (int i = 0; i < num_elements; i++) {
    lower_bound[i] = i + 40 + 10;
    upper_bound[i] = (i + 1) * 40 + 70;
    i_area[i] = 0;
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
    // Call the DpcppParallel function with lower_bound and upper_bound as
    // inputs and i_area array as the output
    DpcppParallel(q, lower_bound, upper_bound, i_area);
  } catch (...) {
    // some other exception detected
    std::cout << "Failure" << std::endl;
    std::terminate();
  }
  cout << "****************************************Calculating Integral area "
          "in Parallel********************************************************"
       << std::endl;
  for (int i = 0; i < num_elements; i++) {
    cout << "Area: " << i_area[i] << ' ';
    if (i == num_elements - 1) {
      cout << "\n\n";
    }
  }
  return 0;
}
