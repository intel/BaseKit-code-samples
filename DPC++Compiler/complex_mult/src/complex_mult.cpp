//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iomanip>
#include <vector>

#include "Complex.hpp"

using namespace cl::sycl;
using namespace std;

// Number of complex numbers passing to the DPC++ code
static const int N = 100;

// v1 and v2 are the vectors with N complex nubers and are inputs to the
// parallel function
void dpcpp_parallel(queue &q, std::vector<Complex2> &v1,
                    std::vector<Complex2> &v2, std::vector<Complex2> &v3) {
  // Setup input buffers
  buffer<Complex2, 1> bufv1(v1.data(), range<1>(N));
  buffer<Complex2, 1> bufv2(v2.data(), range<1>(N));

  // Setup Output buffers
  buffer<Complex2, 1> bufv3(v3.data(), range<1>(N));

  std::cout << "Target Device: "
            << q.get_device().get_info<info::device::name>() << "\n";
  // Submit Command group function object to the queue
  q.submit([&](handler &h) {
    // Accessors set as read mode
    auto acc_vect1 = bufv1.template get_access<access::mode::read>(h);
    auto acc_vect2 = bufv2.template get_access<access::mode::read>(h);
    // Accessor set to Write mode
    auto acc_vect3 = bufv3.template get_access<access::mode::write>(h);
    h.parallel_for<class CompMult>(range<1>(N), [=](id<1> i) {
      // Kernel code. Call the complex_mul function here.
      acc_vect3[i] = acc_vect1[i].complex_mul(acc_vect2[i]);
    });
  });
  q.wait_and_throw();
}
void dpcpp_scalar(std::vector<Complex2> &v1, std::vector<Complex2> &v2,
                  std::vector<Complex2> &v3) {
  for (int i = 0; i < N; i++) {
    v3[i] = v1[i].complex_mul(v2[i]);
  }
}
// Compare the results of the two output vectors from parallel and scalar. They
// should be equal
int Compare(std::vector<Complex2> &v1, std::vector<Complex2> &v2) {
  int retCode = 1;
  for (int i = 0; i < N; i++) {
    if (v1[i] == v2[i]) {
      continue;
    } else {
      retCode = -1;
      break;
    }
  }
  return retCode;
}
int main() {
  // Declare your Input and Output vectors of the Complex2 class
  vector<Complex2> vect1;
  vector<Complex2> vect2;
  vector<Complex2> vect3;
  vector<Complex2> vect4;

  for (int i = 0; i < N; i++) {
    vect1.push_back(Complex2(i + 2, i + 4));
    vect2.push_back(Complex2(i + 4, i + 6));
    vect3.push_back(Complex2(0, 0));
    vect4.push_back(Complex2(0, 0));
  }

  // this exception handler with catch async exceptions
  auto exception_handler = [&](cl::sycl::exception_list eList) {
    for (std::exception_ptr const &e : eList) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const &e) {
        std::cout << "Failure" << std::endl;
        std::terminate();
      }
    }
  };

  // Initialize your Input and Output Vectors. Inputs are initialized as below.
  // Outputs are initialized with 0
  try {
    // queue constructor passed exception handler
    queue q(default_selector{}, exception_handler);
    // Call the dpcpp_parallel with the required inputs and outputs
    dpcpp_parallel(q, vect1, vect2, vect3);
  } catch (...) {
    // some other exception detected
    std::cout << "Failure" << std::endl;
    std::terminate();
  }

  cout << "****************************************Multiplying Complex numbers "
          "in Parallel********************************************************"
       << std::endl;
  // Print the outputs of the Parallel function
  for (int i = 0; i < N; i++) {
    cout << vect3[i] << ' ';
    if (i == N - 1) {
      cout << "\n"
           << "\n";
    }
  }
  cout << "****************************************Multiplying Complex numbers "
          "in Serial***********************************************************"
       << std::endl;
  // Call the dpcpp_scalar function with the required input and outputs
  dpcpp_scalar(vect1, vect2, vect4);
  for (auto it = vect4.begin(); it != vect4.end(); it++) {
    cout << *it << ' ';
    if (it == vect4.end() - 1) {
      cout << "\n"
           << "\n";
    }
  }

  // Compare the outputs from the parallel and the scalar functions. They should
  // be equal

  int retCode = Compare(vect3, vect4);
  if (retCode == 1) {
    cout << "********************************************Success. Results are "
            "matched******************************"
         << "\n";
  } else
    cout << "*********************************************Failed. Results are "
            "not matched**************************"
         << "\n";

  // cout<<d<<"\n";
  return 0;
}
