//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <vector>
#include "Projectile.hpp"

using namespace cl::sycl;
using namespace std;

#define PI 3.1415
#define g 9.81
#define NITERATIONS 100

// Function to calculate the range, maximum height and total flight time of a
// projectile
inline void calculate_range(Projectile& obj, Projectile& pObj) {
  float nAngle = obj.getangle();
  float nVel = obj.getvelocity();
  // for trignometric functions use cl::sycl::sin/cos
  float sinV = cl::sycl::sin(nAngle * PI / 180);
  float cosV = cl::sycl::cos(nAngle * PI / 180);
  float totalT = cl::sycl::fabs((2 * nVel * sinV)) / g;
  float range_s = cl::sycl::fabs(nVel * totalT * cosV);
  float max_height =
      (nVel * nVel * sinV * sinV) / 2 * g;  // h = v^2 * sin^2theta/2g

  pObj.setRangeandTime(range_s, totalT, nAngle, nVel, max_height);
}

// Compare the results of the two output vectors from parallel and scalar. They
// should be equa
int Compare(std::vector<Projectile>& v1, std::vector<Projectile>& v2) {
  int retCode = 1;
  for (int i = 0; i < NITERATIONS; i++) {
    if (v1[i] == v2[i]) {
      continue;
    } else {
      retCode = -1;
      break;
    }
  }
  return retCode;
}

// v1 and v2 are the vectors with N Projectile numbers and are inputs to the
// parallel function
void dpcpp_parallel(queue& q, std::vector<Projectile>& v1,
                    std::vector<Projectile>& v2) {

  buffer<Projectile, 1> bufv1(v1.data(), range<1>(NITERATIONS));
  buffer<Projectile, 1> bufv2(v2.data(), range<1>(NITERATIONS));

  std::cout << "Target Device: "
            << q.get_device().get_info<info::device::name>() << "\n";

  // Submit Command group function object to the queue
  q.submit([&](handler& h) {
    // Input accessors set as read_write mode
    auto acc_vect1 = bufv1.get_access<access::mode::read_write>(h);

    // Output accessor set to write mode.
    auto acc_vect2 = bufv2.get_access<access::mode::write>(h);

    h.parallel_for<class projectIP>(range<1>(NITERATIONS), [=](id<1> i) {
      // Call the Inline function calculate_range
      calculate_range(acc_vect1[i], acc_vect2[i]);
    });
  });
  q.wait_and_throw();
}
// scalar function to calculate the range
void dpcpp_scalar(std::vector<Projectile>& v1, std::vector<Projectile>& v2) {
  for (int i = 0; i < NITERATIONS; i++) {
    calculate_range(v1[i], v2[i]);
  }
}

int main() {
  srand(time(NULL));
  float angle = 0;
  float vel = 0;
  vector<Projectile> vecT1, vecT2, vecT3;
  // Initialize the Input and Output vectors
  for (int i = 0; i < NITERATIONS; i++) {
    angle = rand() % 90 + 10;
    vel = rand() % 400 + 10;
    vecT1.push_back(Projectile(angle, vel, 1.0, 1.0, 1.0));
    vecT2.push_back(Projectile());
    vecT3.push_back(Projectile());
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
    // Call the dpcpp_parallel with the required inputs and outputs
    dpcpp_parallel(q, vecT1, vecT2);
  } catch (...) {
    // some other exception detected
    std::cout << "Failure" << std::endl;
    std::terminate();
  }

  // call the dpcpp_scalar with vecT1 as input and vecT3 as output
  dpcpp_scalar(vecT1, vecT3);
  for (int i = 0; i < NITERATIONS; i++) {
    // Displaying the Scalar computation results. Uncomment to viee the results
    // cout<<"Scalar "<<vecT3[i];
  }
  for (int i = 0; i < NITERATIONS; i++) {
    // Displaying the Parallel computation results.
    cout << "Parallel " << vecT2[i];
  }
  // Compare the vectors of both the outputs of parallal and scalar functions
  int retCode = Compare(vecT2, vecT3);
  if (retCode == 1) {
    cout
        << "********************************************Success..The Results "
           "are matched********************************************************"
        << "\n";
  } else
    cout << "*********************************************Failed***************"
            "*****************************************"
         << "\n";
  return 0;
}
