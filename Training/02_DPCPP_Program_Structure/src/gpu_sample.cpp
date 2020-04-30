//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

constexpr int num=16;
using namespace cl::sycl;

int main() {
// ****************************************
// The following defines host code
    
  // create a device queue
  //Specify the device type via device selector or use default selector.
  //In the below case we are selecting the gpu _selector    
  
  gpu_selector selector;
  //cpu_selector selector;
  //default_selector selector;
  //host_selector selector;
  queue q(selector);
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;
    
  auto R = range<1>{ num };
  //create buffer (represent both host and device memory)
  buffer<int> A{ R };
  //'q' submits a command for execution
  q.submit([&](handler& h) {
  //create accessors to access buffer data on the device
    auto out =
      A.get_access<access::mode::write>(h);

// ****************************************
    //The following defines device code
    //send a kernel (lambda) for execution  
    h.parallel_for(R, [=](id<1> idx) {
      out[idx] = idx[0]; }); });

// ****************************************
// And the following is back to device code
  
  auto result =
    A.get_access<access::mode::read>();
  for (int i=0; i<num; ++i)
    std::cout << result[i] << "\n";
  return 0;
}