//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <chrono>
#include <cmath>
#include "CL/sycl.hpp"
#include "device_selector.hpp"

#define IMG_WIDTH 2048
#define IMG_HEIGHT 2048
#define IMG_SIZE (IMG_WIDTH*IMG_HEIGHT)
#define CHANNELS_PER_PIXEL 4

static void init(float *image, size_t len) {
  for (size_t i = 0; i < len; i++) {
    image[i] = i % 255;
  }
}

static size_t verify(float *gold, float *test, size_t len) {
  size_t error_cnt = 0;

  for (size_t i = 0; i < len; i++) {
    float g = gold[i];
    float v = test[i];

    if (fabs(v - g) > 0.0001f) {
      if (++error_cnt < 10) {
        std::cout << "ERROR AT [" << i << "]: " << v << " != " << g << " (expected)" << std::endl;
      }
    }
  }
  return error_cnt;
}

static void report_time(const std::string &msg, cl::sycl::event e) {
  cl::sycl::cl_ulong time_start = e.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
  cl::sycl::cl_ulong time_end = e.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  std::cout << msg << elapsed << " milliseconds" << std::endl;
}

// SYCL does not need any special mark-up for functions which are called from
// SYCL kernel and defined in the same compilation unit. SYCL compiler must be
// able to find the full call graph automatically.
// always_inline as calls are expensive on Gen GPU.
// Notes:
// - coeffs can be declared outside of the function, but still must be constant
// - SYCL compiler will automatically deduce the address space for the two
//   pointers; cl::sycl::multi_ptr specialization for particular address space
//   can used for more control
__attribute__((always_inline))
static void sepia_impl(float *src_image, float *dst_image, int i) {
  const float coeffs[] =
  { 0.2f, 0.3f, 0.3f, 0.0f,
    0.1f, 0.5f, 0.5f, 0.0f,
    0.3f, 0.1f, 0.1f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f };

  i *= CHANNELS_PER_PIXEL;

  for (int j = 0; j < 4; ++j) {
    float w = 0.0f;

    for (int k = 0; k < 4; ++k) {
      w += coeffs[4 * j + k] * src_image[i + k];
    }
    dst_image[i + j] = w;
  }
}

// Few useful acronyms; 'using namespace cl::sycl;' also helps.
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;
constexpr auto sycl_global_buffer = cl::sycl::access::target::global_buffer;

// This is alternative (to a lambda) representation of a SYCL kernel.
// Internally, compiler transforms lambdas into instances of a very simlar
// class. With functors, capturing kernel parameters is done manually via the
// constructor, unlike automatic capturing with lambdas.
class SepiaFunctor {
public:
  // Constructor captures needed data into fields
  SepiaFunctor(
    cl::sycl::accessor<float, 1, sycl_read, sycl_global_buffer> &image_acc_,
    cl::sycl::accessor<float, 1, sycl_write, sycl_global_buffer> &image_exp_acc_)
    :
    image_acc(image_acc_), image_exp_acc(image_exp_acc_)
  {}

  // The '()' operator is the actual kernel
  void operator()(cl::sycl::id<1> i) {
    sepia_impl(image_acc.get_pointer(), image_exp_acc.get_pointer(), i.get(0));
  }

private:
  // Captured values:
  cl::sycl::accessor<float, 1, sycl_read, sycl_global_buffer> image_acc;
  cl::sycl::accessor<float, 1, sycl_write, sycl_global_buffer> image_exp_acc;
};

int main(int argc, char **argv) {
  // prepare data
  size_t num_pixels = IMG_SIZE;
  size_t img_len = num_pixels * CHANNELS_PER_PIXEL + CHANNELS_PER_PIXEL;
  float *image = new float[img_len];
  float *image_ref = new float[img_len];
  float *image_exp1 = new float[img_len];
  float *image_exp2 = new float[img_len];
  init(image, img_len);
  std::memset(image_ref, 0, img_len * sizeof(float));
  std::memset(image_exp1, 0, img_len * sizeof(float));
  std::memset(image_exp2, 0, img_len * sizeof(float));
  img_len -= CHANNELS_PER_PIXEL;

  // Create a device selector which rates available devices in the preferred order
  // for the runtime to select the highest rated device
  MyDeviceSelector sel;
  // Using these events to time command group execution
  cl::sycl::event e1, e2;

  // Wrap main SYCL API calls into a try/catch to diagnose potential errors
  try {
    // Create a command queue using the device selector above, and request profiling
    auto propList = cl::sycl::property_list{ cl::sycl::property::queue::enable_profiling() };
    cl::sycl::queue q(sel, propList);

    // See what device was actually selected for this queue.
    std::cout << "Running on " << q.get_device().get_info<cl::sycl::info::device::name>() << "\n";

    // Create SYCL buffer representing source data .
    //
    // By default, this buffers will be created with global_buffer access
    // target, which means the buffer "projection" to the device (actual
    // device memory chunk allocated or mapped on the device to reflect
    // buffer's data) will belong to the SYCL global address space - this
    // is what host data usually maps to. Other address spaces are:
    // private, local and constant.
    // Notes:
    // - access type (read/write) is not specified when creating a buffer -
    //   this is done when actuall accessor is created
    // - there can be multiple accessors to the same buffer in multuple command
    //   groups
    // - 'image' pointer was passed to the constructor, so this host memory will
    //   be used for "host projection", no allocation will happen on host
    cl::sycl::buffer<float, 1> image_buf(image, cl::sycl::range<1>(img_len));

    // This is the output buffer device writes to
    cl::sycl::buffer<float, 1> image_buf_exp1(image_exp1, cl::sycl::range<1>(img_len));

    std::cout << "submitting lambda kernel..." << std::endl;

    // Submit a command group for execution. Returns immediately, not waiting
    // for command group completion.
    e1 = q.submit([&](cl::sycl::handler& cgh) {
      // This lambda defines a "command group" - a set of commands for the
      // device sharing some state and executed in-order - i.e. creation of
      // accessors may lead to on-device memory allocation, only after that
      // the kernel will be enqueued.
      // A command group can contain at most one parallel_for, single_task or
      // parallel_for_workgroup construct.
      auto image_acc = image_buf.get_access<sycl_read>(cgh);
      auto image_exp_acc = image_buf_exp1.get_access<sycl_write>(cgh);

      // This is the simplest form cl::sycl::handler::parallel_for -
      // - it specifies "flat" 1D ND range (num_pixels), runtime will select
      //   local size
      // - kernel lambda accepts single cl::sycl::id argument, which has very
      //   limited API; see the spec for more complex forms
      // <clas sepia> is the kernel name required by the spec, the lambda
      // parameter of the parallel_for is the kernel, which actually executes
      // on device
      cgh.parallel_for<class sepia>(cl::sycl::range<1>(IMG_SIZE), [=](cl::sycl::id<1> i) {
        sepia_impl(image_acc.get_pointer(), image_exp_acc.get_pointer(), i.get(0));
      });
    });

    std::cout << "submitting functor kernel..." << std::endl;

    cl::sycl::buffer<float, 1> image_buf_exp2(image_exp2, cl::sycl::range<1>(img_len));

    // Submit another command group. This time kernel is represented as a
    // functor object.
    e2 = q.submit([&](cl::sycl::handler& cgh) {
      auto image_acc = image_buf.get_access<sycl_read>(cgh);
      auto image_exp_acc = image_buf_exp2.get_access<sycl_write>(cgh);
      SepiaFunctor kernel(image_acc, image_exp_acc);
      cgh.parallel_for(cl::sycl::range<1>(num_pixels), kernel);
    });

    std::cout << "waiting for execution to complete..." << std::endl;
    // q.wait_and_throw();
    // don't need to explicitly wait at this point: image_buf_exp1 and
    // image_buf_exp2 destructors will automatically block execution until all
    // tasks which write to them finish, then update the host images of the
    // buffers (copy from device to host)
  }
  catch (cl::sycl::exception e) {
    // This catches only synchronous exceptions happened in current thread
    // during execution. To catch asynchronous exceptions caused by execution
    // of the command group above, the asynchronous error handler mechanism
    // must be used (not shown here). Asynchronous error handler can be
    // installed globally and/or per queue.
    // Synchronous exceptions are usually those which are thrown from the SYCL
    // runtime code, such as on invalid constructor arguments.
    // Ans example of asynchronous exceptions is error occurred during
    // execution of a kernel.
    // Make sure cl::sycl::exception is caught, not std::exception.
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  }
  std::cout << "Execution completed" << std::endl;

  // report execution times:
  report_time("lambda kernel time: ", e1);
  report_time("functor kernel time: ", e2);

  // get reference result
  for (size_t i = 0; i < num_pixels; i++) {
    sepia_impl(image, image_ref, i);
  }

  // verify
  std::cout << "Verifying kernel..." << std::endl;
  size_t error_cnt = verify(image_ref, image_exp1, img_len);
  std::cout << "Verifying functor..." << std::endl;
  error_cnt += verify(image_ref, image_exp2, img_len);
  std::cout << (error_cnt ? "FAILED" : "passed") << std::endl;
  return error_cnt ? 1 : 0;
}

