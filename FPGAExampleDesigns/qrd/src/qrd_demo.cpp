// ==============================================================
// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

#include <math.h>

#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>
#include <list>

#include "qrd.hpp"

#define RANDOM_SEED 1138
#define RANDOM_MIN 1
#define RANDOM_MAX 10

using namespace std;
using namespace std::chrono;
using namespace cl::sycl;

auto exception_handler = [](exception_list exceptions) {
  for (exception_ptr const &e : exceptions) {
    try {
      rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
      terminate();
    }
  }
};

void SyclDevice(vector<float> &in_matrix, vector<float> &out_matrix, queue &q,
                int matrices, int reps);

int main(int argc, char *argv[]) {
  int matrices = argc > 1 ? atoi(argv[1]) : 1;
  if (matrices < 1) {
    cout << "Must run at least 1 matrix"
         << "\n";
    return 1;
  }

  try {
#if defined(FPGA_EMULATOR)
    intel::fpga_emulator_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif

    queue q = queue(device_selector, exception_handler);
    device device = q.get_device();
    cout << "Device name: " << device.get_info<info::device::name>().c_str()
         << "\n";

    vector<float> a_matrix;
    vector<float> qr_matrix;

    a_matrix.resize(matrices * ROWS_COMPONENT * COLS_COMPONENT * 2);
    qr_matrix.resize(matrices * (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3);

    // For output-postprocessing
    float q_matrix[ROWS_COMPONENT][COLS_COMPONENT][2];
    float r_matrix[R_COMPONENT][R_COMPONENT][2];

    cout << "Generating " << matrices << " random matri"
         << ((matrices == 1) ? "x " : "ces ") << "\n";

    srand(RANDOM_SEED);

    for (int i = 0; i < matrices; i++) {
      for (int row = 0; row < ROWS_COMPONENT; row++) {
        for (int col = 0; col < COLS_COMPONENT; col++) {
          int random_val = rand();
          float random_double =
              random_val % (RANDOM_MAX - RANDOM_MIN) + RANDOM_MIN;
          a_matrix[i * ROWS_COMPONENT * COLS_COMPONENT * 2 +
                   col * ROWS_COMPONENT * 2 + row * 2] = random_double;
          int random_val_imag = rand();
          random_double =
              random_val_imag % (RANDOM_MAX - RANDOM_MIN) + RANDOM_MIN;
          a_matrix[i * ROWS_COMPONENT * COLS_COMPONENT * 2 +
                   col * ROWS_COMPONENT * 2 + row * 2 + 1] = random_double;
        }
      }
    }

    SyclDevice(a_matrix, qr_matrix, q, 1, 1);  // Accelerator warmup

#if defined(FPGA_EMULATOR)
    int reps = 2;
#else
    int reps = 32;
#endif
    cout << "Running QR decomposition of " << matrices << " matri"
         << ((matrices == 1) ? "x " : "ces ")
         << ((reps > 1) ? "repeatedly" : "") << "\n";

    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    SyclDevice(a_matrix, qr_matrix, q, matrices, reps);
    high_resolution_clock::time_point end_time = high_resolution_clock::now();
    duration<double> diff = end_time - start_time;
    q.throw_asynchronous();

    cout << "   Total duration:   " << diff.count() << " s"
         << "\n";
    cout << "Throughput: " << reps * matrices / diff.count() / 1000
         << "k matrices/s"
         << "\n";

    list<int> to_check;
    // We will check at least matrix 0
    to_check.push_back(0);
    // Spot check the last and the middle one
    if (matrices > 2) to_check.push_back(matrices / 2);
    if (matrices > 1) to_check.push_back(matrices - 1);

    cout << "Verifying results on matrix";

    for (int matrix : to_check) {
      cout << " " << matrix;
      int idx = 0;
      for (int i = 0; i < R_COMPONENT; i++) {
        for (int j = 0; j < R_COMPONENT; j++) {
          if (j < i)
            r_matrix[i][j][0] = r_matrix[i][j][1] = 0;
          else {
            r_matrix[i][j][0] =
                qr_matrix[matrix * (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 +
                          idx++];
            r_matrix[i][j][1] =
                qr_matrix[matrix * (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 +
                          idx++];
          }
        }
      }

      for (int j = 0; j < COLS_COMPONENT; j++) {
        for (int i = 0; i < ROWS_COMPONENT; i++) {
          q_matrix[i][j][0] =
              qr_matrix[matrix * (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 +
                        idx++];
          q_matrix[i][j][1] =
              qr_matrix[matrix * (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 +
                        idx++];
        }
      }

      float acc_real = 0;
      float acc_imag = 0;
      float v_matrix[ROWS_COMPONENT][COLS_COMPONENT][2] = {{{0}}};
      for (int i = 0; i < ROWS_COMPONENT; i++) {
        for (int j = 0; j < COLS_COMPONENT; j++) {
          acc_real = 0;
          acc_imag = 0;
          for (int k = 0; k < COLS_COMPONENT; k++) {
            acc_real += q_matrix[i][k][0] * r_matrix[k][j][0] -
                        q_matrix[i][k][1] * r_matrix[k][j][1];
            acc_imag += q_matrix[i][k][0] * r_matrix[k][j][1] +
                        q_matrix[i][k][1] * r_matrix[k][j][0];
          }
          v_matrix[i][j][0] = acc_real;
          v_matrix[i][j][1] = acc_imag;
        }
      }

      float error = 0;
      int count = 0;
      for (int row = 0; row < ROWS_COMPONENT; row++) {
        for (int col = 0; col < COLS_COMPONENT; col++) {
          if (std::isnan(v_matrix[row][col][0]) ||
              std::isnan(v_matrix[row][col][1])) {
            count++;
          }
          float real = v_matrix[row][col][0] -
                       a_matrix[matrix * ROWS_COMPONENT * COLS_COMPONENT * 2 +
                                col * ROWS_COMPONENT * 2 + row * 2];
          float imag = v_matrix[row][col][1] -
                       a_matrix[matrix * ROWS_COMPONENT * COLS_COMPONENT * 2 +
                                col * ROWS_COMPONENT * 2 + row * 2 + 1];
          if (sqrt(real * real + imag * imag) >= 1e-4) {
            error += sqrt(real * real + imag * imag);
            count++;
          }
        }
      }

      if (count > 0) {
        cout << "\n"
             << "!!!!!!!!!!!!!! Error = " << error << " in " << count << " / "
             << ROWS_COMPONENT * COLS_COMPONENT << "\n";
        return 1;
      }
    }

    cout << "\n"
         << "PASSED"
         << "\n";

    return 0;
  } catch (cl::sycl::exception const &e) {
    cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    cout << "   If you are targeting an FPGA hardware, "
            "ensure that your system is plugged to an FPGA board that is "
            "set up correctly"
         << "\n";
    cout << "   If you are targeting the FPGA emulator, compile with "
            "-DFPGA_EMULATOR"
         << "\n";

    terminate();
  }
}
