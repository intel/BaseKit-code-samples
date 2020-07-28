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

#include "qrd.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>
#include <cstring>
#include <vector>

using std::vector;
using namespace cl::sycl;

template <int Begin, int End>
struct Unroller {
  template <typename Action>
  static void Step(const Action &action) {
    action(Begin);
    Unroller<Begin + 1, End>::Step(action);
  }
};

template <int End>
struct Unroller<End, End> {
  template <typename Action>
  static void Step(const Action &action) {}
};

struct MyComplex {
  float xx;
  float yy;
  float &x() { return xx; }
  float &y() { return yy; }
  MyComplex(float x, float y) {
    xx = x;
    yy = y;
  }
  MyComplex() {}
  const MyComplex operator+(const MyComplex other) const {
    return MyComplex(xx + other.xx, yy + other.yy);
  }
};

MyComplex MulMycomplex(MyComplex a, MyComplex b) {
  MyComplex c;
  c.x() = a.x() * b.x() + a.y() * b.y();
  c.y() = a.y() * b.x() - a.x() * b.y();
  return c;
}

class QRD;

void SyclDevice(vector<float> &in_matrix, vector<float> &out_matrix, queue &q,
                int matrices, int reps) {
  int load_iter = COLS_COMPONENT * ROWS_COMPONENT / 4;
  int store_iter = COLS_COMPONENT * ROWS_COMPONENT / 4;

  const int kNum = 4;
  int chunk = 2048;
  if (matrices % chunk) {
    chunk = 1;
  }

  // Create buffers and allocate space for them.
  buffer<float, 1> *input_matrix[kNum], *output_matrix[kNum];
  for (int i = 0; i < kNum; i++) {
    input_matrix[i] =
        new buffer<float, 1>(ROWS_COMPONENT * COLS_COMPONENT * 2 * chunk);
    output_matrix[i] =
        new buffer<float, 1>((ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 * chunk);
  }

  for (int i = 0; i < reps; i++) {
    for (int i = 0, it = 0; it < matrices; it += chunk, i = (i + 1) % kNum) {
      const float *kptr =
          in_matrix.data() + ROWS_COMPONENT * COLS_COMPONENT * 2 * it;
      float *kptr2 =
          out_matrix.data() + (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 * it;
      int matrices = chunk;

      q.submit([&](handler &h) {
        auto in_matrix2 =
            input_matrix[i]->get_access<access::mode::discard_write>(h);
        h.copy(kptr, in_matrix2);
      });

      q.submit([&](handler &h) {
        auto in_matrix = input_matrix[i]->get_access<access::mode::read>(h);
        auto out_matrix =
            output_matrix[i]->get_access<access::mode::discard_write>(h);
        auto out_matrix2 = out_matrix;
        h.single_task<class QRD>([=]() [[intel::kernel_args_restrict]] {
          for (int l = 0; l < matrices; l++) {
            [[intelfpga::bankwidth(4 * 8),
              intelfpga::numbanks(ROWS_VECTOR / 4)]] struct {
              MyComplex d[ROWS_COMPONENT];
            } a_matrix[COLS_COMPONENT], ap_matrix[COLS_COMPONENT],
                aload_matrix[COLS_COMPONENT];

            MyComplex vector_ai[ROWS_COMPONENT], vector_ti[ROWS_COMPONENT];
            MyComplex s_or_i[COLS_COMPONENT];

            // Copy data from DDR memory to on-chip memory.
            int idx = l * ROWS_COMPONENT / 4 * COLS_COMPONENT;
            for (short i = 0; i < load_iter; i++) {
              MyComplex tmp[4];
              Unroller<0, 4>::Step([&](int k) {
                tmp[k].x() = in_matrix[idx * 2 * 4 + k * 2];
                tmp[k].y() = in_matrix[idx * 2 * 4 + k * 2 + 1];
              });

              idx++;
              int jtmp = i % (ROWS_COMPONENT / 4);

              Unroller<0, ROWS_COMPONENT / 4>::Step([&](int k) {
                Unroller<0, 4>::Step([&](int t) {
                  if (jtmp == k) {
                    aload_matrix[i / (ROWS_COMPONENT / 4)].d[k * 4 + t] =
                        tmp[t];
                  }

                  // Delay data signals to create a vine-based data distribution
                  // to lower signal fanout.
                  tmp[t].x() = intel::fpga_reg(tmp[t].x());
                  tmp[t].y() = intel::fpga_reg(tmp[t].y());
                });

                jtmp = intel::fpga_reg(jtmp);
              });
            }

            float p_ii_x, i_r_ii_x;
            short i = -1;
            short j = N_VALUE - SAFE_COLS < 0 ? (N_VALUE - SAFE_COLS) : 0;
            int qr_idx = l * (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 / 2;

            [[intelfpga::ii(1)]] [[intelfpga::ivdep(
                FIXED_ITERATIONS)]] for (int s = 0; s < ITERATIONS; s++) {
              MyComplex vector_t[ROWS_COMPONENT];
              MyComplex sori[ROWS_COMPONENT / 4];

              bool j_eq_i[ROWS_COMPONENT / 4], i_gt_0[ROWS_COMPONENT / 4],
                  i_ge_0_j_eq_i[ROWS_COMPONENT / 4],
                  j_eq_i_plus_1[ROWS_COMPONENT / 4], i_lt_0[ROWS_COMPONENT / 4];

              Unroller<0, ROWS_COMPONENT / 4>::Step([&](int k) {
                i_gt_0[k] = intel::fpga_reg(i > 0);
                i_lt_0[k] = intel::fpga_reg(i < 0);
                j_eq_i[k] = intel::fpga_reg(j == i);
                i_ge_0_j_eq_i[k] = intel::fpga_reg(i >= 0 && j >= i);
                j_eq_i_plus_1[k] = intel::fpga_reg(j == i + 1);
                sori[k].x() = intel::fpga_reg(s_or_i[j].x());
                sori[k].y() = intel::fpga_reg(s_or_i[j].y());
              });

              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                vector_t[k] = aload_matrix[j].d[k];
                if (i_gt_0[k / 4]) vector_t[k] = a_matrix[j].d[k];
                if (j_eq_i[k / 4]) vector_ai[k] = vector_t[k];
              });

              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                vector_t[k] =
                    MulMycomplex(vector_ai[k], i_lt_0[k / 4]
                                                   ? MyComplex(0.0, 0.0)
                                                   : sori[k / 4]) +
                    (j_eq_i[k / 4] ? MyComplex(0.0, 0.0) : vector_t[k]);
                if (i_ge_0_j_eq_i[k / 4])
                  ap_matrix[j].d[k] = a_matrix[j].d[k] = vector_t[k];
                if (j_eq_i_plus_1[k / 4]) vector_ti[k] = vector_t[k];
              });

              MyComplex p_ij = MyComplex(0, 0);
              Unroller<0, ROWS_COMPONENT>::Step([&](int k) {
                p_ij = p_ij + MulMycomplex(vector_t[k], vector_ti[k]);
              });

              if (j == i + 1) {
                p_ii_x = p_ij.x();
                i_r_ii_x = rsqrt(p_ij.x());
              }

              MyComplex s_ij =
                  MyComplex(0.0f - (p_ij.x()) / p_ii_x, p_ij.y() / p_ii_x);

              if (j >= 0) {
                s_or_i[j] = MyComplex(j == i + 1 ? i_r_ii_x : s_ij.x(),
                                      j == i + 1 ? 0.0f : s_ij.y());
              }

              MyComplex r_ii = j == i + 1 ? MyComplex(sqrt(p_ii_x), 0.0)
                                          : MyComplex(i_r_ii_x * p_ij.x(),
                                                      i_r_ii_x * p_ij.y());

              if (j >= i + 1 && i + 1 < N_VALUE) {
                out_matrix[qr_idx * 2] = r_ii.x();
                out_matrix[qr_idx * 2 + 1] = r_ii.y();
                qr_idx++;
              }

              if (j == N_VALUE - 1) {
                j = ((N_VALUE - SAFE_COLS) > i) ? (i + 1)
                                                : (N_VALUE - SAFE_COLS);
                i++;
              } else {
                j++;
              }
            }

            qr_idx /= 4;
            for (short i = 0; i < store_iter; i++) {
              int desired = i % (ROWS_COMPONENT / 4);
              bool get[ROWS_COMPONENT / 4];
              Unroller<0, ROWS_COMPONENT / 4>::Step([&](int k) {
                get[k] = desired == k;
                desired = intel::fpga_reg(desired);
              });

              MyComplex tmp[4];
              Unroller<0, ROWS_COMPONENT / 4>::Step([&](int t) {
                Unroller<0, 4>::Step([&](int k) {
                  tmp[k].x() =
                      get[t]
                          ? ap_matrix[i / (ROWS_COMPONENT / 4)].d[t * 4 + k].x()
                          : intel::fpga_reg(tmp[k].x());
                  tmp[k].y() =
                      get[t]
                          ? ap_matrix[i / (ROWS_COMPONENT / 4)].d[t * 4 + k].y()
                          : intel::fpga_reg(tmp[k].y());
                });
              });

              Unroller<0, 4>::Step([&](int k) {
                out_matrix2[qr_idx * 2 * 4 + k * 2] = tmp[k].x();
                out_matrix2[qr_idx * 2 * 4 + k * 2 + 1] = tmp[k].y();
              });

              qr_idx++;
            }
          }
        });
      });

      q.submit([&](handler &h) {
        auto final_matrix = output_matrix[i]->get_access<access::mode::read>(h);
        h.copy(final_matrix, kptr2);
      });
    }
  }

  for (int i = 0; i < kNum; i++) {
    delete input_matrix[i];
    delete output_matrix[i];
  }
}
