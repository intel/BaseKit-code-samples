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

#include "qrd.h"

#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>
#include <cstring>
#include <vector>

using std::vector;
using namespace std::chrono;
using namespace cl::sycl;

template <int Begin, int End>
struct unroller {
  template <typename Action>
  static void step(const Action &action) {
    action(Begin);
    unroller<Begin + 1, End>::step(action);
  }
};

template <int End>
struct unroller<End, End> {
  template <typename Action>
  static void step(const Action &action) {}
};

struct mycomplex {
  float xx;
  float yy;
  float &x() { return xx; }
  float &y() { return yy; }
  mycomplex(float x, float y) {
    xx = x;
    yy = y;
  }
  mycomplex() {}
  const mycomplex operator+(const mycomplex other) const {
    return mycomplex(xx + other.xx, yy + other.yy);
  }
};

mycomplex mul_mycomplex(mycomplex a, mycomplex b) {
  mycomplex c;
  c.x() = a.x() * b.x() + a.y() * b.y();
  c.y() = a.y() * b.x() - a.x() * b.y();
  return c;
}

class QRD;

void sycl_device(vector<float> &in_matrix, vector<float> &out_matrix,
                 cl::sycl::queue &deviceQueue, int iterations, int reps) {
  int loaditer = COLS_COMPONENT * ROWS_COMPONENT / 4;
  int storeiter = COLS_COMPONENT * ROWS_COMPONENT / 4;

  const int NUM = 4;
  int chunk = 2048;
  if (iterations % chunk) {
    chunk = 1;
  }
  // create buffers, allocate space for them
  buffer<float, 1> *inputMatrix[NUM], *outputMatrix[NUM];
  for (int i = 0; i < NUM; i++) {
    inputMatrix[i] =
        new buffer<float, 1>(ROWS_COMPONENT * COLS_COMPONENT * 2 * chunk);
    outputMatrix[i] =
        new buffer<float, 1>((ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 * chunk);
  }
  for (int i = 0; i < reps; i++) {
    for (int i = 0, it = 0; it < iterations; it += chunk, i = (i + 1) % NUM) {
      const float *ptr =
          in_matrix.data() + ROWS_COMPONENT * COLS_COMPONENT * 2 * it;
      float *ptr2 =
          out_matrix.data() + (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 * it;
      int iterations = chunk;
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto in_matrix2 =
            inputMatrix[i]->template get_access<access::mode::discard_write>(
                cgh);
        cgh.copy(ptr, in_matrix2);
      });
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto in_matrix =
            inputMatrix[i]->template get_access<access::mode::read>(cgh);
        auto out_matrix =
            outputMatrix[i]->template get_access<access::mode::discard_write>(
                cgh);
        auto out_matrix2 = out_matrix;
        cgh.single_task<class QRD>([=]() {
          for (int iter = 0; iter < iterations; iter++) {
            [[intelfpga::bankwidth(4 * 8)]] [
                [intelfpga::numbanks(ROWS_VECTOR / 4)]] struct {
              mycomplex d[ROWS_COMPONENT];
            } a_matrix[COLS_COMPONENT], ap_matrix[COLS_COMPONENT],
                aload_matrix[COLS_COMPONENT];
            mycomplex vector_ai[ROWS_COMPONENT], vector_ti[ROWS_COMPONENT];
            mycomplex s_or_i[COLS_COMPONENT];

            // ===== Copying data from DDR to on-chip =====
            int idx = iter * ROWS_COMPONENT / 4 * COLS_COMPONENT;
            for (short i = 0; i < loaditer; i++) {
              mycomplex tmp[4];
              unroller<0, 4>::step([&](int k) {
                tmp[k].x() = in_matrix[idx * 2 * 4 + k * 2];
                tmp[k].y() = in_matrix[idx * 2 * 4 + k * 2 + 1];
              });
              idx++;
              int jtmp = i % (ROWS_COMPONENT / 4);

              unroller<0, ROWS_COMPONENT / 4>::step([&](int k) {
                unroller<0, 4>::step([&](int t) {
                  if (jtmp == k)
                    aload_matrix[i / (ROWS_COMPONENT / 4)].d[k * 4 + t] =
                        tmp[t];
                  // delay data signals to create vine-based data distribution
                  // to lower signal fanout
                  tmp[t].x() = intel::fpga_reg(tmp[t].x());
                  tmp[t].y() = intel::fpga_reg(tmp[t].y());
                });
                jtmp = intel::fpga_reg(jtmp);
              });
            }

            float piix, iriix;
            short i = -1, j = N_VALUE - SAFE_COLS < 0 ? (N_VALUE - SAFE_COLS) : 0;
            int qr_idx = iter * (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3 / 2;
            [[intelfpga::ii(1)]] [[intelfpga::ivdep(
                FIXED_ITERATIONS)]] for (int s = 0; s < ITERATIONS; s++) {
              mycomplex vector_t[ROWS_COMPONENT];
              mycomplex sori[ROWS_COMPONENT / 4];
              bool jeqi[ROWS_COMPONENT / 4], igt0[ROWS_COMPONENT / 4],
                  ige0jgeqi[ROWS_COMPONENT / 4], jeqiplus1[ROWS_COMPONENT / 4],
                  ilt0[ROWS_COMPONENT / 4];
              unroller<0, ROWS_COMPONENT / 4>::step([&](int k) {
                igt0[k] = intel::fpga_reg(i > 0);
                ilt0[k] = intel::fpga_reg(i < 0);
                jeqi[k] = intel::fpga_reg(j == i);
                ige0jgeqi[k] = intel::fpga_reg(i >= 0 && j >= i);
                jeqiplus1[k] = intel::fpga_reg(j == i + 1);
                sori[k].x() = intel::fpga_reg(s_or_i[j].x());
                sori[k].y() = intel::fpga_reg(s_or_i[j].y());
              });
              unroller<0, ROWS_COMPONENT>::step([&](int k) {
                vector_t[k] = aload_matrix[j].d[k];
                if (igt0[k / 4]) vector_t[k] = a_matrix[j].d[k];
                if (jeqi[k / 4]) vector_ai[k] = vector_t[k];
              });

              unroller<0, ROWS_COMPONENT>::step([&](int k) {
                vector_t[k] = mul_mycomplex(vector_ai[k],
                                            ilt0[k / 4] ? mycomplex(0.0, 0.0)
                                                        : sori[k / 4]) +
                              (jeqi[k / 4] ? mycomplex(0.0, 0.0) : vector_t[k]);
                if (ige0jgeqi[k / 4])
                  ap_matrix[j].d[k] = a_matrix[j].d[k] = vector_t[k];
                if (jeqiplus1[k / 4]) vector_ti[k] = vector_t[k];
              });
              mycomplex pij = mycomplex(0, 0);
              unroller<0, ROWS_COMPONENT>::step([&](int k) {
                pij = pij + mul_mycomplex(vector_t[k], vector_ti[k]);
              });
              if (j == i + 1) {
                piix = pij.x();
                iriix = rsqrt(pij.x());
              }
              mycomplex sij =
                  mycomplex(0.0f - (pij.x()) / piix, pij.y() / piix);
              if (j >= 0) {
                s_or_i[j] = mycomplex(j == i + 1 ? iriix : sij.x(),
                                      j == i + 1 ? 0.0f : sij.y());
              }
              mycomplex rii = j == i + 1
                                  ? mycomplex(cl::sycl::sqrt(piix), 0.0)
                                  : mycomplex(iriix * pij.x(), iriix * pij.y());
              if (j >= i + 1 && i + 1 < N_VALUE) {
                out_matrix[qr_idx * 2] = rii.x();
                out_matrix[qr_idx * 2 + 1] = rii.y();
                qr_idx++;
              }
              if (j == N_VALUE - 1) {
                j = ((N_VALUE - SAFE_COLS) > i) ? (i + 1) : (N_VALUE - SAFE_COLS);
                i++;
              } else {
                j++;
              }
            }
            qr_idx /= 4;
            for (short i = 0; i < storeiter; i++) {
              int desired = i % (ROWS_COMPONENT / 4);
              bool get[ROWS_COMPONENT / 4];
              unroller<0, ROWS_COMPONENT / 4>::step([&](int k) {
                get[k] = desired == k;
                desired = intel::fpga_reg(desired);
              });
              mycomplex tmp[4];
              unroller<0, ROWS_COMPONENT / 4>::step([&](int t) {
                unroller<0, 4>::step([&](int k) {
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
              unroller<0, 4>::step([&](int k) {
                out_matrix2[qr_idx * 2 * 4 + k * 2] = tmp[k].x();
                out_matrix2[qr_idx * 2 * 4 + k * 2 + 1] = tmp[k].y();
              });
              qr_idx++;
            }
          }
        });
      });
      deviceQueue.submit([&](cl::sycl::handler &cgh) {
        auto final_matrix =
            outputMatrix[i]->template get_access<access::mode::read>(cgh);
        cgh.copy(final_matrix, ptr2);
      });
    }
  }
  for (int i = 0; i < NUM; i++) {
    delete inputMatrix[i];
    delete outputMatrix[i];
  }
}
