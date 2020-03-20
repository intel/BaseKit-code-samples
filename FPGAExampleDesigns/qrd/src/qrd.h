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

// -- matrix parameters
#ifndef ROWS_COMPONENT
#define ROWS_COMPONENT 128
#endif

#ifndef COLS_COMPONENT
#define COLS_COMPONENT 128
#endif

#ifndef V_FACTOR
#define V_FACTOR 1
#endif

#ifndef FIXED_ITERATIONS
#define FIXED_ITERATIONS 64
#endif

#define ROWS_VECTOR (ROWS_COMPONENT / V_FACTOR)

#define MAT_SIZE (ROWS_COMPONENT * COLS_COMPONENT)

#define R_COMPONENT COLS_COMPONENT

// -- architecture/design parameters
#define N_VALUE COLS_COMPONENT

#define SAFE_COLS ((FIXED_ITERATIONS + V_FACTOR - 1 + V_FACTOR - 1) / V_FACTOR)

#define M_MINUS_COLS \
  (SAFE_COLS > COLS_COMPONENT ? SAFE_COLS - COLS_COMPONENT : 0)
#define ITERATIONS                                        \
  (COLS_COMPONENT + M_MINUS_COLS +                        \
   (COLS_COMPONENT + 1) * COLS_COMPONENT * V_FACTOR / 2 + \
   SAFE_COLS * (SAFE_COLS - 1) * V_FACTOR / 2 -           \
   M_MINUS_COLS * (M_MINUS_COLS - 1) * V_FACTOR / 2 + V_FACTOR - 1)
