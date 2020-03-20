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

#ifndef __CRR_COMMON_H__
#define __CRR_COMMON_H__

#define MAX_N_STEPS 8189
#define INPUT_FILE "src/data/ordered_inputs.csv"
#define OUTPUT_FILE "src/data/ordered_outputs.csv"

// SPATIAL_UNROLL defines unroll factor of L6 in crr_main_func()
#ifndef SPATIAL_UNROLL
#define SPATIAL_UNROLL 32
#endif

// Data structure as the inputs to FPGA.
typedef struct {
 public:
  double nSteps; /* nSteps = number of time steps in the binomial tree. */
  double u[3]; /* u = the increase factor of a up movement in the binomial tree,
                  same for each time step. */
  double u2[3]; /* u2 = the square of increase factor. */
  double c1[3]; /* c1 = the probality of a down movement in the binomial tree,
                   same for each time step. */
  double c2[3]; /* c2 = the probality of a up movement in the binomial tree. */
  double umin[3]; /* umin = minimum price of the underlying at the maturity. */
  double param_1[3];
  double param_2;
} crr_in_params;

// Data structure for original input data.
typedef struct {
  int CP;        /* CP = -1 or 1 for Put & Call respectively. */
  double nSteps; /* nSteps = number of time steps in the binomial tree. */
  double Strike; /* Strike = exercise price of option. */
  double Spot;   /* Spot = spot price of the underlying. */
  double Fwd;    /* Fwd = forward price of the underlying. */
  double Vol;    /* Vol = per cent volatility, input as a decimal. */
  double DF;     /* DF = Spot/Fwd */
  double T;      /* T = time in years to the maturity of the option. */

} input_data;

// Data structure as the output from function crr_main_func().
typedef struct {
  double pgreek[4]; /* pgreek[] = auxiliary array for post-calculate five
                       Greeks. */
  double vals[3];   /* vals[] = auxiliary array for post-calculate option prices
                       and five Greeks. */

} crr_res_params;

// Data structure as the output from FPGA.
typedef struct {
  double pgreek[4]; /* pgreek[] = auxiliary array for post-calculate five
                       Greeks. */
  double val; /* val = auxiliary variable for post-calculate option prices and
                 five Greeks. */

} func_params;

// Data structure for option price and five Greeks.
typedef struct {
  double value; /* value = option price. */
  double delta;
  double gamma;
  double vega;
  double theta;
  double rho;
} crr_res;

#endif
