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

/////////////////////////////////////////////////////////////////////////////////////////
//
// CRRSolver CPU/FPGA Accelerator Demo Program
//
/////////////////////////////////////////////////////////////////////////////////////////
//
// This design implments simple Cox-Ross-Rubinstein(CRR) binomial tree model
// with greeks for American exercise options. Optimization summay:
//    -- Area-consuming and low-frequency calculation (pre-calculation and
//    post-calculation) is done on CPU.
//    -- Parallel operations in critical loop(L6).
//    -- Pipelined loop L1, L2, L3, L4 and L6.
//
// The following diagram shows the mechanism of optimizations to CRR.
//
//
//                                               +------+         ^
//                                 +------------>|optVal|         |
//                                 |             | [2]  |         |
//                                 |             +------+         |
//                                 |                              |
//                                 |                              |
//                              +--+---+                          |
//                +------------>|optVal|                          |
//                |             | [1]  |                          |
//                |             +--+---+                          |
//                |                |                              |
//                |                |                              |
//                |                |                              |   Loop6(L6)
//                |                |                              |   updates
//            +---+--+             +------------>+------+         |   multiple
//            |optVal|                           |optVal|         |   elements
//            | [0]  |                           | [1]  |         |   in
//            optVal[]
//            +---+--+             +------------>+------+         |
//            simultaneously
//                |                |                              |
//                |                |                              |
//                |                |                              |
//                |                |                              |
//                |             +--+---+                          |
//                |             |optVal|                          |
//                +------------>| [0]  |                          |
//                              +--+---+                          |
//                                 |                              |
//                                 |                              |
//                                 |             +------+         |
//                                 |             |optVal|         |
//                                 +------------>| [0]  |         |
//                                               +------+         +
//
//
//
//
//                             nSteps=1          nSteps=2
//
//
//                <------------------------------------------+
//                  Loop5(L5) updates each level of the tree
//
//
//

#include "CRR_common.h"
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

using namespace std;
using namespace std::chrono;
using namespace cl::sycl;

void ReadInputFromFile(ifstream &inputFile, vector<input_data> &inp) {

  string lineOfArgs;
  while (getline(inputFile, lineOfArgs)) {
    input_data temp;
    istringstream lineOfArgs_ss(lineOfArgs);
    lineOfArgs_ss >> temp.nSteps;
    lineOfArgs_ss.ignore(1, ',');
    lineOfArgs_ss >> temp.CP;
    lineOfArgs_ss.ignore(1, ',');
    lineOfArgs_ss >> temp.Spot;
    lineOfArgs_ss.ignore(1, ',');
    lineOfArgs_ss >> temp.Fwd;
    lineOfArgs_ss.ignore(1, ',');
    lineOfArgs_ss >> temp.Strike;
    lineOfArgs_ss.ignore(1, ',');
    lineOfArgs_ss >> temp.Vol;
    lineOfArgs_ss.ignore(1, ',');
    lineOfArgs_ss >> temp.DF;
    lineOfArgs_ss.ignore(1, ',');
    lineOfArgs_ss >> temp.T;

    inp.push_back(temp);
  }
}

static string to_string_with_precision(const double value, const int p = 6) {
  ostringstream out;
  out.precision(p);
  out << std::fixed << value;
  return out.str();
}

void WriteOutputToFile(ofstream &outputFile, vector<crr_res> &outp) {
  int n = outp.size();
  for (int i = 0; i < n; ++i) {
    crr_res temp;
    temp = outp[i];
    string line = to_string_with_precision(temp.value, 12) + " " +
                  to_string_with_precision(temp.delta, 12) + " " +
                  to_string_with_precision(temp.gamma, 12) + " " +
                  to_string_with_precision(temp.vega, 12) + " " +
                  to_string_with_precision(temp.theta, 12) + " " +
                  to_string_with_precision(temp.rho, 12) + "\n";

    outputFile << line;
  }
}

#define MAX_STRING_LEN 40

bool findGetArgString(std::string arg, const char *str, char *str_value,
                      size_t maxchars) {
  std::size_t found = arg.find(str, 0, strlen(str));
  if (found != std::string::npos) {
    const char *sptr = &arg.c_str()[strlen(str)];
    for (int i = 0; i < maxchars - 1; i++) {

      char ch = sptr[i];
      switch (ch) {
      case ' ':
      case '\t':
      case '\0':
        str_value[i] = 0;
        return true;
        break;
      default:
        str_value[i] = ch;
        break;
      }
    }
    return true;
  }
  return false;
}

// Perform data pre-processing work using the CPU.
crr_in_params prepare_data(input_data &inp) {
  crr_in_params in_params;
  in_params.nSteps = inp.nSteps;

  double R[2];
  R[0] = cl::sycl::pow(inp.DF, 1.0 / inp.nSteps);
  double dDF = exp(inp.T / 10000);
  R[1] = cl::sycl::pow(inp.DF * dDF, 1.0 / inp.nSteps);
  in_params.u[0] = exp(inp.Vol * sqrt(inp.T / inp.nSteps));
  in_params.u[1] = in_params.u[0];
  in_params.u[2] = exp((inp.Vol + 0.0001) * sqrt(inp.T / inp.nSteps));

  in_params.u2[0] = in_params.u[0] * in_params.u[0];
  in_params.u2[1] = in_params.u[1] * in_params.u[1];
  in_params.u2[2] = in_params.u[2] * in_params.u[2];
  in_params.umin[0] =
      inp.Spot * cl::sycl::pow(1 / in_params.u[0], inp.nSteps + 2);
  in_params.umin[1] = inp.Spot * cl::sycl::pow(1 / in_params.u[1], inp.nSteps);
  in_params.umin[2] = inp.Spot * cl::sycl::pow(1 / in_params.u[2], inp.nSteps);
  in_params.c1[0] =
      R[0] *
      (in_params.u[0] - cl::sycl::pow(inp.Fwd / inp.Spot, 1.0 / inp.nSteps)) /
      (in_params.u[0] - 1 / in_params.u[0]);
  in_params.c1[1] = R[1] *
                    (in_params.u[1] - cl::sycl::pow((inp.Fwd / dDF) / inp.Spot,
                                                    1.0 / inp.nSteps)) /
                    (in_params.u[1] - 1 / in_params.u[1]);
  in_params.c1[2] =
      R[0] *
      (in_params.u[2] - cl::sycl::pow(inp.Fwd / inp.Spot, 1.0 / inp.nSteps)) /
      (in_params.u[2] - 1 / in_params.u[2]);
  in_params.c2[0] = R[0] - in_params.c1[0];
  in_params.c2[1] = R[1] - in_params.c1[1];
  in_params.c2[2] = R[0] - in_params.c1[2];

  in_params.param_1[0] = inp.CP * in_params.umin[0];
  in_params.param_1[1] = inp.CP * in_params.umin[1];
  in_params.param_1[2] = inp.CP * in_params.umin[2];
  in_params.param_2 = inp.CP * inp.Strike;

  return in_params;
}

// Perform data post-processing work using the CPU.
crr_res postprocess_data(input_data &inp, crr_in_params &in_params,
                         crr_res_params &res_params) {
  double h;
  crr_res res;
  h = inp.Spot * (in_params.u2[0] - 1 / in_params.u2[0]);
  res.value = res_params.pgreek[1];
  res.delta = (res_params.pgreek[2] - res_params.pgreek[0]) / h;
  res.gamma = 2 / h *
              ((res_params.pgreek[2] - res_params.pgreek[1]) / inp.Spot /
                   (in_params.u2[0] - 1) -
               (res_params.pgreek[1] - res_params.pgreek[0]) / inp.Spot /
                   (1 - (1 / in_params.u2[0])));
  res.theta =
      (res_params.vals[0] - res_params.pgreek[3]) / 4 / inp.T * inp.nSteps;
  res.rho = (res_params.vals[1] - res.value) * 10000;
  res.vega = (res_params.vals[2] - res.value) * 10000;
  return res;
}

// Perform CRR solving using the CPU and compare FPGA resutls with CPU results
// to test correctness.
void test_correctness(int k, int n_crrs, bool &pass, input_data &inp,
                      crr_in_params &vals, crr_res &fpga_res) {
  if (k == 0) {
    std::cout << std::endl
              << "============= Correctness Test =============" << std::endl;
    std::cout << "Running analytical correctness checks..." << std::endl;
  }

  // threshold = control the machting decimal digits.
  // This crr benchmark ensure 4 decimal digits matching between FPGA and CPU.
  float threshold = 0.00001;
  int i, j, q;
  double x;
  int nSteps = vals.nSteps;
  int m = nSteps + 2;
  vector<double> pvalue(MAX_N_STEPS + 3);
  vector<double> pvalue_1(MAX_N_STEPS + 1);
  vector<double> pvalue_2(MAX_N_STEPS + 1);
  vector<double> pgreek(5);
  crr_res_params cpu_res_params;
  crr_res cpu_res;

  x = vals.umin[0];
  for (i = 0; i <= m; i++, x *= vals.u2[0]) {
    pvalue[i] = cl::sycl::fmax(inp.CP * (x - inp.Strike), 0.0);
  }

  for (i = m - 1; i >= 0; i--) {
    vals.umin[0] *= vals.u[0];
    x = vals.umin[0];
    for (j = 0; j <= i; j++, x *= vals.u2[0]) {
      pvalue[j] =
          cl::sycl::fmax(vals.c1[0] * pvalue[j] + vals.c2[0] * pvalue[j + 1],
                         inp.CP * (x - inp.Strike));
    }
    if (i == 4) {
      pgreek[4] = pvalue[2];
    }
    if (i == 2) {
      for (q = 0; q <= 2; q++) {
        pgreek[q + 1] = pvalue[q];
      }
    }
  }
  cpu_res_params.vals[0] = pvalue[0];

  x = vals.umin[1];
  for (i = 0; i <= nSteps; i++, x *= vals.u2[1]) {
    pvalue_1[i] = cl::sycl::fmax(inp.CP * (x - inp.Strike), 0.0);
  }

  for (i = nSteps - 1; i >= 0; i--) {
    vals.umin[1] *= vals.u[1];
    x = vals.umin[1];

    for (j = 0; j <= i; j++, x *= vals.u2[1]) {
      pvalue_1[j] = cl::sycl::fmax(vals.c1[1] * pvalue_1[j] +
                                       vals.c2[1] * pvalue_1[j + 1],
                                   inp.CP * (x - inp.Strike));
    }
  }
  cpu_res_params.vals[1] = pvalue_1[0];

  x = vals.umin[2];
  for (i = 0; i <= nSteps; i++, x *= vals.u2[2]) {
    pvalue_2[i] = cl::sycl::fmax(inp.CP * (x - inp.Strike), 0.0);
  }

  for (i = nSteps - 1; i >= 0; i--) {
    vals.umin[2] *= vals.u[2];
    x = vals.umin[2];
    for (j = 0; j <= i; j++, x *= vals.u2[2]) {
      pvalue_2[j] = cl::sycl::fmax(vals.c1[2] * pvalue_2[j] +
                                       vals.c2[2] * pvalue_2[j + 1],
                                   inp.CP * (x - inp.Strike));
    }
  }
  cpu_res_params.vals[2] = pvalue_2[0];
  pgreek[0] = 0;

  for (i = 1; i < 5; ++i) {
    cpu_res_params.pgreek[i - 1] = pgreek[i];
  }

  cpu_res = postprocess_data(inp, vals, cpu_res_params);

  if (abs(cpu_res.value - fpga_res.value) > threshold) {
    pass = false;
    std::cout << "fpga_res.value " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.value << std::endl;
    std::cout << "cpu_res.value " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.value << std::endl;
    std::cout << "Mismatch detected for value of crr " << k << std::endl;
  }
  if (abs(cpu_res.delta - fpga_res.delta) > threshold) {
    pass = false;
    std::cout << "fpga_res.delta " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.delta << std::endl;
    std::cout << "cpu_res.delta " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.delta << std::endl;
    std::cout << "Mismatch detected for value of crr " << k << std::endl;
  }
  if (abs(cpu_res.gamma - fpga_res.gamma) > threshold) {
    pass = false;
    std::cout << "fpga_res.gamma " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.gamma << std::endl;
    std::cout << "cpu_res.gamma " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.gamma << std::endl;
    std::cout << "Mismatch detected for value of crr " << k << std::endl;
  }
  if (abs(cpu_res.vega - fpga_res.vega) > threshold) {
    pass = false;
    std::cout << "fpga_res.vega " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.vega << std::endl;
    std::cout << "cpu_res.vega " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.vega << std::endl;
    std::cout << "Mismatch detected for value of crr " << k << std::endl;
  }
  if (abs(cpu_res.theta - fpga_res.theta) > threshold) {
    pass = false;
    std::cout << "fpga_res.theta " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.theta << std::endl;
    std::cout << "cpu_res.theta " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.theta << std::endl;
    std::cout << "Mismatch detected for value of crr " << k << std::endl;
  }
  if (abs(cpu_res.rho - fpga_res.rho) > threshold) {
    pass = false;
    std::cout << "fpga_res.rho " << k << " = " << std::fixed
              << std::setprecision(20) << fpga_res.rho << std::endl;
    std::cout << "cpu_res.rho " << k << " = " << std::fixed
              << std::setprecision(20) << cpu_res.rho << std::endl;
    std::cout << "Mismatch detected for value of crr " << k << std::endl;
  }

  if (k == n_crrs - 1) {
    std::cout << "CPU-FPGA Equivalence: " << (pass ? "PASS" : "FAIL")
              << std::endl;
  }
}

// Test throughput.
void test_throughput(double &time, const int &n_crrs) {
  std::cout << std::endl
            << "============= Throughput Test =============" << std::endl;

  std::cout << "   Avg throughput:   " << std::fixed << std::setprecision(1)
            << (n_crrs / time) << " assets/s" << std::endl;
}

// Tempate for loop unrolling.
template <int Begin, int End> struct unroller {
  template <typename Action> static void step(const Action &action) {
    action(Begin);
    unroller<Begin + 1, End>::step(action);
  }
};

template <int End> struct unroller<End, End> {
  template <typename Action> static void step(const Action &action) {}
};

// Main function for CRR running on FPGA. It calcutles the option prices and
// parameters for post-calculte Greeks.
func_params crr_main_func(double nSteps, double u, double u2, double c1,
                          double c2, double umin, double param_1,
                          double param_2) {
  [[intelfpga::numbanks(SPATIAL_UNROLL),
    intelfpga::singlepump]] double init_optVal[MAX_N_STEPS + 3];
  [[intelfpga::numbanks(SPATIAL_UNROLL),
    intelfpga::singlepump]] double optVal[MAX_N_STEPS + 3];
  double final_val;
  double bottom_of_tree;
  func_params params;
  // L4:
  // Initialization -- calculte the last level of the binomial tree.
  for (int i = 0; i <= nSteps; i++) {
    double val =
        cl::sycl::fmax(param_1 * cl::sycl::pow(u2, (double)i) - param_2, 0.0);
    init_optVal[i] = val;
    if (i == 0) {
      final_val = val;
      bottom_of_tree = val;
    }
  }
  // L5:
  // Update optVal[] -- calculate each level of the binomial tree.
  // reg[] helps to achieve updating "SPATIAL_UNROLL" elements in optVal[]
  // simultaneously.
  for (int i = 1; i <= nSteps; i++) {
    const int spatial_unroll = SPATIAL_UNROLL;
    double next_reg0 = bottom_of_tree;
    double val_1;
    double val_2;
    double pre_param = param_1 * cl::sycl::pow(u, (double)i);

    // L6:
    // Calculate all the elements in optVal[] -- all the tree nodes for one
    // level of the tree It is safe to use ivdep as the only access that
    // disobeys it is captured through a data dependency (reg[0])
    [[intelfpga::ivdep]] for (int j = 0; j <= nSteps - i;
                              j = j + spatial_unroll) {
      double reg[spatial_unroll + 1];
      reg[0] = next_reg0;

      unroller<1, spatial_unroll + 1>::step([&](int m) {
        if (i == 1)
          reg[m] = init_optVal[j / spatial_unroll * spatial_unroll + m];
        if (i > 1)
          reg[m] = optVal[j / spatial_unroll * spatial_unroll + m];
      });

      next_reg0 = reg[spatial_unroll];

      unroller<0, spatial_unroll>::step([&](int m) {
        double tmp = cl::sycl::fmax(
            c1 * reg[m] + c2 * reg[m + 1],
            pre_param * cl::sycl::pow(u2, (double)(j + m)) - param_2);

        // hack: Add reg[spatial_unroll] >= 0 as an dummy condition on the
        // store, so that the store to optVal depends on the previous load.
        // Without this dummy dependence, the scheduler attempts to schedule the
        // store too early in the pipeline, which hurts II. This is a
        // performance bug in the compiler that is expected to be fixed in a
        // future release.
        if (m != 0 || reg[spatial_unroll] >= 0) {
          optVal[j + m] = tmp;
        }
        if (j + m == 0) {
          final_val = tmp;
          bottom_of_tree = tmp;
        }
        if (j + m == 1) {
          val_1 = tmp;
        }
        if (j + m == 2) {
          val_2 = tmp;
        }
      });
    }

    if (i == nSteps - 4) {
      params.pgreek[3] = val_2;
    }
    if (i == nSteps - 2) {
      params.pgreek[0] = final_val;
      params.pgreek[1] = val_1;
      params.pgreek[2] = val_2;
    }
  }
  params.val = final_val;
  return params;
}

class CRRSolver;
double sycl_device(vector<crr_in_params> &vals,
                   vector<crr_res_params> &res_params,
                   cl::sycl::queue &deviceQueue) {

  high_resolution_clock::time_point start_time = high_resolution_clock::now();
  {
    cl::sycl::buffer<crr_in_params, 1> inputParams(vals.data(), vals.size());
    cl::sycl::buffer<crr_res_params, 1> resParams(res_params.data(),
                                                  res_params.size());

    int n_crr = vals.size();
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto accessorVals =
          inputParams.template get_access<cl::sycl::access::mode::read>(cgh);
      auto accessorRes =
          resParams.template get_access<cl::sycl::access::mode::write>(cgh);

      cgh.single_task<CRRSolver>([=]() {
        // L1:
        // n_crr is the number of CRRs. Each iteration of this loop implement
        // one CRR with Greeks.
        [[intelfpga::ivdep]] for (int i = 0; i < n_crr; i++) {

          // L2: one CRR with Greeks problem will run crr_main_func three times.
          [[intelfpga::ivdep]] for (int j = 0; j < 3; ++j) {
            double iteration = accessorVals[i].nSteps + (j == 0 ? 2 : 0);

            func_params params;
            params = crr_main_func(
                iteration, accessorVals[i].u[j], accessorVals[i].u2[j],
                accessorVals[i].c1[j], accessorVals[i].c2[j],
                accessorVals[i].umin[j], accessorVals[i].param_1[j],
                accessorVals[i].param_2);

            accessorRes[i].vals[j] = params.val;

            // L3: Save parameters for post-calculate fives Greeks.
            for (int k = 0; k < 4; ++k) {
              if (j == 0) {
                accessorRes[i].pgreek[k] = params.pgreek[k];
              }
            }
          }
        }
      });
    });
  }
  high_resolution_clock::time_point end_time = high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;

  res_params.assign(res_params.begin(), res_params.end());
  deviceQueue.throw_asynchronous();
  return diff.count();
}

int main(int argc, char *argv[]) {

  std::string infilename = "";
  std::string outfilename = "";

  char str_buffer[MAX_STRING_LEN] = {0};
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      std::string sarg(argv[i]);

      findGetArgString(sarg, "-o=", str_buffer, MAX_STRING_LEN);
      findGetArgString(sarg, "--output-file=", str_buffer, MAX_STRING_LEN);
    } else {
      infilename = std::string(argv[i]);
    }
  }

  try {

#if defined(FPGA_EMULATOR)
    cl::sycl::intel::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
    cl::sycl::host_selector device_selector;
#else
    cl::sycl::intel::fpga_selector device_selector;
#endif

    cl::sycl::queue deviceQueue(device_selector);

    std::cout << "Running on device:  "
              << deviceQueue.get_device()
                     .get_info<cl::sycl::info::device::name>()
                     .c_str()
              << std::endl;

    cl::sycl::device device = deviceQueue.get_device();
    std::cout << "Device name: "
              << device.get_info<cl::sycl::info::device::name>().c_str()
              << std::endl
              << std::endl
              << std::endl;

    vector<input_data> inp;

    // Get input file name, if users don't have their test input file, this
    // design will use the default input file under ../data/ordered_inputs.csv
    if (infilename == "") {
      infilename = INPUT_FILE;
    }
    ifstream inputFile(infilename);

    if (!inputFile.is_open()) {
      std::cerr << "Input file doesn't exist " << std::endl;
      return 1;
    }

    // Check input file format
    string filename = infilename;
    std::size_t found = filename.find_last_of(".");
    if (!(filename.substr(found + 1).compare("csv") == 0)) {
      std::cerr << "Input file format only support .csv" << std::endl;
      return 1;
    }

    // Get output file name, if users don't define output file name, the default
    // output file is ../data/ordered_outputs.csv
    outfilename = OUTPUT_FILE;
    if (strlen(str_buffer)) {
      outfilename = std::string(str_buffer);
    }

    // Check output file format
    filename = outfilename;
    found = filename.find_last_of(".");
    if (!(filename.substr(found + 1).compare("csv") == 0)) {
      std::cerr << "Output file format only support .csv" << std::endl;
      return 1;
    }

    // Read inputs data from input file
    ReadInputFromFile(inputFile, inp);

// Get the number of data from the input file
#if defined(FPGA_EMULATOR) || defined(CPU_HOST)
    const int n_crrs = 1;
#else
    const int n_crrs = inp.size();
#endif

    vector<crr_in_params> in_params(n_crrs);
    vector<crr_res_params> res_params(n_crrs);
    vector<crr_res_params> res_params_dummy(n_crrs);
    vector<crr_res> result(n_crrs);

    for (int i = 0; i < n_crrs; ++i) {
      in_params[i] = prepare_data(inp[i]);
    }

    // warmup run - use this run to warmup accelerator
    sycl_device(in_params, res_params_dummy, deviceQueue);
    // Timed run - profile performance
    double time = sycl_device(in_params, res_params, deviceQueue);
    bool pass = true;

    for (int i = 0; i < n_crrs; ++i) {
      result[i] = postprocess_data(inp[i], in_params[i], res_params[i]);
      test_correctness(i, n_crrs, pass, inp[i], in_params[i], result[i]);
    }

    // Write outputs data to output file
    ofstream outputFile(outfilename);

    WriteOutputToFile(outputFile, result);

    // Test throughput
    test_throughput(time, n_crrs);

  } catch (cl::sycl::exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cout << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cout << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;
    std::cout
        << "   If you are targeting a CPU host device, compile with -DCPU_HOST"
        << std::endl;
    return 1;
  }
  return 0;
}
