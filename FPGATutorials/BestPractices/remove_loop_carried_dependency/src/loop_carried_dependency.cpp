#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>

using namespace cl::sycl;
using namespace std;

class UnOptKernel;
event Unoptimized(queue &q, const vector<double> &A, const vector<double> &B,
                  double &result, size_t N) {
  buffer b_a(A);
  buffer b_b(B);
  buffer b_result(&result, range(1));

  auto e = q.submit([&](handler &h) {
    auto a = b_a.get_access<access::mode::read>(h);
    auto b = b_b.get_access<access::mode::read>(h);
    auto result = b_result.get_access<access::mode::write>(h);

    h.single_task<UnOptKernel>([=]() {
      double sum = 0;
      for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
          sum += a[i * N + j];
        }
        sum += b[i];
      }
      result[0] = sum;
    });
  });
  return e;
}

class OptKernel;
event Optimized(queue &q, const vector<double> &A, const vector<double> &B,
                double &result, size_t N) {
  buffer b_a(A);
  buffer b_b(B);
  buffer b_result(&result, range(1));

  auto e = q.submit([&](handler &h) {
    auto a = b_a.get_access<access::mode::read>(h);
    auto b = b_b.get_access<access::mode::read>(h);
    auto result = b_result.get_access<access::mode::write>(h);

    h.single_task<OptKernel>([=]() {
      double sum = 0;

      for (size_t i = 0; i < N; i++) {
        // Step 1: Definition
        double sum2 = 0;

        // Step 2: Accumulation of array A values for one outer loop iteration
        for (size_t j = 0; j < N; j++) {
          sum2 += a[i * N + j];
        }

        // Step 3: Addition of array B value for an outer loop iteration
        sum += sum2;
        sum += b[i];
      }

      result[0] = sum;
    });
  });
  return e;
}

void PrintTime(const event &e, queue &q, const char *kind) {
  q.wait_and_throw();
  auto start_k = e.get_profiling_info<info::event_profiling::command_start>();
  auto end_k = e.get_profiling_info<info::event_profiling::command_end>();
  auto kernel_time = (double)(end_k - start_k) * 1e-6f;

  cout << "Run: " << kind << ":\n";
  cout << "kernel time : " << kernel_time << " ms\n";
}

int main(int argc, char *argv[]) {
  size_t N = 16000;

  if (argc > 1) {
    string option(argv[1]);
    if (option == "-h" || option == "--help") {
      cout << "Usage: <executable> <data size>\n";
      return 1;
    } else {
      N = stoi(option);
    }
  }
  // Cap the value of N.
  N = std::max(std::min((size_t)N, (size_t)16000), (size_t)100);
  cout << "Number of elements: " << N << '\n';

  // Initialize queue with device selector and enabling profiling
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  intel::fpga_emulator_selector selector;
  cout << "\nEmulator output does not demonstrate true hardware "
          "performance. The design may need to run on actual hardware "
          "to observe the performance benefit of the optimization "
          "exemplified in this tutorial.\n\n";
#else
  intel::fpga_selector selector;
#endif
  // Create a profiling queue
  queue q(selector, property_list{property::queue::enable_profiling()});

  vector<double> A(N * N);
  vector<double> B(N);

  double answer = 0;

  // initialize data and compute testbench reference
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i + j;
      answer += i + j;
    }
    B[i] = i;
    answer += i;
  }

  // compute result in component
  double unopt_sum = -1, opt_sum = -1;
  PrintTime(Unoptimized(q, A, B, unopt_sum, N), q, "Unoptimized");
  PrintTime(Optimized(q, A, B, opt_sum, N), q, "Optimized");

  // error check
  bool failed = false;
  if (unopt_sum != answer) {
    cout << "Unoptimized: expected: " << answer << ", result: " << unopt_sum
         << '\n';
    failed = true;
  }
  if (opt_sum != answer) {
    cout << "Optimized: expected: " << answer << ", result: " << opt_sum
         << '\n';
    failed = true;
  }
  if (failed) {
    cout << "FAILED\n";
    return 1;
  }
  cout << "PASSED\n";
  return 0;
}
