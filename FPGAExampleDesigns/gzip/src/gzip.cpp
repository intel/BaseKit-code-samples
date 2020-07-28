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

#include <fcntl.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <chrono>
#include <fstream>
#include <string>

#include "CompareGzip.hpp"
#include "WriteGzip.hpp"
#include "crc32.hpp"
#include "gzipkernel.hpp"
#include "kernels.hpp"

using namespace cl::sycl;

// The minimum file size of a file to be compressed.
// Any filesize less than this results in an error.
constexpr int minimum_filesize = kVec + 1;

bool help = false;

int CompressFile(queue &q, std::string &input_file, std::string &output_file,
                 int iterations, bool report);

void Help(void) {
  // Command line arguments.
  // gzip [options] filetozip [options]
  // -h,--help                    : help

  // future options?
  // -p,performance : output perf metrics
  // -m,maxmapping=#  : maximum mapping size

  std::cout << "gzip filename [options]\n";
  std::cout << "  -h,--help                                : this help text\n";
  std::cout
      << "  -o=<filename>,--output-file=<filename>   : specify output file\n";
}

bool FindGetArg(std::string &arg, const char *str, int defaultval, int *val) {
  std::size_t found = arg.find(str, 0, strlen(str));
  if (found != std::string::npos) {
    int value = atoi(&arg.c_str()[strlen(str)]);
    *val = value;
    return true;
  }
  return false;
}

constexpr int kMaxStringLen = 40;

bool FindGetArgString(std::string &arg, const char *str, char *str_value,
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

auto exception_handler = [](exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n";
      std::terminate();
    }
  }
};

int main(int argc, char **argv) {
  std::string infilename = "";
  std::string outfilename = "";

  char str_buffer[kMaxStringLen] = {0};

  // Check the number of arguments specified
  if (argc != 3) {
    std::cerr << "Incorrect number of arguments. Correct usage: " << argv[0]
              << " <input-file> -o=<output-file>\n";
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      std::string sarg(argv[i]);
      if (std::string(argv[i]) == "-h") {
        help = true;
      }
      if (std::string(argv[i]) == "--help") {
        help = true;
      }

      FindGetArgString(sarg, "-o=", str_buffer, kMaxStringLen);
      FindGetArgString(sarg, "--output-file=", str_buffer, kMaxStringLen);
    } else {
      infilename = std::string(argv[i]);
    }
  }

  if (help) {
    Help();
    return 1;
  }

  try {
#ifdef FPGA_EMULATOR
    intel::fpga_emulator_selector device_selector;
#else
    intel::fpga_selector device_selector;
#endif
    queue q(device_selector, exception_handler);

    std::cout << "Running on device:  "
              << q.get_device().get_info<info::device::name>().c_str() << "\n";

    if (infilename == "") {
      std::cout << "Must specify a filename to compress\n\n";
      Help();
      return 1;
    }

    // next, check valid and acceptable parameter ranges.
    // if output filename not set, use the default
    // name, else use the name specified by the user
    outfilename = std::string(infilename) + ".gz";
    if (strlen(str_buffer)) {
      outfilename = std::string(str_buffer);
    }

#ifdef FPGA_EMULATOR
    CompressFile(q, infilename, outfilename, 1, true);
#else
    // warmup run - use this run to warmup accelerator
    CompressFile(q, infilename, outfilename, 1, false);
    // profile performance
    CompressFile(q, infilename, outfilename, 100, true);
#endif
    q.throw_asynchronous();
  } catch (exception const &e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::cout << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly\n";
    std::cout << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR\n";
    return 1;
  }
  return 0;
}

struct KernelInfo {
  buffer<struct GzipOutInfo, 1> *gzip_out_buf;
  buffer<unsigned, 1> *current_crc;
  buffer<char, 1> *pobuf;
  buffer<char, 1> *pibuf;
  char *pobuf_decompress;

  uint32_t buffer_crc[kMinBufferSize];
  uint32_t refcrc;

  const char *pref_buffer;
  char *poutput_buffer;
  size_t file_size;
  struct GzipOutInfo out_info[kMinBufferSize];
  int iteration;
  bool last_block;
};

// returns 0 on success, otherwise a non-zero failure code.
int CompressFile(queue &q, std::string &input_file, std::string &output_file,
                 int iterations, bool report) {
  size_t isz;
  char *pinbuf;

  std::ifstream file(input_file,
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    isz = file.tellg();
    pinbuf = new char[isz];
    file.seekg(0, std::ios::beg);
    file.read(pinbuf, isz);
    file.close();
  } else {
    std::cout << "Error: cannot read specified input file\n";
    return 1;
  }

  if (isz < minimum_filesize) {
    std::cout << "Minimum filesize for compression is " << minimum_filesize
              << "\n";
    return 1;
  }

  int buffers_count = iterations;

  // Array of kernel info structures...
  struct KernelInfo *kinfo =
      (struct KernelInfo *)malloc(sizeof(struct KernelInfo) * buffers_count);

  if (kinfo == NULL) {
    std::cout << "Cannot allocate kernel info buffer.\n";
    return 1;
  }

  for (int i = 0; i < buffers_count; i++) {
    kinfo[i].file_size = isz;
    // Allocating slightly larger buffers (+ 16 * kVec) to account for
    // granularity of kernel writes
    int outputSize = kinfo[i].file_size + 16 * kVec < kMinBufferSize
                         ? kMinBufferSize
                         : kinfo[i].file_size + 16 * kVec;

    kinfo[i].poutput_buffer = (char *)malloc(outputSize);
    if (kinfo[i].poutput_buffer == NULL) {
      std::cout << "Cannot allocate output buffer.\n";
      free(kinfo);
      return 1;
    }
    // zero pages to fully allocate them
    memset(kinfo[i].poutput_buffer, 0, outputSize);

    kinfo[i].last_block = true;
    kinfo[i].iteration = i;
    kinfo[i].pref_buffer = pinbuf;

    kinfo[i].gzip_out_buf =
        i >= 3 ? kinfo[i - 3].gzip_out_buf
               : new buffer<struct GzipOutInfo, 1>(kMinBufferSize);
    kinfo[i].current_crc = i >= 3 ? kinfo[i - 3].current_crc
                                  : new buffer<unsigned, 1>(kMinBufferSize);
    kinfo[i].pibuf =
        i >= 3 ? kinfo[i - 3].pibuf : new buffer<char, 1>(kinfo[i].file_size);
    kinfo[i].pobuf =
        i >= 3 ? kinfo[i - 3].pobuf : new buffer<char, 1>(outputSize);
    kinfo[i].pobuf_decompress = (char *)malloc(kinfo[i].file_size);
  }

  event event_output[buffers_count];
  event event_crc[buffers_count];
  event event_size[buffers_count];

  std::chrono::high_resolution_clock::time_point start_time0 =
      std::chrono::high_resolution_clock::now();

  for (int index = 0; index < buffers_count; index++) {
    q.submit([&](handler &h) {
      auto in_data =
          kinfo[index].pibuf->get_access<access::mode::discard_write>(h);
      h.copy(kinfo[index].pref_buffer, in_data);
    });

    SubmitGzipTasks(q, kinfo[index].file_size, kinfo[index].pibuf,
                    kinfo[index].pobuf, kinfo[index].gzip_out_buf,
                    kinfo[index].current_crc, kinfo[index].last_block);

    event_output[index] = q.submit([&](handler &h) {
      auto out_data = kinfo[index].pobuf->get_access<access::mode::read>(h);
      h.copy(out_data, kinfo[index].poutput_buffer);
    });

    event_size[index] = q.submit([&](handler &h) {
      auto out_data =
          kinfo[index].gzip_out_buf->get_access<access::mode::read>(h);
      h.copy(out_data, kinfo[index].out_info);
    });

    event_crc[index] = q.submit([&](handler &h) {
      auto out_data =
          kinfo[index].current_crc->get_access<access::mode::read>(h);
      h.copy(out_data, kinfo[index].buffer_crc);
    });
  }

  size_t compressed_sz = 0;
  for (int index = 0; index < buffers_count; index++) {
    event_output[index].wait();
    event_size[index].wait();
    event_crc[index].wait();
    if (kinfo[index].out_info[0].compression_sz > kinfo[index].file_size) {
      std::cerr << "Unsupported: compressed file larger than input file( "
                << kinfo[index].out_info[0].compression_sz << " )\n";
      return 1;
    }
    kinfo[index].buffer_crc[0] =
        Crc32(kinfo[index].pref_buffer, kinfo[index].file_size,
              kinfo[index].buffer_crc[0]);
    compressed_sz += kinfo[index].out_info[0].compression_sz;
  }

  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff_total = (end_time - start_time0);

#ifndef FPGA_EMULATOR
  double gbps = iterations * isz / (double)diff_total.count() / 1000000000.0;
#endif

  // delete the file mapping now that all kernels are complete, and we've
  // snapped the time delta
  delete pinbuf;

  // Write the first copy to file
  if (report && WriteBlockGzip(input_file, output_file, kinfo[0].poutput_buffer,
                               kinfo[0].out_info[0].compression_sz,
                               kinfo[0].file_size, kinfo[0].buffer_crc[0])) {
    std::cout << "FAILED\n";
    return 1;
  }

  if (report && CompareGzipFiles(input_file, output_file)) {
    std::cout << "FAILED\n";
    return 1;
  }

  if (report) {
    double compression_ratio =
        (double)((double)compressed_sz / (double)isz / iterations);
#ifndef FPGA_EMULATOR
    std::cout << "Throughput: " << gbps << " GB/s\n";
#endif
    std::cout << "Compression Ratio " << compression_ratio * 100 << "%\n";
  }

  // Cleanup anything that was allocated by this routine.
  for (int i = 0; i < buffers_count; i++) {
    if (i < 3) {
      delete kinfo[i].gzip_out_buf;
      delete kinfo[i].current_crc;
      delete kinfo[i].pibuf;
      delete kinfo[i].pobuf;
    }
    free(kinfo[i].poutput_buffer);
    free(kinfo[i].pobuf_decompress);
  }
  free(kinfo);

  if (report) std::cout << "PASSED\n";
  return 0;
}
