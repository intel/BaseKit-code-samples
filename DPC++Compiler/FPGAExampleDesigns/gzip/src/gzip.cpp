// ==============================================================
// Copyright (C) 2019 Intel Corporation
// 
// SPDX-License-Identifier: MIT
// =============================================================
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
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
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include <chrono>
#include <fstream>
#include <fcntl.h>
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <memory.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/types.h>

#include "crc32.h"
#include "kernels.h"
#include "gzipkernel.h"
#include "CompareGzip.h"
#include "WriteGzip.h"

// The minimum file size of a file to be compressed. 
// Any filesize less than this results in an error.
#define MINIMUM_FILESIZE (VEC+1) 

bool bHelp = false;

int compress_file(
        cl::sycl::queue &device_queue,
        std::string inputfile,
        std::string outputfile,
        int iterations, bool report);

void help(void)
{
    // Command line arguments.
    // gzip [options] filetozip [options] 
    // -h,--help                    : help

    // future options?
    // -p,performance : output perf metrics
    // -m,maxmapping=#  : maximum mapping size

    std::cout << "gzip filename [options]" << std::endl;
    std::cout << "  -h,--help                                : this help text" << std::endl;
    std::cout << "  -o=<filename>,--output-file=<filename>   : specify output file" << std::endl;
}

bool findGetArg(std::string arg, const char *str, int defaultval, int *val)
{
    std::size_t found = arg.find(str, 0, strlen(str));
    if (found != std::string::npos) {
        int value = atoi(&arg.c_str()[strlen(str)]);
        *val = value;
        return true;
    }
    return false;
}

#define MAX_STRING_LEN 40

bool findGetArgString(std::string arg, const char *str, char *str_value, size_t maxchars)
{
    std::size_t found = arg.find(str, 0, strlen(str));
    if (found != std::string::npos) {
        const char *sptr = &arg.c_str()[strlen(str)];
        for (int i=0; i<maxchars-1; i++) {

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

int main(int argc, char **argv)
{
    std::string infilename = "";
    std::string outfilename = "";

    char str_buffer[MAX_STRING_LEN] = {0};
    for (int i=1; i < argc; i++) {
        if (argv[i][0] == '-') {
            std::string sarg(argv[i]);
            if (std::string(argv[i]) == "-h") {
                bHelp = true;
            }
            if (std::string(argv[i]) == "--help") {
                bHelp = true;
            }

            findGetArgString(sarg, "-o=", str_buffer, MAX_STRING_LEN);
            findGetArgString(sarg, "--output-file=", str_buffer, MAX_STRING_LEN);
        } else {
            infilename = std::string(argv[i]);
        }  
    }

    if (bHelp) {
        help();
        return 1;
    }

    try {
#ifdef FPGA_EMULATOR
       cl::sycl::intel::fpga_emulator_selector device_selector;
#else
       cl::sycl::intel::fpga_selector device_selector;
#endif
       cl::sycl::queue device_queue(device_selector, cl::sycl::async_handler{});

       std::cout << "Running on device:  " <<  device_queue.get_device().get_info<cl::sycl::info::device::name>().c_str() << std::endl;

       if (infilename == "") {
           std::cout << "Must specify a filename to compress" << std::endl << std::endl;
           help();
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
       compress_file(device_queue, infilename, outfilename, 1, true);
#else
       // warmup run - use this run to warmup accelerator
       compress_file(device_queue, infilename, outfilename, 1, false);
       // profile performance 
       compress_file(device_queue, infilename, outfilename, 100, true);
#endif
       device_queue.throw_asynchronous();
    } catch (cl::sycl::exception const& e) {
      std::cout << "Caught a synchronous SYCL exception: " << e.what() << std::endl;
      std::cout << "   If you are targeting an FPGA hardware, "
                    "ensure that your system is plugged to an FPGA board that is set up correctly" << std::endl;
      std::cout << "   If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR" << std::endl;
      std::cout << "   If you are targeting a CPU host device, compile with -DCPU_HOST" << std::endl;
      return 1;
    }
    return 0;
}


struct _kernel_info {
    cl::sycl::buffer<struct gzip_out_info_t, 1> *   gzip_out_buf;
    cl::sycl::buffer<unsigned, 1> *   current_crc;
    cl::sycl::buffer<char, 1> *            pobuf;
    cl::sycl::buffer<char, 1> *            pibuf;
    char *                                 pobuf_decompress;

    uint32_t                buffer_crc[16384];
    uint32_t                refcrc;

    const char *   pRefBuffer;
    char *         poutput_buffer;
    size_t                  fileSize;
    struct gzip_out_info_t  out_info[16384];
    int                     iteration;
    bool                    lastblock;
};

// returns 0 on success, otherwise a non-zero failure code.
int compress_file(
        cl::sycl::queue &device_queue,
        std::string inputfile,
        std::string outputfile,
        int iterations,
        bool report ) 
{
    int status = 0;    
    size_t isz;
    char *pinbuf;

    std::ifstream file (inputfile, std::ios::in|std::ios::binary|std::ios::ate);
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

    if ( isz < MINIMUM_FILESIZE ) {
        std::cout << "Minimum filesize for compression is " << MINIMUM_FILESIZE << std::endl;
        return 1;
    }

    int buffers_count = iterations;

    // Array of kernel info structures...
    struct _kernel_info *kinfo = (struct _kernel_info *)
        malloc(sizeof(struct _kernel_info) * buffers_count  );

    if (kinfo == NULL) {
        std::cout << "Cannot allocate kernel info buffer" << std::endl;
        return 1;
    }

    for (int i=0; i<buffers_count; i++) {

        kinfo[i].fileSize = isz;
        // Allocating slightly larger buffers (+ 16 * VEC) to account for granularity of kernel writes
        int outputSize = kinfo[i].fileSize + 16 * VEC < 16384 ? 16384 : kinfo[i].fileSize + 16 * VEC;

        kinfo[i].poutput_buffer = (char *) malloc(outputSize);
        if (kinfo[i].poutput_buffer == NULL) {
            std::cout << "Cannot allocate output buffer" << std::endl;
            free(kinfo);
            return 1;
        }
        // zero pages to fully allocate them
	memset(kinfo[i].poutput_buffer, 0, outputSize);

        kinfo[i].lastblock = true;
        kinfo[i].iteration = i;
        kinfo[i].pRefBuffer = pinbuf;

        kinfo[i].gzip_out_buf = i >= 3 ? kinfo[i - 3].gzip_out_buf : new cl::sycl::buffer<struct gzip_out_info_t,1>(16384);
        kinfo[i].current_crc = i >= 3 ? kinfo[i - 3].current_crc : new cl::sycl::buffer<unsigned,1>(16384);
        kinfo[i].pibuf = i >= 3 ? kinfo[i - 3].pibuf : new cl::sycl::buffer<char, 1>(kinfo[i].fileSize);
        kinfo[i].pobuf = i >= 3 ? kinfo[i - 3].pobuf : new cl::sycl::buffer<char, 1>(outputSize);
        kinfo[i].pobuf_decompress = (char *)malloc(kinfo[i].fileSize);

    }

    cl::sycl::event event_output[buffers_count], event_crc[buffers_count], event_size[buffers_count];
    std::chrono::high_resolution_clock::time_point start_time0  = std::chrono::high_resolution_clock::now();

     for (int index=0; index<buffers_count; index++) {
         device_queue.submit([&](cl::sycl::handler& cgh) {
            auto in_data = kinfo[index].pibuf->template get_access<cl::sycl::access::mode::discard_write>(cgh);
            cgh.copy(kinfo[index].pRefBuffer, in_data);
         });
         submit_gzip_tasks(device_queue,
                           kinfo[index].fileSize,
                           kinfo[index].pibuf,
                           kinfo[index].pobuf,
                           kinfo[index].gzip_out_buf,
                           kinfo[index].current_crc,
                           kinfo[index].lastblock);
         event_output[index] = device_queue.submit([&](cl::sycl::handler& cgh) {
            auto out_data = kinfo[index].pobuf->template get_access<cl::sycl::access::mode::read>(cgh);
            cgh.copy(out_data, kinfo[index].poutput_buffer);
         });
         event_size[index] = device_queue.submit([&](cl::sycl::handler& cgh) {
            auto out_data = kinfo[index].gzip_out_buf->template get_access<cl::sycl::access::mode::read>(cgh);
            cgh.copy(out_data, kinfo[index].out_info);
         });
         event_crc[index] = device_queue.submit([&](cl::sycl::handler& cgh) {
            auto out_data = kinfo[index].current_crc->template get_access<cl::sycl::access::mode::read>(cgh);
            cgh.copy(out_data, kinfo[index].buffer_crc);
         });
    }
    size_t compressed_sz = 0;
    for (int index=0; index<buffers_count; index++) {
        event_output[index].wait(); 
        event_size[index].wait(); 
        event_crc[index].wait(); 
        if (kinfo[index].out_info[0].compression_sz > kinfo[index].fileSize) {
           std::cerr << "Unsupported: compressed file larger than input file( " << kinfo[index].out_info[0].compression_sz << " )\n";
           return 1;
        }
        kinfo[index].buffer_crc[0] = crc32(kinfo[index].pRefBuffer, kinfo[index].fileSize, kinfo[index].buffer_crc[0]);
        compressed_sz += kinfo[index].out_info[0].compression_sz;
    }

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff_total = (end_time - start_time0); 
         
    double gbps = iterations * isz / (double) diff_total.count() / 1000000000.0;
    double compression_ratio = (double)((double)compressed_sz / (double)isz / iterations); 

    // delete the file mapping now that all kernels are complete, and we've snapped the time delta
    delete pinbuf;

    // Write the first copy to file
    if (report && write_block_gzip(inputfile, outputfile, kinfo[0].poutput_buffer, kinfo[0].out_info[0].compression_sz, kinfo[0].fileSize, kinfo[0].buffer_crc[0])) {
       std::cout << "FAILED" << std::endl;
       return 1;
    }

    if (report && compare_gzip_files(inputfile, outputfile)) {
       std::cout << "FAILED" << std::endl;
       return 1;
    }

    if (report) {
#ifndef FPGA_EMULATOR
       std::cout << "Throughput: " << gbps << " GB/s " << std::endl;
#endif
       std::cout << "Compression Ratio " << compression_ratio * 100 << "%" << std::endl;
    }

    // Cleanup anything that was allocated by this routine. 
    for (int i=0; i<buffers_count; i++) {
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

    if (report) std::cout << "PASSED" << std::endl;
    return 0;
}

