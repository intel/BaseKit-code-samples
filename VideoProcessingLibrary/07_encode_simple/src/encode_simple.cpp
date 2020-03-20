/*############################################################################
  # Copyright (C) 2019 Intel Corporation
  #
  # SPDX-License-Identifier: MIT
  ############################################################################*/
/// Demonstration of simple video encode.
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "vpl/vpl.hpp"

#define ALIGN_UP(addr, size) \
  (((addr) + ((size)-1)) & (~((decltype(addr))(size)-1)))

#define PROGRAM_NAME "encode_simple"
const int SUCCESS = 0;
const int FAILURE = -1;

void LogTrace(const char* fmt, ...);
int EncodeFile(const char* input, int width, int height, std::ofstream& dest);
void PrintUsage(FILE* stream);

/// Simple timer that tracks total time elapsed between starts and stops
class Timer {
 public:
  Timer() : elapsed_time_(elapsed_time_.zero()) {}
  void Start() { start_time_ = std::chrono::system_clock::now(); }
  void Stop() {
    stop_time_ = std::chrono::system_clock::now();
    elapsed_time_ += (stop_time_ - start_time_);
  }
  double Elapsed() const { return elapsed_time_.count(); }

 private:
  std::chrono::system_clock::time_point start_time_, stop_time_;
  std::chrono::duration<double> elapsed_time_;
};

/// Program entry point
int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "%s: missing file operand\n", PROGRAM_NAME);
    PrintUsage(stderr);
    return FAILURE;
  }
  if (argc < 3) {
    fprintf(stderr, "%s: missing width operand\n", PROGRAM_NAME);
    PrintUsage(stderr);
    return FAILURE;
  }
  if (argc < 4) {
    fprintf(stderr, "%s: missing height operand\n", PROGRAM_NAME);
    PrintUsage(stderr);
    return FAILURE;
  }

  std::ofstream output_stream("out.h264", std::ios::binary);
  int status = EncodeFile(argv[1], std::stoi(argv[2]), std::stoi(argv[3]),
                          output_stream);
  output_stream.flush();
  return status;
}

////////////////////////////////////////////////////////////////////////////////
/// Main encode function
////////////////////////////////////////////////////////////////////////////////
int EncodeFile(const char* input, int width, int height, std::ofstream& dest) {
  int status = FAILURE;
  LogTrace("Creating H.264 encoder using Gen GPU ");
  vpl::Workstream encoder(VPL_TARGET_DEVICE_GPU_GEN, VPL_WORKSTREAM_ENCODE);
  encoder.SetConfig(VPL_PROP_DST_BITSTREAM_FORMAT, VPL_FOURCC_H264);
  VplVideoSurfaceResolution srcResolution = {0};
  srcResolution.height = height;
  srcResolution.width  = width;
  encoder.SetConfig(VPL_PROP_OUTPUT_RESOLUTION, srcResolution); 
  VplFile* fInput = vplOpenFile(input, "rb");

  size_t frame_count = 0;
  Timer timer;
  int raw_buffer_size = (width * height * 3) / 2;
  std::vector<uint8_t> raw_buffer(raw_buffer_size);
  std::vector<uint8_t> enc_buffer(1024 * 1024 * 80);
  size_t total_encoded_bytes = 0;
  bool encode_done = false;

  // Record format of incomming frames
  vplm_image_info info = {};
  info.width = width;
  info.height = height;
  info.aligned_width = ALIGN_UP(width,16);
  info.aligned_height = ALIGN_UP(height,16);
  info.format = VPLM_PIXEL_FORMAT_NV12;

  LogTrace("Entering main encode loop");
  vplStatus sts;
  while (!encode_done) {
    size_t encoded_bytes = 0;

    switch (encoder.GetState()) {
      case VPL_STATE_READ_INPUT:
        vplm_mem* raw_image;
        vplm_create_cpu_image(&info, &raw_image);
        sts = vplReadData(fInput, raw_image);
        if (sts == 0) {
          frame_count++;
          fprintf(stderr, "Frame: %zu\r", frame_count);
          timer.Start();
          vplm_ref(raw_image);
          encoded_bytes = encoder.EncodeFrame(raw_image, &enc_buffer[0]);
          timer.Stop();
        } else {
          timer.Start();
          encoded_bytes = encoder.EncodeFrame(nullptr, &enc_buffer[0]);
          timer.Stop();
          encode_done = encoded_bytes == 0;
          if (encode_done) status = SUCCESS;
        }
        break;

      case VPL_STATE_END_OF_OPERATION:
        // The encoder has completed operation, and has no frames left to give.
        LogTrace("Encode complete");
        encode_done = true;
        status = SUCCESS;
        break;

      case VPL_STATE_ERROR:
        LogTrace("Error during encode. Exiting.");
        encode_done = true;
        status = FAILURE;
        break;
    }

    if (encoded_bytes) {
      total_encoded_bytes += encoded_bytes;
      dest.write((char*)&enc_buffer[0], encoded_bytes);
    }
  }

  LogTrace("Frames encoded   : %zu", frame_count);
  LogTrace("Frames per second: %02f", frame_count / timer.Elapsed());
  double ratio =
      (double)total_encoded_bytes / (double)(frame_count * raw_buffer_size);
  LogTrace("Compression Ratio   : %f", ratio);

  return status;
}

/// Print command line usage
void PrintUsage(FILE* stream) {
  fprintf(stream, "Usage: %s FILE WIDTH HEIGHT\n\n", PROGRAM_NAME);
  fprintf(stream,
          "Encode FILE using Intel(R) oneAPI Video Processing Library.\n\n"
          "FILE must be in NV12 format\n\n"
          "Example:\n"
          "  %s %s\n",
          PROGRAM_NAME, "content/cars_1280x720.nv12 1280 720");
}

/// Print message to stderr
void LogTrace(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}
