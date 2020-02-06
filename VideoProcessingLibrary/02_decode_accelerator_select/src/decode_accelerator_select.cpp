/*############################################################################
  # Copyright (C) 2019 Intel Corporation
  #
  # SPDX-License-Identifier: MIT
  ############################################################################*/
/// Demonstration of video decode accelerator selection.
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "vpl/vpl.hpp"

#define PROGRAM_NAME "decode_accelerator_select"
const int SUCCESS = 0;
const int FAILURE = -1;

/// Number of bytes to read from file at a time and pass to decoder
const size_t kChunkSize = 1024 * 1024;

void LogTrace(const char* fmt, ...);
void DisplayFrame(vplm_mem* image);
int DecodeAndRenderFile(std::ifstream& stream);
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
  std::ifstream input_stream(argv[1], std::ios::binary);

  if (!input_stream.is_open()) {
    fprintf(stderr, "%s: could not open input file '%s'\n", PROGRAM_NAME,
            argv[1]);
    return FAILURE;
  }
  int status = DecodeAndRenderFile(input_stream);
  return status;
}

////////////////////////////////////////////////////////////////////////////////
/// Main decode and render function
////////////////////////////////////////////////////////////////////////////////
int DecodeAndRenderFile(std::ifstream& stream) {
  int status = FAILURE;
  LogTrace("Creating H.264 decoder using CPU device");
  vpl::Workstream decoder(VPL_TARGET_DEVICE_CPU, VPL_WORKSTREAM_DECODE);
  decoder.SetConfig(VPL_PROP_SRC_BITSTREAM_FORMAT, VPL_FOURCC_H264);

  size_t frame_count = 0;
  Timer timer;

  std::vector<uint8_t> buffer(kChunkSize);
  bool decode_done = false;

  LogTrace("Entering main decode loop");
  while (!decode_done) {
    vplm_mem* image = nullptr;

    switch (decoder.GetState()) {
      case VPL_STATE_READ_INPUT:
        // The decoder can accept more data, read it from file and pass it in.
        stream.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        timer.Start();
        image = decoder.DecodeFrame(buffer.data(), stream.gcount());
        timer.Stop();
        break;

      case VPL_STATE_INPUT_BUFFER_FULL:
        // The decoder cannot accept more data, call DecodeFrame to drain.
        timer.Start();
        image = decoder.DecodeFrame(nullptr, 0);
        timer.Stop();
        break;

      case VPL_STATE_END_OF_OPERATION:
        // The decoder has completed operation, and has no frames left to give.
        LogTrace("Decode complete");
        decode_done = true;
        status = SUCCESS;
        break;

      case VPL_STATE_ERROR:
        LogTrace("Error during decode. Exiting.");
        decode_done = true;
        status = FAILURE;
        break;
    }

    if (image) {
      // DecodeFrame returned a frame, use it.
      frame_count++;
      fprintf(stderr, "Frame: %zu\r", frame_count);
      DisplayFrame(image);
      // Release the reference to the frame, so the memory can be reclaimed
      vplm_unref(image);
    }
  }

  LogTrace("Frames decoded   : %zu", frame_count);
  LogTrace("Frames per second: %02f", frame_count / timer.Elapsed());

  return status;
}

/// Print command line usage
void PrintUsage(FILE* stream) {
  fprintf(stream, "Usage: %s FILE\n\n", PROGRAM_NAME);
  fprintf(stream,
          "Decode FILE using Intel(R) oneAPI Video Processing Library CPU "
          "Accelerator.\n\n"
          "FILE must be in H264 format\n\n"
          "Example:\n"
          "  %s %s\n",
          PROGRAM_NAME, "content/cars_1280x720.h264");
}

/// Render frame to display
void DisplayFrame(vplm_mem* image) {
  vplm_cpu_image handle = {0};
  vplm_image_info desc;
  unsigned char* data;
  bool have_display = true;
  static bool first_call = true;
#ifdef __linux__
  const char* display = getenv("DISPLAY");
  if (!display) {
    if (first_call) LogTrace("Display unavailable, continuing without...");
    have_display = false;
  }
#endif
  // Read image description (width, height, etc) from vpl memory
  vplm_get_image_info(image, &desc);
  // Access data in read mode
  vplm_status err = vplm_map_image(image, VPLM_ACCESS_MODE_READ, &handle);
  // Need to rearrange data because of stride size
  data = new unsigned char[desc.height * 3 / 2 * desc.width];
  cv::Mat img_i420, img_bgra;
  unsigned char* ydata = data;
  unsigned char* udata = data + (desc.width * desc.height);
  unsigned char* vdata = udata + ((desc.width / 2) * (desc.height / 2));
  size_t pitch0 = handle.planes[0].stride;
  size_t pitch1 = handle.planes[1].stride;
  size_t pitch2 = handle.planes[2].stride;
  for (size_t y = 0; y < desc.height; y++) {
    memcpy(ydata + (desc.width * y), handle.planes[0].data + (pitch0 * y),
           desc.width);
  }
  for (size_t y = 0; y < desc.height / 2; y++) {
    memcpy(udata + ((desc.width / 2) * y), handle.planes[1].data + (pitch1 * y),
           (desc.width / 2));
  }
  for (size_t y = 0; y < desc.height / 2; y++) {
    memcpy(vdata + ((desc.width / 2) * y), handle.planes[2].data + (pitch2 * y),
           (desc.width / 2));
  }

  img_i420 = cv::Mat(desc.height * 3 / 2, desc.width, CV_8UC1, data);
  // Convert NV12 to BGRA format for displaying with OpenCV
  cv::cvtColor(img_i420, img_bgra, cv::COLOR_YUV2BGRA_I420);
  if (have_display) cv::imshow("Display decoded output", img_bgra);
  cv::waitKey(24);
  vplm_unmap_image(&handle);
  delete data;
  return;
}

/// Print message to stderr
void LogTrace(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}
