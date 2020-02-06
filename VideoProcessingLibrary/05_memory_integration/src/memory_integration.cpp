/*############################################################################
  # Copyright (C) 2019 Intel Corporation
  #
  # SPDX-License-Identifier: MIT
  ############################################################################*/
/// Demonstration of integration in video pipeline
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <vplmemory/vplm++.h>
#include <opencv2/opencv.hpp>
#include "vpl/vpl.hpp"

extern "C" {
#include <libavformat/avformat.h>
}

#define PROGRAM_NAME "memory_integration"
const int SUCCESS = 0;
const int FAILURE = -1;

void LogTrace(const char* fmt, ...);
void InvertImage(vplm_mem* image);
void DisplayFrame(vplm_mem* image);
int DecodeAndRenderFile(const char* filename);
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
  FILE* input_stream = fopen(argv[1], "rb");
  if (!input_stream) {
    fprintf(stderr, "%s: could not open input file '%s'\n", PROGRAM_NAME,
            argv[1]);
    return FAILURE;
  }
  fclose(input_stream);
  int status = DecodeAndRenderFile(argv[1]);
  return status;
}

////////////////////////////////////////////////////////////////////////////////
/// Main decode and render function
////////////////////////////////////////////////////////////////////////////////
int DecodeAndRenderFile(const char* filename) {
  int status = FAILURE;
  int avsts;
  LogTrace("Creating H.264 decoder using default device (GPU if available)");
  vpl::Workstream decoder(VPL_TARGET_DEVICE_DEFAULT,
                          VPL_WORKSTREAM_DECODEVIDEOPROC);
  decoder.SetConfig(VPL_PROP_SRC_BITSTREAM_FORMAT, VPL_FOURCC_H264);

  LogTrace("Setting target format and color-space (CSC).");
  decoder.SetConfig(VPL_PROP_DST_RAW_FORMAT, VPL_FOURCC_NV12);

  LogTrace("Setting target resolution (scaling).");
  VplVideoSurfaceResolution output_size = {352, 288};
  decoder.SetConfig(VPL_PROP_OUTPUT_RESOLUTION, output_size);

  LogTrace("Creating and initialize demux context.");
  AVFormatContext* fmt_ctx = NULL;
  avsts = avformat_open_input(&fmt_ctx, filename, NULL, NULL);
  if (0 != avsts) {
    fprintf(stderr, "Could not open input file '%s'\n", filename);
    return FAILURE;
  }

  LogTrace("Selecting video stream from demux outputs.");
  avformat_find_stream_info(fmt_ctx, NULL);
  int stream_index =
      av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
  LogTrace("stream_index %d.", stream_index);
  AVPacket pkt = {0};
  av_init_packet(&pkt);

  size_t frame_count = 0;
  Timer timer;

  bool decode_done = false;
  LogTrace("Entering main decode loop");
  while (!decode_done) {
    vplm_mem* image = nullptr;
    size_t bytes_read = 0;

    switch (decoder.GetState()) {
      case VPL_STATE_READ_INPUT:
        // The decoder can accept more data, read it from file and pass it in.
        timer.Start();
        avsts = av_read_frame(fmt_ctx, &pkt);
        if (avsts >= 0) {
          if (pkt.stream_index == stream_index) {
            image = decoder.DecodeProcessFrame(pkt.data, pkt.size);
          }
        } else {
          image = decoder.DecodeFrame(nullptr, 0);
        }
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
      InvertImage(image);
      // DecodeFrame returned a frame, use it.
      frame_count++;
      fprintf(stderr, "Frame: %zu\r", frame_count);
      DisplayFrame(image);
      // Release the reference to the frame, so the memory can be reclaimed
      vplm_unref(image);
    }
  }
  LogTrace("Close demux context input file.");
  avformat_close_input(&fmt_ctx);

  LogTrace("Frames decoded   : %zu", frame_count);
  LogTrace("Frames per second: %02f", frame_count / timer.Elapsed());

  return status;
}

/// Invert pixel data
void InvertImage(vplm_mem* image) {
  vplm_cpu_image handle = {0};
  vplm_image_info desc;
  vplm_get_image_info(image, &desc);
  vplm_status err = vplm_map_image(image, VPLM_ACCESS_MODE_READWRITE, &handle);

  size_t pitch0 = handle.planes[0].stride;
  size_t pitch1 = handle.planes[1].stride;
  for (size_t y = 0; y < desc.height; y++) {
    for (size_t x = 0; x < desc.width; x++) {
      int value = handle.planes[0].data[x + (y * pitch0)];
      value = 255 - value;
      handle.planes[0].data[x + (y * pitch0)] = value;
    }
  }
  for (size_t y = 0; y < desc.height / 2; y++) {
    for (size_t x = 0; x < desc.width; x++) {
      int value = handle.planes[1].data[x + (y * pitch1)];
      value = value - 128;
      value = 128 - value;
      handle.planes[1].data[x + (y * pitch1)] = value;
    }
  }
  vplm_unmap_image(&handle);
}

/// Print command line usage
void PrintUsage(FILE* stream) {
  fprintf(stream, "Usage: %s FILE\n\n", PROGRAM_NAME);
  fprintf(stream,
          "Demux and decode FILE using Intel(R) oneAPI Video Processing "
          "Library.\n\n"
          "Then directly manipulate the decoded frame in code"
          "Demux is done using 3rd party library.\n\n"
          "FILE must be in H264 format\n\n"
          "Example:\n"
          "  %s %s\n",
          PROGRAM_NAME, "content/cars_1280x720.avi");
}

/// Render frame to display
void DisplayFrame(vplm_mem* image) {
  cv::Mat img_nv12, img_bgra;
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

  size_t pitch0 = handle.planes[0].stride;
  size_t pitch1 = handle.planes[1].stride;
  for (size_t y = 0; y < desc.height; y++) {
    memcpy(data + (desc.width * y), handle.planes[0].data + (pitch0 * y),
           desc.width);
  }
  for (size_t y = 0; y < desc.height / 2; y++) {
    memcpy(data + (desc.width * desc.height) + (desc.width * y),
           handle.planes[1].data + (pitch1 * y), desc.width);
  }

  img_nv12 = cv::Mat(desc.height * 3 / 2, desc.width, CV_8UC1, data);
  // Convert NV12 to BGRA format for displaying with OpenCV
  cv::cvtColor(img_nv12, img_bgra, cv::COLOR_YUV2BGRA_NV12);
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
