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

#include <opencv2/opencv.hpp>
#include "vpl/vpl.hpp"
#include "vplmemory/vplm_sycl++.h"

extern "C" {
#include <libavformat/avformat.h>
}

#include <CL/cl.h>
using namespace cl;
using namespace cl::sycl;

#define PROGRAM_NAME "decode_sycl_invert"
const int SUCCESS = 0;
const int FAILURE = -1;

const int kMaxFramesToProcess = 170;
const int kOutputWidth = 512;
const int kOutputHeight = 512;

void LogTrace(const char* fmt, ...);
void InvertImage(vplm_mem* image, const vplm_mem* inverseImage);
void DisplayFrame(const vplm_mem* image);
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
  {
    std::ifstream input_stream(argv[1], std::ios::binary);
    if (!input_stream.is_open()) {
      fprintf(stderr, "%s: could not open input file '%s'\n", PROGRAM_NAME,
              argv[1]);
      return FAILURE;
    }
  }

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
  decoder.SetConfig(VPL_PROP_DST_RAW_FORMAT, VPL_FOURCC_BGRA);

  LogTrace("Setting target resolution (scaling).");
  VplVideoSurfaceResolution output_size = {kOutputWidth, kOutputHeight};
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

      default:
        LogTrace("Unexpected state during decode. Exiting.");
        decode_done = true;
        status = FAILURE;
    }

    if (image) {
      vplm_image_info desc;
      vplm_get_image_info(image, &desc);
      vplm::memory inverseImage = vplm::cpu::make_memory(
          desc.width, desc.height, VPLM_PIXEL_FORMAT_RGBA);
      InvertImage(image, inverseImage());
      // DecodeFrame returned a frame, use it.
      frame_count++;
      fprintf(stderr, "Frame: %zu\r", frame_count);
      DisplayFrame(inverseImage());
      // Release the reference to the frame, so the memory can be reclaimed
      vplm_unref(image);
      if (frame_count > kMaxFramesToProcess) {
        decode_done = true;
        status = SUCCESS;
      }
    }
  }
  LogTrace("Close demux context input file.");
  avformat_close_input(&fmt_ctx);

  LogTrace("Frames decoded   : %zu", frame_count);
  LogTrace("Frames per second: %02f", frame_count / timer.Elapsed());

  return status;
}

/// Invert pixel data
void InvertImage(vplm_mem* image, const vplm_mem* inverseImage) {
  // exception handler
  /*
  The exception_list parameter is an iterable list of std::exception_ptr
  objects. But those pointers are not always directly readable. So, we rethrow
  the pointer, catch it,  and then we have the exception itself. Note: depending
  upon the operation there may be several exceptions.
  */
  auto exception_handler = [](sycl::exception_list exceptionList) {
    for (std::exception_ptr const& e : exceptionList) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& e) {
        // std::terminate() will exit the process, return non-zero, and output
        // a message to the user about the exception
        std::terminate();
      }
    }
  };
  sycl::host_selector device_selector;
  sycl::queue sycl_queue(device_selector, exception_handler);

  vplm_image_info desc;
  vplm_get_image_info(image, &desc);

  // Enter SYCL domain
  {
    // SYCL wrappers for oneVPL images.
    vplm::sycl::memory sycl_mem_in(image);
    vplm::sycl::memory sycl_mem_out(inverseImage);
    cl::sycl::image<2> sycl_image_in =
        sycl_mem_in.acquire_image(sycl_queue, access::mode::read);
    cl::sycl::image<2> sycl_image_out =
        sycl_mem_out.acquire_image(sycl_queue, access::mode::write);

    // Kernel for RGB->Inverse RGB processing.
    sycl_queue.submit([&](handler& cgh) {
      // This accessor provides access to the decoded images.
      cl::sycl::accessor<uint4, 2, access::mode::read, access::target::image>
          accessorSRC(sycl_image_in, cgh);
      // This accessor provides means to write to the resulting oneVPL images.
      auto accessorDST =
          sycl_image_out.get_access<cl::sycl::uint4, access::mode::write>(cgh);
      // Traverses across the entire two dimensional input.
      cgh.parallel_for<class RGB2Inverse>(
          range<2>(desc.width, desc.height), [=](item<2> item) {
            // Locate pixels
            auto coords = cl::sycl::int2(item[0], item[1]);
            // Read pixels
            cl::sycl::uint4 rgba = accessorSRC.read(coords);
            // Invert Pixels
            cl::sycl::uint4 inversepixel = cl::sycl::uint4(
                255 - rgba.x(), 255 - rgba.y(), 255 - rgba.z(), rgba.w());
            // Write inverse pixels to the inverse image
            accessorDST.write(coords, inversepixel);
          });
    });
    sycl_queue.wait_and_throw();
  }
}

/// Print command line usage
void PrintUsage(FILE* stream) {
  fprintf(stream, "Usage: %s FILE\n\n", PROGRAM_NAME);
  fprintf(stream,
          "Demux and decode FILE using Intel(R) oneAPI Video Processing "
          "Library.\n\n"
          "Then manipulate the decoded frame using Sycl code"
          "Demux is done using 3rd party library.\n\n"
          "FILE must be in H264 format\n\n"
          "Example:\n"
          "  %s %s\n",
          PROGRAM_NAME, "content/cars_1280x720.avi");
}

/// Render frame to display
void DisplayFrame(const vplm_mem* image) {
  cv::Mat img_bgra;
  vplm_cpu_image handle = {0};
  vplm_image_info desc;

  bool have_display = true;
#ifdef __linux__
  static bool first_call = true;
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

  unsigned char* data = new unsigned char[desc.height * desc.width * 4];
  size_t pitch0 = handle.planes[0].stride;
  for (size_t y = 0; y < desc.height; y++) {
    memcpy(data + (desc.width * 4 * y), handle.planes[0].data + (pitch0 * y),
           desc.width * 4);
  }

  img_bgra = cv::Mat(desc.height, desc.width, CV_8UC4, data);
  if (have_display) cv::imshow("Display decoded output", img_bgra);
  cv::waitKey(24);
  vplm_unmap_image(&handle);
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
