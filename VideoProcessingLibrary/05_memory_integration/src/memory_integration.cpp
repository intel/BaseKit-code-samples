/*############################################################################
  # Copyright (C) 2019 Intel Corporation
  #
  # SPDX-License-Identifier: MIT
  ############################################################################*/
/// @example 05_memory_integration.cpp
/// Demonstration of integration in video pipeline
///   3rd party demux -> decode -> custom operation
/// @code
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vplmemory/vplm++.h>
#include "vpl/vpl.hpp"
#include <iostream> 

extern "C" {
#include <libavformat/avformat.h>
}

void DisplayOutput(vplm_mem* img);
#define IS_ARG_EQ(a, b) (!strcmp((a), (b)))

int main(int argc, char* argv[]) {
  printf("Demonstration of integration in video pipeline.\n");
  printf("  3rd party demux -> decode -> custom operation.\n");

  bool printHelp = false, verbose = false, bshow = true;
  int opt_count = 0;
  for (int argIdx = 1; argIdx < argc; argIdx++) {
    if (IS_ARG_EQ(argv[argIdx], "-h")) {
      printHelp = true;
      opt_count++;
    }
    if (IS_ARG_EQ(argv[argIdx], "-v")) {
      verbose = true;
      opt_count++;
    }
    if (IS_ARG_EQ(argv[argIdx], "-o")) {
      bshow = false;
      opt_count++;
    }
  }
  int pos_argc = argc - opt_count - 1;
  if (1 != pos_argc) printHelp = true;
  if (printHelp) {
    printf("Usage: %s [container input file]\n", argv[0]);
    printf("-h\t\tprint help options\n");
    printf("-v\t\tverbose mode\n");
    printf("-o\t\twrite decoded frames to output file\n");
    printf("Example: %s content/cars_1280x720.avi\n", argv[0]);
    printf("Note: may not work with all inputs. ");
    printf("For simplicity this sample does not include ");
    printf("bitstream filter conversion to annex b.");
    return 1; 
  }

  // Create H.264 decoder, default device is GPU if available
  if (verbose) printf("Create H.264 decoder using default device.\n");
  vpl::Decode decoder(VPL_FOURCC_H264);

  // Set output color format
  if (verbose) printf("Set target format and color-space (CSC).\n");
  decoder.SetConfig(VPL_PROP_DST_FORMAT, VPL_FOURCC_I420);

  // Set output resolution
  if (verbose) printf("Set target resolution (scaling).\n");
  VplVideoSurfaceResolution output_size = {352, 288};
  decoder.SetConfig(VPL_PROP_OUTPUT_RESOLUTION, output_size);

  if (verbose) printf("Create and initialize demux context with" \
    "input file '%s'.\n", argv[1]);

  AVFormatContext* fmt_ctx = NULL;
  int avsts = avformat_open_input(&fmt_ctx, argv[1], NULL, NULL);
  if (0 != avsts) {
    printf("Error: could not open input file '%s'\n", argv[1]);
    return 1;
  }
  if (verbose) printf("Select video stream from demux outputs.\n");
  avformat_find_stream_info(fmt_ctx, NULL);
  int stream_index =
      av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
  AVPacket pkt = {0};
  av_init_packet(&pkt);
  
  VplFile* fOutput = nullptr;
  if (!bshow) {
    if (verbose) printf("Open Output file 'out_352x288.yuv'.\n");
    fOutput = vplOpenFile("out_352x288.yuv", "wb");
  }

  // Loop until demux indicates stream read is done
  // Note: To simplify code draining of cached frames is omitted.
  if (verbose) {
    printf("Enter main decode loop.\n");
    printf("  If decoder has room read from demux.\n");
    printf("  Request decoded frame.\n");
    printf("  Apply custom operation (invert pixel values).\n");
    printf("  If decoder has data write to output file.\n");
  }

  int frameCount = 0;
  double elapsedTime = 0.0;
  while (av_read_frame(fmt_ctx, &pkt) >= 0) {
    // select video packets to decode
    if (pkt.stream_index == stream_index) {
      auto decTimeStart = std::chrono::system_clock::now();
      vplm_mem* image = decoder.DecodeFrame(pkt.data, pkt.size);
      auto decTimeEnd = std::chrono::system_clock::now();
      std::chrono::duration<double> t = decTimeEnd - decTimeStart;
      elapsedTime += t.count();
      if (!image) continue;
      frameCount++;
      // Access output frame data.  Here each pixel is read,
      // inverted, and result is written back to the same
      // location.  This is representative of a variety of
      // frame processing operations such as inference or
      // manipulation.
      {
        auto invTimeStart = std::chrono::system_clock::now();
        vplm::cpu::image cpu_image(image, VPLM_ACCESS_MODE_READWRITE);

        uint8_t* data = cpu_image.data(0);
        for (int i = 0; i < cpu_image.buffer_size(); i++) {
          data[i] = 255 - data[i];
        }
        auto invTimeEnd = std::chrono::system_clock::now();
        std::chrono::duration<double> t = invTimeEnd - invTimeStart;
        elapsedTime += t.count();
      }

      if (bshow) {
        // If decode resulted in a frame of output, display on screen
        DisplayOutput(image);
      } else {    
        // If decode resulted in a frame of output write it to file
        vplWriteData(fOutput, image);
        printf(".");
        fflush(stdout);        
      }
    }
  }

  if (verbose) printf("\nClose demux context input file '%s'.\n", argv[1]);
  avformat_close_input(&fmt_ctx);
  if (!bshow) {
    printf("Output file out_352x288.yuv written, ");
    printf("containing i420 raw video format ");
    printf("at the resolution of 352x288.\n");
    if (verbose) printf("Close output file 'out_352x288.yuv'.\n");
    vplCloseFile(fOutput); 
    printf("\nTo view output: \n");
    printf("\"ffplay -s 352x288 -pix_fmt yuv420p -f rawvideo out_352x288.yuv\".\n");
  }

  if (verbose) printf("\nDemux, Decode, VPP processing, and image " \
    "inversion frames per second: %0.2f\n", frameCount / elapsedTime);

  return 0;
}

void DisplayOutput(vplm_mem* img) {
  cv::Mat img_nv12, img_bgr;
  vplm_cpu_image handle = {0};
  vplm_image_info desc;
  unsigned char *data;

  // Read image description (width, height, etc) from vpl memory
  vplm_get_image_info(img, &desc);
  // Access data in read mode
  vplm_status err = vplm_map_image(img, VPLM_ACCESS_MODE_READ, &handle);

  // Need to rearrange data because of stride size
  data = new unsigned char[((desc.height * 3) / 2) * desc.width];

  size_t pitch0 = handle.planes[0].stride;
  size_t pitch1 = handle.planes[1].stride;
  size_t pitch2 = handle.planes[2].stride;
  for(size_t y = 0; y < desc.height; y++){
    memcpy(data + (desc.width * y), handle.planes[0].data + (pitch0 * y), desc.width);
  }
  unsigned char * dest_uv_base = &data[desc.width * desc.height];
  unsigned char * src_u_base = handle.planes[1].data;
  unsigned char * src_v_base = handle.planes[2].data;
  for(size_t y = 0; y < desc.height/2; y++){
    for(size_t x = 0; x < desc.width/2; x++){
      dest_uv_base[x * 2] = src_u_base[x];
      dest_uv_base[(x * 2) + 1] = src_v_base[x];
    }
    dest_uv_base += desc.width;
    src_u_base += pitch1;
    src_v_base += pitch2;
  }

  img_nv12 = cv::Mat(desc.height * 3/2, desc.width, CV_8UC1, data);
  // Convert NV12 to BGRA format for displaying with OpenCV
  cv::cvtColor(img_nv12, img_bgr, cv::COLOR_YUV2BGRA_NV12);
  cv::imshow("Display decoded output", img_bgr);
  cv::waitKey(24);
  vplm_unmap_image(&handle);
  delete data;
  return;  
}
/// @endcode
