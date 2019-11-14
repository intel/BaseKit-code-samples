/*############################################################################
  # Copyright (C) 2019 Intel Corporation
  #
  # SPDX-License-Identifier: MIT
  ############################################################################*/
/// @example 03_decode_vpp.cpp
/// Demonstration of video decode and processing.
/// @code
#include <opencv2/opencv.hpp>
#include <chrono>
#include "vpl/vpl.hpp"

#define BUFFER_SIZE 1024 * 1024
#define IS_ARG_EQ(a, b) (!strcmp((a), (b)))

void DisplayOutput(vplm_mem* img);

int main(int argc, char* argv[]) {
  printf("Demonstration of video decode and processing.\n");

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
    printf("Usage: %s [h264 input file]\n", argv[0]);
    printf("-h\t\tprint help options\n");
    printf("-v\t\tverbose mode\n");
    printf("-o\t\twrite decoded frames to output file\n");
    printf("Example: %s content/cars_1280x720.h264\n", argv[0]);
    return 1;
  }

  // Create H.264 decoder, default device is GPU if available
  if (verbose) printf("Create H.264 decoder using default device.\n");
  vpl::Decode decoder(VPL_FOURCC_H264);

  // Set output color format
  if (verbose) printf("Set target format and color-space (CSC).\n");
  decoder.SetConfig(VPL_PROP_DST_FORMAT, VPL_FOURCC_RGBA);

  // Set output resolution
  if (verbose) printf("Set target resolution (scaling).\n");
  VplVideoSurfaceResolution output_size = {352, 288};
  decoder.SetConfig(VPL_PROP_OUTPUT_RESOLUTION, output_size);

  if (verbose) printf("Open Input file '%s'.\n", argv[1]);
  uint8_t *pbs=new uint8_t[BUFFER_SIZE];
  FILE* fInput = fopen(argv[1], "rb");
  if (!fInput) {
    printf("Error: could not open input file '%s'\n", argv[1]);
    return 1;
  }
  
  VplFile* fOutput = nullptr;
  if (!bshow) {
    if (verbose) printf("Open Output file 'out_352x288.rgba'.\n");
    fOutput = vplOpenFile("out_352x288.rgba", "wb");
  }

  // Loop until done.  Decode state of END_OF_OPERATION or
  // ERROR indicates loop exit.
  if (verbose) {
    printf("Enter main decode loop.\n");
    printf("  If decoder has room read from input file.\n");
    printf("  Request decoded frame.\n");
    printf("  If decoder has data write to output file.\n");
  }
  vplm_mem* image = nullptr;
  bool bdrain_mode = false;
  vplWorkstreamState decode_state = VPL_STATE_READ_INPUT;
  int frameCount = 0;
  double elapsedTime = 0.0;
  // decode loop
  for (; decode_state != VPL_STATE_END_OF_OPERATION &&
         decode_state != VPL_STATE_ERROR;
       decode_state = decoder.GetState()) {

    // read more input if state indicates buffer space
    // is available
    uint32_t bs_size = 0;
    if ((decode_state == VPL_STATE_READ_INPUT) && (!bdrain_mode)) {
      bs_size = (uint32_t)fread(pbs, 1, BUFFER_SIZE, fInput);
    }

    if (bs_size == 0 || decode_state == VPL_STATE_INPUT_BUFFER_FULL) {
      bdrain_mode = true;
    }

    // Attempt to decode a frame.  If more data is needed read again
    auto decTimeStart = std::chrono::system_clock::now();
    if (bdrain_mode)
      image = decoder.DecodeFrame(nullptr, 0);
    else
      image = decoder.DecodeFrame(pbs, bs_size);
    auto decTimeEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> t = decTimeEnd - decTimeStart;
    elapsedTime += t.count();
    if (!image) continue;
    frameCount++;

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

  if (verbose) printf("\nClose input file '%s'.\n", argv[1]);
  fclose(fInput);
  if (!bshow) {
    printf("Output file out_352x288.rgba written, ");
    printf("containing rgba raw video format ");
    printf("at the resolution of 352x288.\n");
    if (verbose) printf("Close output file 'out_352x288.rgba'.\n");
    vplCloseFile(fOutput);
    printf("\nTo view output: \n");
    printf("\"ffplay -s 352x288 -pix_fmt rgba -f rawvideo out_352x288.rgba\".\n");
  }

  if (verbose) printf("\nDecode and VPP processing " \
    "frames per second: %0.2f\n", frameCount / elapsedTime);

  delete[] pbs;
  return 0;
}

void DisplayOutput(vplm_mem* img) {
  cv::Mat img_rgba;
  vplm_cpu_image handle = {0};
  vplm_image_info desc;
  unsigned char *data;

  // Read image description (width, height, etc) from vpl memory
  vplm_get_image_info(img, &desc);
  // Access data in read mode
  vplm_status err = vplm_map_image(img, VPLM_ACCESS_MODE_READ, &handle);

  // Need to rearrange data because of stride size
  data = new unsigned char[desc.height * desc.width * 4];

  size_t pitch = handle.planes[0].stride;
  
  for(size_t y = 0; y < desc.height; y++){    
    memcpy(data + ((desc.width * 4) * y), handle.planes[0].data + (pitch * y), desc.width * 4);
  }

  img_rgba = cv::Mat(desc.height, desc.width, CV_8UC4, data);

  cv::imshow("Display decoded output", img_rgba);
  cv::waitKey(24);
  vplm_unmap_image(&handle);
  delete data;  
  return;  
}
/// @endcode
