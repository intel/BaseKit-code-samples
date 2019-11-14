/*############################################################################
  # Copyright (C) 2019 Intel Corporation
  #
  # SPDX-License-Identifier: MIT
  ############################################################################*/
/// @example 02_decode_accelerator_select.cpp
/// Demonstration of video decode accelerator selection.
/// @code
#include <opencv2/opencv.hpp>
#include <chrono>
#include "vpl/vpl.hpp"

#define BUFFER_SIZE 1024 * 1024
#define IS_ARG_EQ(a, b) (!strcmp((a), (b)))

void DisplayOutput(vplm_mem* img);

int main(int argc, char* argv[]) {
  printf("Demonstration of video decode accelerator selection.\n");

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

  // Create H.264 decoder, specifying a target device
  if (verbose) printf("Create H.264 decoder using desired device device.\n");
  vpl::Decode decoder(VPL_FOURCC_H264, VPL_TARGET_DEVICE_CPU);

  if (verbose) printf("Open Input file '%s'.\n", argv[1]);
  uint8_t *pbs=new uint8_t[BUFFER_SIZE];
  FILE* fInput = fopen(argv[1], "rb");
  if (!fInput) {
    printf("Error: could not open input file '%s'\n", argv[1]);
    return 1;
  }
  
  VplFile* fOutput = nullptr;
  if (!bshow) {
    if (verbose) printf("Open output file 'out.nv12'.\n");
    fOutput = vplOpenFile("out.nv12", "wb");
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
    printf("Output file out.nv12 written, ");
    printf("containing nv12 raw video format ");
    printf("at the resolution of the input H264 file.\n");
    if (verbose) printf("Close output file 'out.nv12'.\n");
    vplCloseFile(fOutput);
    printf("\nTo view output: \n");
    printf("\"ffplay -s 1280x720 -pix_fmt nv12 -f rawvideo out.nv12\".\n");
  }

  if (verbose) printf("\nDecode frames per second: %0.2f\n",
    frameCount / elapsedTime);

  delete[] pbs;
  return 0;
}

void DisplayOutput(vplm_mem* img) {
  cv::Mat img_nv12, img_bgra;
  vplm_cpu_image handle = {0};
  vplm_image_info desc;
  unsigned char *data;

  // Read image description (width, height, etc) from vpl memory
  vplm_get_image_info(img, &desc);
  // Access data in read mode
  vplm_status err = vplm_map_image(img, VPLM_ACCESS_MODE_READ, &handle);

  // Need to rearrange data because of stride size
  data = new unsigned char[desc.height * 3/2 * desc.width];

  size_t pitch0 = handle.planes[0].stride;
  size_t pitch1 = handle.planes[1].stride;
  for(size_t y = 0; y < desc.height; y++){        
    memcpy(data + (desc.width * y), handle.planes[0].data + (pitch0 * y), desc.width);
  }
  for(size_t y = 0; y < desc.height/2; y++){
    memcpy(data + (desc.width * desc.height) + (desc.width * y), handle.planes[1].data + (pitch1 * y), desc.width);
  }

  img_nv12 = cv::Mat(desc.height * 3/2, desc.width, CV_8UC1, data);
  // Convert NV12 to BGRA format for displaying with OpenCV
  cv::cvtColor(img_nv12, img_bgra, cv::COLOR_YUV2BGRA_NV12);
  cv::imshow("Display decoded output", img_bgra);
  cv::waitKey(24);
  vplm_unmap_image(&handle);
  delete data;
  return;  
}
/// @endcode
