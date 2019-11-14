/*############################################################################
  # Copyright (C) 2019 Intel Corporation
  #
  # SPDX-License-Identifier: MIT
  ############################################################################*/

#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vpl/vpl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "vpl/vpl.h"
#include "vplmemory/vplm_sycl.h"
#include <opencv2/opencv.hpp>

using namespace cl;
using namespace cl::sycl;

#define BUFFER_SIZE 1024

void DisplayOutput(vplm_mem* img);

int main(int argc, char* argv[]) {
  vplStatus sts = VPL_OK;
  vplWorkstreamState decode_state = VPL_STATE_READ_INPUT;
  vplm_mem* frame = 0;
  uint32_t bs_size = 0;
  size_t frame_count = 0, frame_count_max = (size_t)-1;
  uint8_t pbs[BUFFER_SIZE];
  VplFourCC src_format, dst_format;
  FILE* fInput;
  VplFile* fOutput;
  VplFile* fInverse_Output;
  VplTargetDevice device;
  char output_file[256] = {0};
  char inverse_output_file[256] = {0};
  bool bshow = false;

  // For future expansion. Currently, only RGBA is supported in DPC++. In the
  // future more formats will be added.
  //  if (argc < 9){
  //    printf("Error: invalid arguments.\nUsage:\n\t%s [avc | hevc]
  //    <input_bitstream_path> "
  //      "<output_frames_path> <out_width> <out_height> [nv12 | rgba | yuy2]
  //      [gpu | cpp] <frame_count>\n", argv[0]);
  //    return -1;
  //  }
  if (argc < 8) {
    printf(
        "Error: invalid arguments.\nUsage:\n\t%s [avc] <input_bitstream_path> "
        "<output_frames_path> <out_width> <out_height>  [gpu | cpp] "
        "<frame_count>\n",
        argv[0]);
    return -1;
  }

  if (argc < 9) {
    frame_count_max = atoi(argv[7]);
    printf("decoding first %i frames\n", (int)frame_count_max);
  }

  // Currently only AVC is supported
  src_format = (strcmp(argv[1], "avc") == 0) ? VPL_FOURCC_H264 : (VplFourCC)-1;

  // Currently DPCPP only supports RGBA input format
  dst_format = VPL_FOURCC_RGBA;
  //  dst_format =(strcmp(argv[6], "rgba") == 0) ? VPL_FOURCC_RGBA :
  //              (VplFourCC)-1;

  // This pertains to media engine. DPCPP device is selected through
  // SYCL_DEVICE_TYPE environment variable
  device = (strcmp(argv[6], "gpu") == 0)
               ? VPL_TARGET_DEVICE_GPU_GEN
               : (strcmp(argv[6], "cpu") == 0) ? VPL_TARGET_DEVICE_CPU
                                               : (VplTargetDevice)-1;

  fInput = fopen(argv[2], "rb");
  if (!fInput) printf("Error: Invalid input file '%s'\n", argv[2]);
  
  if (strcmp(argv[3], "screen") == 0) {
    bshow = true;
    printf("output will be displayed on screen\n");
  } else {
    sprintf(output_file, "%s_%s_%sx%s.rgba", argv[3], argv[6], argv[4], argv[5]);
    sprintf(inverse_output_file, "%s_%s_%sx%s_inverse.rgba", argv[3], argv[6],
            argv[4], argv[5]);
    printf("%s\n", output_file);
    printf("%s\n", inverse_output_file);  

    fOutput = vplOpenFile(output_file, "wb");
    if (!fOutput) printf("Error: Invalid output file '%s'\n", output_file);

    fInverse_Output = vplOpenFile(inverse_output_file, "wb");
    if (!fInverse_Output)
      printf("Error: Invalid output file '%s'\n", inverse_output_file);
  }

  // Preparation for Sycl interface
  vplm_image_info desc = {};
  desc.width = static_cast<uint32_t>(atoi(argv[4]));
  desc.height = static_cast<uint32_t>(atoi(argv[5]));
  int width = static_cast<uint32_t>(atoi(argv[4]));
  int height = static_cast<uint32_t>(atoi(argv[5]));

  // sycl_device_selector device_selector(argv[7]);
  // sycl_device_selector device_selector(SYCL_DEVICE);

  sycl::default_selector device_selector;
  sycl::queue sycl_queue(device_selector);

  std::cout << "\nSycl kernels executing on: "
            << sycl_queue.get_device().get_info<info::device::name>() << "\n\n";

  vplm::memory inverseframe =
      vplm::cpu::make_memory(width, height, VPLM_PIXEL_FORMAT_RGBA);
  //   vplm::memory inverseframe = vplm::cpu::make_memory(width, height,
  //   VPLM_PIXEL_FORMAT_RGBP);

  vplWorkstream decoder = vplCreateWorkStream(VPL_WORKSTREAM_DECODE, device);

  VplVideoSurfaceResolution output_size = {
      static_cast<uint32_t>(atoi(argv[4])),
      static_cast<uint32_t>(atoi(argv[5]))};
  sts = vplSetConfigProperty(decoder, VPL_PROP_SRC_FORMAT, &src_format,
                             sizeof(src_format));
  sts = vplSetConfigProperty(decoder, VPL_PROP_DST_FORMAT, &dst_format,
                             sizeof(dst_format));
  sts = vplSetConfigProperty(decoder, VPL_PROP_OUTPUT_RESOLUTION, &output_size,
                             sizeof(output_size));

  for (; decode_state != VPL_STATE_END_OF_OPERATION &&
         decode_state != VPL_STATE_ERROR;
       decode_state = vplWorkstreamGetState(decoder)) {
    vplWorkstreamState state = vplWorkstreamGetState(decoder);
    if (state == VPL_STATE_READ_INPUT) {
      bs_size = (uint32_t)fread(pbs, 1, BUFFER_SIZE, fInput);
    }
    frame = vplDecodeFrame(decoder, pbs, bs_size);

    if (frame) {
      if(!bshow) {
        sts = vplWriteData(fOutput, frame);
      }

      char img[360000];  // This is layman's debug tool for inside the kernel.
                         // the size is widthxheightx4

      // Entering SYCL domain
      {
        // Wrapper for VPL memory object. the result will be a SYCL image data
        // type. there is no copy.
        vplm::sycl::memory sycl_mem1(frame);
        vplm::sycl::memory sycl_mem2(inverseframe);
        cl::sycl::image<2> sycl_image1 =
            sycl_mem1.acquire_image(sycl_queue, access::mode::read);
        cl::sycl::image<2> sycl_image2 =
            sycl_mem2.acquire_image(sycl_queue, access::mode::write);

        // This is alternate way of defining SYCL image. it is for debug purpose
        // inside kernel if needed.
        cl::sycl::image<2> sycl_image3(img, image_channel_order::rgba,
                                       image_channel_type::unsigned_int8,
                                       range<2>{300, 300});

        // Kernel for RGB->Inverse RGB processing
        sycl_queue.submit([&](handler& cgh) {
          // Definition of source accessor. This accessor gains access to the
          // decoded frame
          cl::sycl::accessor<uint4, 2, access::mode::read,
                             access::target::image>
              accessorSRC(sycl_image1, cgh);

          // Alternate method to define accessors. This accessor provides means
          // to write to the resulting VPL compatible data type (inverseframe)
          //                   auto accessorDST =
          //                   sycl_image2.get_access<cl::sycl::uint4,
          //                   access::mode::write> (cgh);
          auto accessorDST =
              sycl_image2.get_access<cl::sycl::uint4, access::mode::write>(cgh);
          auto accessorDST2 =
              sycl_image3.get_access<cl::sycl::uint4, access::mode::write>(cgh);

          //                    Streamer to be used for debug purposes.
          //                    cl::sycl::stream Out(2073600, 80, cgh);

          // Parallel processing of the input frame. it traverses across the
          // entire two dimensional input. There is no tiling.
          cgh.parallel_for<class RGB2Inverse>(
              range<2>(desc.width, desc.height), [=](item<2> item) {
                auto coords = int2(item[0], item[1]);  // locating pixels
                cl::sycl::uint4 rgba = accessorSRC.read(
                    coords);  // Reading pixel values using their location
                cl::sycl::uint4 inversepixel = cl::sycl::uint4(
                    255 - rgba.x(), 255 - rgba.y(), 255 - rgba.z(),
                    rgba.w());  // Forming new inversed pixel

                // Writing the resulting inverse pixels to the inverseframe
                // using its accessor
                accessorDST.write(coords, inversepixel);
              });
        });
      }

      // Writing resulting frames to output file at frame boundary
      if(bshow) {
        DisplayOutput((vplm_mem*)inverseframe());
      } else {
        sts = vplWriteData(fInverse_Output, (vplm_mem*)inverseframe());
      }

      // frame_count_max is set through command line option when program is
      // executed.
      if (++frame_count >= frame_count_max) {
        break;
      }
    }
  }
  fclose(fInput);

  if(!bshow) {
    vplCloseFile(fOutput);
    vplCloseFile(fInverse_Output);
  }
  vplDestroyWorkstream(&decoder);

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