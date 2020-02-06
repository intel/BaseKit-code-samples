# Decode and invert with DPC++ kernel

This sample shows how to perform custom video processing using oneVPL and a
DPC++ kernel.


| Optimized for   | Description
|---------------- | ----------------------------------------
| OS              | Ubuntu* 18.04
| Hardware        | Intel® Processor Graphics GEN9 or newer
| Software        | Intel® oneAPI Video Processing Library; Intel® Data Parallel C++ Compiler

## What You Will Learn

- How decode an elementary video stream
- How to select the decode format
- How to select a VPP operation
- How invert decoded frames using the DPCPP API


## Time to Complete

  5 minutes


## Sample Details

This sample is a command line application that takes an elementary stream as an
argument and decodes it with the oneVPL decoder, takes those frames and performs
custom processing (inverting the pixel values) using a DPC++ kernel, and
displays the decoded raw frames to the screen.  The output can also be written
to a file.


| Config            | Default setting
| ----------------- | ----------------------------------
| Target device     | GPU
| Input format      | H.264 video elementary stream
| Input resolution  | 1920x1080
| Output format     | RBGA
| Output resolution | 512x512

## Build and Run the Sample

To build and run the sample you need to install prerequisite software and set up
your environment.

### Install Prerequisite Software

 - Intel® oneAPI Base Toolkit for Linux*
 - [CMake](https://cmake.org)

### Set Up Your Environment

#### Linux

Run `setvars.sh` every time you open a new terminal window:

The `setvars.sh` script can be found in the root folder of your oneAPI
installation, which is typically `/opt/intel/inteloneapi/` when installed as
root or sudo, and `~/intel/inteloneapi/` when installed as a normal user.  If
you customized the installation folder, the `setvars.sh` is in your custom
location.

To use the tools, whether from the command line or using Eclipse, initialize
your environment. To do it in one step for all tools, use the included
environment variable setup utility: `source <install_dir>/setvars.sh`)

```
source <install_dir>/setvars.sh
```

### Install a Raw Frame Viewer to Display the Output

The sample can write output raw frames to the local filesystem.  A utility to
display the output is needed to see the results.  This tutorial assumes FFmpeg,
which can be quickly installed with 'apt install ffmpeg' in Ubuntu.  Many other
raw frame viewers would also work.


### Build the Sample

From the directory containing this README:

```
mkdir build
cd build
cmake ..
cd ..
```


### Run the Sample

```
cmake --build build --target run --config Release
```

The run target runs the sample executable with the arguments
`avc <input-stream> screen 1920 1080 gpu 100`;
where `<input-stream>` is
`$VPL_DIR/samples/content/cars_1920x1080.h264` on Linux.



## Change Where the DPC++ Kernel Runs

By default the DPC++ kernel will run on the GPU. You can set the
`SYCL_DEVICE_TYPE` environment variable to change where the kernel runs.

| `SYCL_DEVICE_TYPE` | Result
|--------------------- | ----------------------------------------
| CPU                  | Run DPC++ kernel on CPU
| GPU                  | Run DPC++ kernel on GPU
| HOST                 | Run C version of kernel on CPU
