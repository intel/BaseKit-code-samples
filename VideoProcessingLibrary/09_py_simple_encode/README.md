# Simple encode

This sample shows how to use oneVPL to perform a simple video encode.

| Optimized for   | Description
|---------------- | ----------------------------------------
| OS              | Ubuntu* 18.04; Windows* 10
| Hardware        | Intel® Processor Graphics GEN9 or newer
| Software        | Intel® oneAPI Video Processing Library

## What You Will Learn

- How to create a workstream
- How to create a encode loop
- How to output the raw video stream to a file


## Time to Complete

  5 minutes


## Sample Details

This sample is a command line application that takes a file containing a raw
YUV stream as an argument, encodes it with the oneVPL encoder, and
reports the encoded compression ratio. The encoded output can also be
written to file `out.h264`. The printed frame rate is measured over frame
encode.


| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | GPU
| Input format      | H.264 video elementary stream
| Output format     | NV12
| Output resolution | same as input
| Output file name  | out.nv12


## Build and Run the Sample

To build and run the sample you need to install prerequisite software and set up
your environment. In addition you should install a raw frame viewer to display
the output.

### Install Prerequisite Software

 - Intel® oneAPI Base Toolkit for Windows* or Linux*
 - [CMake](https://cmake.org)
 - A C/C++ compiler
 - A Python 3.x 64 bit Interpreter

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


#### Windows

Run `setvars.bat` every time you open a new command prompt:

The `setvars.bat` script can be found in the root folder of your oneAPI
installation, which is typically `C:\Program Files (x86)\inteloneapi\` when
installed using default options. If you customized the installation folder, the
`setvars.bat` is in your custom location.

To use the tools, whether from the command line or using Visual Studio,
initialize your environment. To do it in one step for all tools, use the
included environment variable setup utility: `<install_dir>\setvars.bat`)

```
<install_dir>\setvars.bat
```


### Install a Raw Frame Viewer to Display the Output

The sample can write output raw frames to the local filesystem.  A utility to
display the output is needed to see the results.  This tutorial assumes FFmpeg,
which can be quickly installed with 'apt install ffmpeg' in Ubuntu or from
https://ffmpeg.zeranoe.com/builds for Windows.  Many other raw frame viewers
would also work.


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
cmake --build build --target run
```

The run target runs the sample executable with the argument
`$VPL_DIR/samples/content/cars_1280x720.nv12` on Linux and
`%VPL_DIR%\samples\content\cars_1280x720.nv12` on Windows.


## Check the Output
Only applies if `-o` (write encoded frames to output file) is used.
```
ffplay build/out.h264 on Linux and
ffplay build\out.h264 on Windows
```
