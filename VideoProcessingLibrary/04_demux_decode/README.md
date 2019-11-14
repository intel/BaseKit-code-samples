# Demux and decode

This sample shows how to use VPL to decode a video stream from a media
container.

| Optimized for   | Description
|---------------- | ----------------------------------------
| OS              | Ubuntu* 18.04; Windows* 10
| Hardware        | Intel® Processor Graphics GEN9 or newer
| Software        | Intel® Video Processing Library

## What You Will Learn

- How to create a workstream
- How to use a standard FFmpeg API to demux the video stream and connect to the
  VPL input
- How to configure the workstream to set the color format and resolution
- How to create a decode loop
- How to output the raw video stream to a file


## Time to Complete

  5 minutes


## Sample Details

This sample is a command line application that takes an AVI container with an
H.264 stream as an argument, decodes it with the VPL decoder, converts the
output to BGRA format with 352x288 resolution, and displays decoded raw frames
to the screen.  The decoded output can also be written to file
`out_352x288.rgba`. The printed frame rate is measured over the H.264 stream
decode and VPP processing.

| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | GPU
| Input format      | AVI container with H.264 stream
| Output format     | BGRA
| Output resolution | 352x288
| Output file name  | out_352x288.rgba


## Build and Run the Sample

To build and run the sample you need to install prerequisite software and set up
your environment. In addition you should install a raw frame viewer to display
the output.

### Install Prerequisite Software

 - Intel® oneAPI Base Toolkit for Windows* or Linux*
 - [CMake](https://cmake.org)
 - A C/C++ compiler


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
`$VPL_DIR/samples/content/cars_1280x720.avi` on Linux and
`%VPL_DIR%\samples\content\cars_1280x720.avi` on Windows.


## Check the Output
Only applies if `-o` (write decoded frames to output file) is used.
```
ffplay -s 352x288 -pix_fmt bgra -f rawvideo build/out_352x288.rgba on Linux and
ffplay -s 352x288 -pix_fmt bgra -f rawvideo build\out_352x288.rgba on Windows
```
