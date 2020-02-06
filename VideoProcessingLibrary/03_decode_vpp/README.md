# Decode with video post-processing

This sample shows how to select a color format and output resolution when
decoding with oneVPL.

| Optimized for   | Description
|---------------- | ----------------------------------------
| OS              | Ubuntu* 18.04; Windows* 10
| Hardware        | Intel® Processor Graphics GEN9 or newer
| Software        | Intel® oneAPI Video Processing Library

## What You Will Learn

- How to create a workstream
- How to configure the workstream to set the color format and resolution
- How to create a decode loop
- How to output the raw video stream to a file


## Time to Complete

  5 minutes


## Sample Details

This sample is a command line application that takes a file containing an H.264
video elementary stream as an argument, decodes it with the oneVPL decoder,
converts the output to BGRA format with 352x288 resolution, and displays decoded
raw frames to the screen. The decoded output can also be written to file
`out_352x288.rgba`. If there is a difference between input and output format
resolution video post-processing will be done automatically.  This sample is
different from previous sample decode_accelerator_select in that it shows how to
rescale video, and how to output to a new color format using `SetConfig`.  Both
the rescale and color format change utilize VPP (Video Pre/Post Processing).
The printed frame rate is measured over the decode and VPP processing.

| Configuration     | Default setting
| ----------------- | ----------------------------------
| Target device     | GPU
| Input format      | H.264 video elementary stream
| Output format     | BGRA
| Output resolution | 352x288


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
`$VPL_DIR/samples/content/cars_1280x720.h264` on Linux and
`%VPL_DIR%\samples\content\cars_1280x720.h264` on Windows.

___
***NOTE***

Output is BGRA but format specified is RGBA.  This is a known
issue and will be fixed in a future release.
___
