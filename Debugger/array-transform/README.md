# DPC++ Sample for the Debugger

This is a small DPC++ example that accompanies the
[Get Started Guide](https://software.intel.com/en-us/get-started-with-debugging-dpcpp)
of the debugger.

| Optimized for                   | Description
|---------------------------------|--------------
| OS                              | Linux Ubuntu 18.04
| Hardware                        | Kaby Lake with GEN9 (on GPU) or newer (on CPU)
| Software                        | Intel&reg; oneAPI DPC++ Compiler (beta) 
| What you will learn             | Essential debugger features for effective debugging of DPC++ on CPU and GPU
| Time to complete                | 20 minutes for CPU, 30 minutes for GPU

## License

This code sample is licensed under MIT license.

## How to build and run

### On Linux

To build:

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

*Note:* The cmake configuration enforces the `Debug` build type.

To run the program:

```
$ make run
```

To start a debugging session:

```
$ make debug-session
```

*Note:* This will set the environment variable
`SYCL_PROGRAM_BUILD_OPTIONS` to `"-g -cl-opt-disable"` before starting
the debugger.

To clean up:

```
$ make clean
```
