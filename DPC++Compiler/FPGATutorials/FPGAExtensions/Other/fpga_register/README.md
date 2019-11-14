# FPGA Tutorial: FPGA reg

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel(R); Programmable Acceleration Card (PAC) with Intel(R); Arria(R); 10 GX FPGA
| Software                          | Intel(R) oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates a simple example of using the `intel::fpga_reg` extension to improve the fmax of the design.

## Key Concepts
This tutorial helps you learn the following concepts:
* How to use the `intel::fpga_reg` extension


## Building the `fpga_register` Design (Linux)

1. Install the design in `build` directory from the design directory by running `cmake`:
    ```
    $mkdir build
    $cd build
    $cmake ..
    ```

2. Compile the design using the generated `Makefile`. The following four build targets are provided that match the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulates an FPGA device) using:
     ```
     make fpga_emu
     ./fpga_reg.fpga_emu
     ```

   * Generate HTML optimization reports using:
     ```
     make report
     ```
     Locate the reports in:
     1. `fpga_reg_report.prj/reports/report.html` 
     2. `fpga_reg_registered_report.prj/reports/report.html`

     Observe the Graph Viewer and notice the changes within `Cluster 1` of the `SimpleMath.B1` block.
     You can notice that in the report from 1., the viewer shows a much more shallow graph as compared to the one in 2.
     This is because the operations are performed much closer to one another in 1. as compared to 2. and in doing so, the compiler sacrificed fmax to be able to schedule the operations within a single cycle.

   * Compile and run on FPGA hardware (longer compile time, targets an FPGA device) using:
     ```
     make fpga
     ./fpga_reg.fpga
     ./fpga_reg_registered.fpga
     ```
     >**NOTE**: Only the FPGA hardware flow illustrates the performance difference of using the fpga_reg extension. This will be easily evident with the differences in the fmax of the two designs. The fmax can be found in `fpga_reg.prj/reports/report.html` and `fpga_reg_registered.prj/reports/report.html`.

   * Compile and run on CPU hardware (not optimized) using: 
     ```
     make cpu_host
     ./fpga_reg.cpu_host
     ```
3. Download the design, compiled for FPGA hardware, from this location: [download page](https://www.intel.com/content/www/us/en/programmable/products/design-software/high-level-design/one-api-for-fpga-support.html)

## Building the `fpga_register` Design (Windows)

Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following four build targets are provided that match the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulates an FPGA device) using:
     ```
     ninja fpga_emu
     fpga_reg.fpga_emu.exe
     ```

   * Generate HTML optimization reports.
     ```
     ninja report
     ninja report_registerd
     ```
     Locate the reports in:
     1. `fpga_reg_report.prj/reports/report.html` 
     2. `fpga_reg_registered_report.prj/reports/report.html`

     Observe the Graph Viewer and notice the changes within `Cluster 1` of the `SimpleMath.B1` block.
     You can notice that in the report from 1., the viewer shows a much more shallow graph as compared to the one in 2.
     This is because the operations are performed much closer to one another in 1. as compared to 2. and in doing so, the compiler sacrificed fmax to be able to schedule the operations within a single cycle.

   * **Not supported yet:** Compile and run on FPGA hardware.

   * Compile and run on CPU hardware (not optimized) using: 
     ```
     ninja cpu_host
     fpga_reg.cpu_host.exe
     ```

## Building the `fpga_register` Design in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel(R) oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
