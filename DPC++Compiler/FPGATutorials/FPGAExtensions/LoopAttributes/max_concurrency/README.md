# FPGA Tutorial: Maximum Concurrency of a Loop

| Optimized for                     | Description
---                                 |---
| OS                                | Linux* Ubuntu* 18.04; Windows* 10 or Windows* Server 2016
| Hardware                          | Intel(R) Programmable Acceleration Card (PAC) with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel(R) oneAPI DPC++ Compiler (Beta) 

_Notice: Limited support in Windows*, Compiling for FPGA hardware is not supported in Windows*_

## Purpose
This tutorial demonstrates a simple example of applying the `max_concurrency` attribute to a loop in a task kernel to control the memory use and throughput of the loop.

## Key Concepts
This tutorial covers the following concepts:
* Description of the `max_concurrency` attribute 
* Determine how `max_concurrency` attribute affects loop throughput and resource use
* Determine how to apply the `max_concurrency` attribute to loops in your program
* Identify the correct `max_concurrency` factor for your program

## License  
This code sample is licensed under MIT license

## Description of the `max_concurrency` Attribute
The `max_concurrency` attribute is a loop attribute that enables you to control the number of simultaneously executed loop iterations. To enable this simultanous execution, the compiler creates copies of any memory that is private to a single iteration. These copies are called private copies. The greater the permitted concurrency, the more private copies the compiler must create.

### Example: 

Kernels in this tutorial design apply `[[intelfpga::max_concurrency(N)]]` to an outer loop that contains two inner loops, which perform a partial sum computation on an input array, storing the results in a private (to the outer loop) array `a1`. The following is an example of a loop nest:

```
[[intelfpga::max_concurrency(1)]] 
for (unsigned i = 0; i < MAX_ITER; i++) {                                                      
  float a1[SIZE];                                                                              
  for (int j = 0; j < SIZE; j++)                                                               
    a1[j] = accessorA[i * 4 + j] * shift;                                                      
  for (int j = 0; j < SIZE; j++)                                                               
    result += a1[j];                                                                           
}   
```

In this example, the maximum concurrency allowed for the outer loop is 1, that is, only one iteration of the outer loop is allowed to be simultaneously executing at any given moment. Additionally, the `max_concurrency` attribute in this example forces the compiler to create exactly one private copy of the array `a1`. Passing the parameter `N` to the `max_concurrency` attribute limits the concurrency of the loop to `N` simultaneous iterations, and `N` private copies of privately-declared arrays in that loop.

## Identifying the Correct `max_concurrency` Factor
Generally, increasing the maximum concurrency allowed for a loop through the use of the `max_concurrency` attribute increases the throughput of that loop at the cost of increased memory resource use. Additionally, in nearly all cases, there is a point at which increasing the maximum concurrency does not have any further effect on the throughput of the loop, as the maximum exploitable concurrency of that loop has been achieved. 

The correct `max_concurrency` factor for a loop depends on the goals of your design, the criticality of the loop in question, and its impact on the overall throughput of your design. A typical design flow may be to: 
1. Experiment with different values of `max_concurrency`. 
2. Observe what impact the values have on the overall throughput and memory use of your design.
3. Choose the appropriate value that allows you to achive your desired throughput and area goals.

## Building the `max_concurrency` Example (Linux)
1. Install the design into the `build` directory from the design directory by running `cmake`:

```
mkdir build
cd build
cmake ..
```

2. Compile the design through the generated `Makefile`. The following four build targets are provided, matching the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device) using: 
      ```
      make fpga_emu
      ./max_concurrency.fpga_emu 
      ```

   * Generate the HTML optimization report using: 

     ```
     make report
     ``` 
     Locate the report in the `max_concurrency.prj/reports/report.html` directory.

     On the main report page, scroll down to the section titled "Estimated Resource Usage". Each kernel name ends in the max_concurrency attribute argument used for that kernel, e.g., kernelCompute1 uses a max_concurrency attribute value of 1. You can verify that the number of RAMs used for each kernel increases with the max_concurrency value used, with the exception of max_concurrency 0, which allows the compiler to choose a default value.

   * Compile and run on an FPGA hardware (longer compile time, targets FPGA device) using: 

     ```
     make fpga 
     ./max_concurrency.fpga 

     The stdout output shows the GFlops for each kernel. On the PAC10 hardware board, we see that the number of GFlops doubles from using max_concurrency 1 to max_concurrency 2, after which increasing the value of max_concurrency does not increase the GFlops achieved, i.e., increasing the max_concurrency above 2 will spend additional RAM usage for no additional throughput gain. As such, for this tutorial design, maximal throughput is best achieved by using max_concurrency 2.
     ```

   * Compile and run on a CPU hardware (unoptimized) using: 

     ```
     make cpu_host
     ./max_concurrency.cpu_host 
     ```

3. Download the design, compiled for FPGA hardware, from this location: [download page](https://www.intel.com/content/www/us/en/programmable/products/design-software/high-level-design/one-api-for-fpga-support.html)

## Building the `max_concurrency` Example (Windows)
Note: `cmake` is not yet supported on Windows, a build.ninja file is provided instead. 

1. Enter source file directory.

```
cd src
```

2. Compile the design. The following four build targets are provided, matching the recommended development flow:

   * Compile and run for emulation (fast compile time, targets emulated FPGA device) using: 
      ```
      ninja fpga_emu
      max_concurrency.fpga_emu.exe
      ```

   * Generate the HTML optimization report.

     ```
     ninja report
     ```
     Locate the report the `../src/max_concurrency.prj/reports/report.html` directory.

     On the main report page, scroll down to the section titled "Estimated Resource Usage". Each kernel name ends in the max_concurrency attribute argument used for that kernel, e.g., kernelCompute1 uses a max_concurrency attribute value of 1. You can verify that the number of RAMs used for each kernel increases with the max_concurrency value used, with the exception of max_concurrency 0, which allows the compiler to choose a default value.

   * **Not supported yet:** Compile and run on an FPGA hardware.

   * Compile and run on a CPU hardware (unoptimized) using: 

     ```
     ninja cpu_host
     max_concurrency.cpu_host.exe 
     ```

## Building the `max_concurrency` Example in Third-Party Integrated Development Environments (IDEs)

You can compile and run this tutorial in the Eclipse* IDE (in Linux*) and the Visual Studio* IDE (in Windows*). For instructions, refer to the following link: [Intel(R) oneAPI DPC++ FPGA Workflows on Third-Party IDEs](https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-fpga-workflow-on-ide)
