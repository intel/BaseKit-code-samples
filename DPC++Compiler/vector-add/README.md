# `vector-add` Sample

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04, Windows 10 
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Arria(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)  
  
## Purpose
The `vector-add` is a simple program that adds two large vectors of integers and verifies the results. This program is implemented using C++ and Data Parallel C++ (DPC++) languages for Intel(R) CPU and accelerators. 

In this example, you can learn how to use the most basic code in C++ language that offloads computations to a GPU or an FPGA using the DPC++ language.

## Key Implementation Details 
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.

## License  
This code sample is licensed under MIT license. 

## Building the `vector-add` Program for CPU and GPU 

### On a Linux* System
Perform the following steps:
1. Build the `vector-add` program using:  
    ```
    make all
    ```

2. Run the program using:  
    ```
    make run
    ```

3. Clean the program using:  
    ```
    make clean 

### On a Windows* System Using a Command Line Interface
1. Select **Programs** > **Intel oneAPI 2021** > **Intel oneAPI Command Prompt** to launch a command window.
2. Build the program using the following `nmake` commands:
   ``` 
   nmake -f Makefile.win clean
   nmake -f Makefile.win
   nmake -f Makefile.win run
   ```
		

### On a Windows* System Using Visual Studio* Version 2017 or Newer
Perform the following steps:
1. Launch the Visual Studio* 2017.
2. Select the menu sequence **File** > **Open** > **Project/Solution**. 
3. Locate the `vector-add` folder.
4. Select the `vector-add.sln` file.
5. Select the configuration 'Debug' or 'Release'  
6. Select **Project** > **Build** menu option to build the selected configuration.
7. Select **Debug** > **Start Without Debugging** menu option to run the program.

## Building the `vector-add` Program for Intel(R) FPGA

### On a Linux* System

Perform the following steps:

1. Clean the `vector-add` program using:
    ```
    make clean -f Makefile.fpga
    ```

2. Based on your requirements, you can perform the following:
   * Build and run for FPGA emulation using the following commands:
    ```
    make fpga_emu -f Makefile.fpga
    make run_emu -f Makefile.fpga
    ```
    * Build and run for FPGA hardware.  
      **NOTE:** The hardware compilation takes a long time to complete.
    ```
    make hw -f Makefile.fpga
    make run_hw -f Makefile.fpga
    ```
    * Generate static optimization reports for design analysis. Path to the reports is `vector-add_report.prj/reports/report.html`
    ```
    make report -f Makefile.fpga
    ```

### On a Windows* System Using a Command Line Interface
Perform the following steps:

**NOTE:** On a Windows* system, you can only compile and run on the FPGA emulator. Generating an HTML optimization report and compiling and running on the FPGA hardware are not currently supported.

1. Select **Programs** > **Intel oneAPI 2021** > **Intel oneAPI Command Prompt** to launch a command window.
2. Build the program using the following `nmake` commands:
   ``` 
   nmake -f Makefile.win.fpga clean
   nmake -f Makefile.win.fpga
   nmake -f Makefile.win.fpga run
   ```
		
### On a Windows* System Using Visual Studio* Version 2017 or Newer
Perform the following steps:
1. Launch the Visual Studio* 2017.
2. Select the menu sequence **File** > **Open** > **Project/Solution**. 
3. Locate the `vector-add` folder.
4. Select the `vector-add.sln` file.
5. Select the configuration 'Debug-fpga' or 'Release-fpga' that have the necessary project settings already below: 

	Under the 'Project Property' dialog:

     a. Select the **DPC++** tab.
     b. In the **General** subtab, the **Perform ahead of time compilation for the FPGA** setting is set to **Yes**.
     c. In the **Preprocessor** subtab, the **Preprocessor Definitions" setting has **FPGA_EMULATOR** added.
     d. Close the dialog.

6. Select **Project** > **Build** menu option to build the selected configuration.
7. Select **Debug** > **Start Without Debugging** menu option to run the program.