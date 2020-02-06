Polynomial Integral is a  program that calculates the Integral of a polynomail equation when a large set of upper bound and lower bound values of x are passed to the DPC++ code 
The program calculates the area of the polynomial curve and verifies the results. This program is implemented using C++ and DPC++ language for Intel CPU and accelerators.

  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10 
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)
| What you will learn               | Offloads Integral computations to GPU using Intel DPC++
| Time to complete                  | 15 minutes  
  
## Key implementation details 
This program shows how we can pass a set of upper and lower bound of x axis and calculates the Integral of any polynomial curve.
The basic DPC++ implementation explained inside the code including device selector, buffer, accessor, kernel and command group.

## License  
This code sample is licensed under MIT license. 

## How to Build for CPU and GPU 

### on Linux*  
   * Build the program using Make  
    make all  

   * Run the program  
    make run  

   * Clean the program  
    make clean 

### On Windows

#### Command line using MSBuild

*  MSBuild Poly_Integral.sln /t:Rebuild /p:Configuration="debug"

#### Visual Studio IDE

* Open Visual Studio 2017
* Select Menu "File > Open > Project/Solution", find "Poly_Integral" folder and select "Poly_Integral.sln"
* Select Menu "Project > Build" to build the selected configuration
* Select Menu "Debug > Start Without Debugging" to run the program