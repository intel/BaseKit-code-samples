Projectile_motion is a  program that implements a ballistic equation in DPC++.
It calculates the maximum distance, maximum height reached and the total flight time of a Projectile body. 
Large vector of Projectile class objects with angle of projection and velocity are passed as input to the DPC++ function and verifies the results. 
This program is implemented using C++ and DPC++ language for Intel CPU and accelerators.
The Projectile class is a custom class and this program shows how we can pass an inline function as kernel and use the cl::sycl trignometric fuctions in the kernel code
  

| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10 
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)
| What you will learn               | Using custom class types that offloads computations to GPU using Intel DPC++
| Time to complete                  | 15 minutes  
  
## Key implementation details 
The Projectile class is a custom class and this program shows how we can pass an inline function as kernel and use the cl::sycl trignometric fuctions in the kernel code
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

*  MSBuild Projectile_motion.sln /t:Rebuild /p:Configuration="debug"

#### Visual Studio IDE

* Open Visual Studio 2017
* Select Menu "File > Open > Project/Solution", find "Projectile_motion" folder and select "Projectile_motion.sln"
* Select Menu "Project > Build" to build the selected configuration
* Select Menu "Debug > Start Without Debugging" to run the program