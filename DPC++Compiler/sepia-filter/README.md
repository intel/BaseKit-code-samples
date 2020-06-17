# sepia-filter sample
Sepia filter is a program that converts a color image to a Sepia tone image which is a monochromatic image with a distinctive Brown Gray color. The program works by converting each pixel to Sepia tone and is implemented using C++ and SYCL language for CPU and GPU.
  
| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10
| Hardware                          | Kaby Lake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)
| What you will learn               | The Sepia Filter sample demonstrates the following using the oneAPI DPC++ compiler <ul><li>Writing a custom device selector class</li><li>Offloading compute intensive parts of the application using both lamba and functor kernels</li><li>Measuring kernel execution time by enabling profiling</li></ul>
| Time to complete                  | 20 minutes

 
## Key implementation details [optional]
SYCL implementation explained. 
 

## License  
This code sample is licensed under MIT license 

## How to Build and Run 

### On Linux  
   * Build sepia-filter program  
    cd sepia-filter &&  
    mkdir build &&  
    cd build &&  
    cmake ../. &&  
    make VERBOSE=1  

   * Run the program  
    make run   

### On Windows

#### Command line using MSBuild

*  MSBuild sepia-filter.sln /t:Rebuild /p:Configuration="debug"

#### Visual Studio IDE

* Open Visual Studio 2017
* Select Menu "File > Open > Project/Solution", find "sepia-filter" folder and select "sepia-filter.sln"
* Select Menu "Project > Build" to build the selected configuration
* Select Menu "Debug > Start Without Debugging" to run the program


