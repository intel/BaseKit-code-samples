The bootstrapping program presents a very simpler introduction to DPC++ than vector-add. In training scenarios the developer is lead an walkthrough of the code to familiarize them with DPC++ classes and concepts as well as the basic use of the dpcpp compiler.
  
| Optimized for                       | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)
| What you will learn               | Developer will learn basic DPC++ classes and concepts 
| Time to complete                  | 10 minutes  
  
## Key implementation details 
The implementation is meant to be very simple. It demonstrates without the use of STL, for example, returning a character buffer from the bootstrap_function() as it is executed in a single_task().

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

*  MSBuild bootstrapping.sln /t:Rebuild /p:Configuration="debug"

#### Visual Studio IDE

* Open Visual Studio 2017
* Select Menu "File > Open > Project/Solution", find "bootstrapping" folder and select "bootstrapping.sln"
* Select Menu "Project > Build" to build the selected configuration
* Select Menu "Debug > Start Without Debugging" to run the program