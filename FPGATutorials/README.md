# FPGA Tutorials

Tutorials for oneAPI FPGA developers.

## Tutorials by Category

### Compilation

| Tutorial Name                         | Description                                                                                 |
|---------------------------------------|---------------------------------------------------------------------------------------------|
| [Compile Flow](./Compilation/compile_flow)  | Introduction to most commonly used compilation commands targeting Intel FPGA devices.       |
| [Device Link](./Compilation/device_link)   | Demonstrates how to use device link mechanism to separate device and host part compilation. |
| [Use Library](./Compilation/use_library)   | Demonstrates usage of device side library.                                               |


### FPGA extensions

| Tutorial Name          | Description                                     |
|------------------------|-------------------------------------------------|
| [Memory Attributes](./FPGAExtensions/MemoryAttributes/memory_attributes_overview)     | Use memory attribute to configure local memory. |
| [Kernel Args Restrict](./FPGAExtensions/MemoryAttributes/kernel_args_restrict)   | Marks all kernel buffer arguments as not aliasing to each other |
| [Max Concurrency](./FPGAExtensions/LoopAttributes/max_concurrency)         | Control the concurrency of loops                |
| [Loop Ivdep](./FPGAExtensions/LoopAttributes/loop_ivdep)             | Assert lack of loop-carried array dependencies using ivdep attribute   |
| [Loop Unroll](./FPGAExtensions/LoopAttributes/loop_unroll)            | Unrolling loop to improve concurrency           |
| [Pipes](./FPGAExtensions/Pipes/pipes)                                 | Basic usage for FPGA Pipes                      |
| [Array of Pipes](./FPGAExtensions/Pipes/pipe_array)                   | How to declare an array of Pipes                |
| [FPGA register](./FPGAExtensions/Other/fpga_register)                 | Use FPGA register to solve fanout issue         |
| [System Level Profiling](./FPGAExtensions/Other/system_profiling)     | Use the OpenCL Intercept Layer to profile at a system level.             |



### Best Practices
| Tutorial Name        | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| [Double Buffering](./BestPractices/double_buffering)      | Overlap kernel execution with buffer transfers and host processing.                  |
| [N-way Buffering](./BestPractices/n_way_buffering)      | Overlap kernel execution with buffer transfers and host processing when host-processing time exceeds kernel execution time.                   |
| [Local Memory Caching](./BestPractices/local_memory_cache) | Build a simple cache (implemented in FPGA registers) to store recently-accessed memory locations. |
| [Remove Loop Carried Dependency](./BestPractices/remove_loop_carried_dependency)      | Remove loop carried dependencies to improve throughput.                  |
| [Triangular Loop](./BestPractices/triangular_loop)      | Optimize triangular loops to improve their throughput.                  |


## Reference

- [oneAPI Programming Guide](https://software.intel.com/en-us/download/intel-oneapi-programming-guide)
