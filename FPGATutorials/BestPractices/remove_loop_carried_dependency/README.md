# Tutorial: Removing loop carried data dependency in nested loops

This tutorial demonstrates how to remove a loop-carried dependency to improve the performance of FPGA device code. In the device code 'Unoptimized' in the file loop_carried_dependency.cpp, a sum is computed over two loops.  The inner loop sums over the 'A' data and the outer loop over the 'B' data. Since the value **sum** is updated in both loops, this introduces a 'loop carried dependency' that causes the outer loop to be serialized, allowing only one invocation of the outer loop to be active at a time, reducing performance.

The device code 'Optimized' demonstrates the use of an independent variable **sum2** that is not updated in the outer loop, and removes the need to serialize the outer loop, improving performance.

## When to Use This Technique

Look at the Compiler Report --> Throughput Analysis --> Loop Analysis section. The report lists the II of all loops and details on the loops. This technique may be  applicable if the **Brief Info** for the loop shows 'Serial exe: Data dependency'.  The details pane may provide mode information:
* Iteration executed serially across _function.block_. Only a single loop iteration will execute inside this region due to data dependency on variable(s):
  *   sum (_filename:line_)

## Implementation Notes

Compile the example with this command:
* dpcpp  -fintelfpga -Xshardware -fsycl-link loop_carried_dependency.cpp -o report

Open the report, located here:
> report.prj/reports/report.html

Navigate to the "Loops analysis" view of the report (under _Throughput Analysis_) and observe that the loop in block **UnOptKernel.B1** is showing _Serial exe: Data dependency_.  Click on the _source location_ field in the table to see the details for the loop. The maximum Interleaving iterations of the loop is 1, as the loop is serialized.

Observe the loop in block **OptKernel.B1** is not marked 'Serialized' .  The maximum Interleaving iterations of the loop is now 12.
## Sample Results
Running this compiled for a Intel Arria® 10 GX FPGA gave this output:
Number of elements: 16000
Run: Unoptimized:
kernel time : 10685.3 ms
Run: Optimized:
kernel time : 2736.47 ms
PASSED

The difference in execution time is a factor of almost 4.  The Initiation Interval (II) for the inner loop is 12 because a double Floating Point add takes 11 cycles.
