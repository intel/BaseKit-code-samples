//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include<vector>
#include<CL/sycl.hpp>
#include<iomanip>

using namespace cl::sycl;
using namespace std;
static const int N = 100;

//Defining the slice of the dx as 1000
#define d_x 1000

//An example polynomail function. ax^2+bx+c
float function_x(float x)
{
    return  (2*x*x* + 3*x* - 3*x/8 + 1/4);
}

//a and v are the arrays of the lower bound and the upper bounds of the x-axis. d is the result of the area after calculating the integral
//of the above Polynomial function
void dpcpp_parallel(float *a,float *v,float *d)
{
	try{
		// Setting up a queue to default DPC++ device selected by runtime
		queue device_queue;
		// Setup buffers for input and output vectors
		buffer<float, 1> bufv1(a,range<1>(N));
		buffer<float, 1> bufv2(v, range<1>(N));
		buffer<float, 1> bufv3(d, range<1>(N));
    
		auto start_time = std::chrono::high_resolution_clock::now();
		std::cout<<"Target Device: "<<device_queue.get_device().get_info<info::device::name>()<<"\n";
		//Submit Command group function object to the queue
		device_queue.submit([&](handler& cgh) {

		auto acc_vect1 = bufv1.get_access<access::mode::read>(cgh);
		auto acc_vect2 = bufv2.get_access<access::mode::read>(cgh);
		auto acc_vect3 = bufv3.get_access<access::mode::write>(cgh);
    
		cgh.parallel_for<class CompIntegral>(range<1>(N), [=](id<1> i) {

			float dx = (float) ((acc_vect1[i] - acc_vect2[i])/d_x);
			float area_int = 0;

			for(int j =0; j< d_x; j++)
			{
				float xC = acc_vect1[i] + dx * j;
				float yC = function_x(xC);
				area_int = xC * yC;
				area_int += area_int;
			} 
			acc_vect3[i] = area_int; 


			});
		});
		device_queue.wait();
		auto current_time = std::chrono::high_resolution_clock::now();
		std::cout << "Parallel: Program has been running for " << std::chrono::duration<double>(current_time - start_time).count() << " seconds" << std::endl;
		}
	 catch (cl::sycl::exception e) {
		 std::cout << "SYCL exception caught: " << e.what() << std::endl;
	 }
} 


int main(){
    float lBound[N],uBound[N],i_area[N],d[N];

	//Initialize the lower bound and upper bound of the x axis arrays. Below we are initializing such that upper bound is always greater than the lower bound.
    for (int i=0;i<N;i++)
    {
        lBound[i] = i+40 + 10;
        uBound[i] = (i+1)*40 + 70;
        i_area[i] = 0;
        d[i] = 0;
    }
	//Call the dpcpp_parallel function with lBound and uBound as inputs and i_area array as the output
    dpcpp_parallel(lBound,uBound,i_area);
    cout<<"****************************************Calculating Integral area in Parallel********************************************************"<<std::endl;
    for(int i=0;i<N;i++)
    {
        cout<<"Area: "<<i_area[i]<<' ';
        if(i==N-1)
        {
            cout<<"\n"<<"\n";
        }
    }
    return 0;
}