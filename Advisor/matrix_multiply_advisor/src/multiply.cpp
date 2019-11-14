//==============================================================
// Copyright  2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// matrix multiply routines
#include "multiply.h"

#ifdef USE_MKL
#include "mkl.h"
#endif //USE_MKL

#ifdef DPCPP
#include <CL/sycl.hpp>
#include <array>
#endif
#include <stdlib.h>

#ifdef USE_THR
void multiply0(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
	int i,j,k;

// Basic serial implementation
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
    	    for(k=0; k<msize; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}

void multiply1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k;

// Naive implementation 
    for(i=tidx; i<msize; i=i+numt) {
        for(j=0; j<msize; j++) {
    	    for(k=0; k<msize; k++) {
					c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}
void multiply2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k;

// Step 2: Loop interchange
// Add compile option for vectorization report Windows: /Qvec-report3 Linux -vec-report3
	for(i=tidx; i<msize; i=i+numt) {
		for(k=0; k<msize; k++) {
#pragma ivdep
			for(j=0; j<msize; j++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}
void multiply3(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k,i0,j0,k0,ibeg,ibound,istep,mblock;

// Step3: Cache blocking
// Add current platform optimization for Windows: /QxHost Linux: -xHost
// Define the ALIGNED in the preprocessor definitions and compile option Windows: /Oa Linux: -fno-alias
    istep = msize / numt;
    ibeg = tidx * istep;
    ibound = ibeg + istep;
    mblock = MATRIX_BLOCK_SIZE;

    for (i0 = ibeg; i0 < ibound; i0 +=mblock) {
        for (k0 = 0; k0 < msize; k0 += mblock) {
            for (j0 =0; j0 < msize; j0 += mblock) {
                for (i = i0; i < i0 + mblock; i++) {
                    for (k = k0; k < k0 + mblock; k++) {
#pragma ivdep
#ifdef ALIGNED 
	#pragma vector aligned
#endif //ALIGNED
                        for (j = j0; j < j0 + mblock; j++) {
                            c[i][j]  = c[i][j] + a[i][k] * b[k][j];
						}
					}
				}
			}
		}
	}
}

void multiply4(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k,istep,ibeg,ibound;
//transpose b
	for(i=0;i<msize;i++) {
	    for(k=0;k<msize;k++) {
		t[i][k] = b[k][i];
		}
	}

    istep = msize / numt;
    ibeg = tidx * istep;
    ibound = ibeg + istep;
/*  for(i=0;i<msize;i+=4) { // use instead for single threaded impl.*/
	for(i=ibeg;i<ibound;i+=4) {
	    for(j=0;j<msize;j+=4) {
#pragma loop count (NUM)
#pragma ivdep
			for(k=0;k<msize;k++) {
				c[i][j] = c[i][j] + a[i][k] * t[j][k];
				c[i+1][j] = c[i+1][j] + a[i+1][k] * t[j][k];
				c[i+2][j] = c[i+2][j] + a[i+2][k] * t[j][k];
				c[i+3][j] = c[i+3][j] + a[i+3][k] * t[j][k];

				c[i][j+1] = c[i][j+1] + a[i][k] * t[j+1][k];
				c[i+1][j+1] = c[i+1][j+1] + a[i+1][k] * t[j+1][k];
				c[i+2][j+1] = c[i+2][j+1] + a[i+2][k] * t[j+1][k];
				c[i+3][j+1] = c[i+3][j+1] + a[i+3][k] * t[j+1][k];

				c[i][j+2] = c[i][j+2] + a[i][k] * t[j+2][k];
				c[i+1][j+2] = c[i+1][j+2] + a[i+1][k] * t[j+2][k];
				c[i+2][j+2] = c[i+2][j+2] + a[i+2][k] * t[j+2][k];
				c[i+3][j+2] = c[i+3][j+2] + a[i+3][k] * t[j+2][k];

				c[i][j+3] = c[i][j+3] + a[i][k] * t[j+3][k];
				c[i+1][j+3] = c[i+1][j+3] + a[i+1][k] * t[j+3][k];
				c[i+2][j+3] = c[i+2][j+3] + a[i+2][k] * t[j+3][k];
				c[i+3][j+3] = c[i+3][j+3] + a[i+3][k] * t[j+3][k];
		  	}
		}
	}

 /*
	// it's the same to the loop above?
	for(i=ibeg;i<ibound;i++) {
	    for(j=0;j<msize;j++) {

#pragma ivdep
#pragma vector aligned

			for(k=0;k<msize;k++) {
				c[i][j] = c[i][j] + a[i][k] * t[j][k];}}}
*/
}
#endif // USE_THR

//SYCL
#ifdef DPCPP
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

template <typename T>
class Matrix;
template <typename T>
class CMatrix;
template <typename T>
class Matrix1;
template <typename T>
class Matrix2;
template <typename T>
class Matrix3;
template <typename T>
class Matrix4;
template <typename T>
class CMatrix1;

//Basic matrix multiply
void multiply1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
	int i,j,k;
 
  // Select device
//#ifdef GPU
  //cl::sycl::gpu_selector device;
//#else
  //cl::sycl::host_selector device;
//#endif
  // Declare a deviceQueue
  cl::sycl::default_selector device;
  cl::sycl::queue deviceQueue(device);
  // Declare a 2 dimensional range
  cl::sycl::range<2> matrix_range{NUM, NUM};

  // Declare 3 buffers and Initialize them
  cl::sycl::buffer<TYPE, 2> bufferA((TYPE*)a, matrix_range);
  cl::sycl::buffer<TYPE, 2> bufferB((TYPE*)b, matrix_range);
  cl::sycl::buffer<TYPE, 2> bufferC((TYPE*)c, matrix_range);
  // Submit our job to the queue
  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    // Declare 3 accessors to our buffers. The first 2 read and the last read_write
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_read_write>(cgh);

    // Execute matrix multiply in parallel over our matrix_range
    // ind is an index into this range
    cgh.parallel_for<class Matrix<TYPE>>(matrix_range,
		[=](cl::sycl::id<2> ind) {
            int k;
    	    for(k=0; k<NUM; k++) {
	       // Perform computation ind[0] is row, ind[1] is col
               accessorC[ind[0]][ind[1]]  += accessorA[ind[0]][k] * accessorB[k][ind[1]];
            }
     });
  });

}

//Replaces accessorC reference with a local variable
void multiply1_1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
	int i,j,k;
 
  // Select device
//#ifdef GPU
  //cl::sycl::gpu_selector device;
//#else
  //cl::sycl::host_selector device;
//#endif
  // Declare a deviceQueue  
  cl::sycl::default_selector device;
  cl::sycl::queue deviceQueue(device);
  // Declare a 2 dimensional range
  cl::sycl::range<2> matrix_range{NUM, NUM};

  // Declare 3 buffers and Initialize them
  cl::sycl::buffer<TYPE, 2> bufferA((TYPE*)a, matrix_range);
  cl::sycl::buffer<TYPE, 2> bufferB((TYPE*)b, matrix_range);
  cl::sycl::buffer<TYPE, 2> bufferC((TYPE*)c, matrix_range);

  // Submit our job to the queue
  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    // Declare 3 accessors to our buffers. The first 2 read and the last read_write
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_read_write>(cgh);

    // Execute matrix multiply in parallel over our matrix_range
    // ind is an index into this range
    cgh.parallel_for<class Matrix1<TYPE>>(matrix_range,
		[=](cl::sycl::id<2> ind) {
            int k;
            TYPE acc = 0.0;
    	    for(k=0; k<NUM; k++) {
	       // Perform computation ind[0] is row, ind[1] is col
               acc += accessorA[ind[0]][k] * accessorB[k][ind[1]];
               //accessorC[ind[0]][ind[1]]  += accessorA[ind[0]][k] * accessorB[k][ind[1]];
            }
            accessorC[ind[0]][ind[1]] = acc;
     });
  });

}

//Replaces accessorC reference with a local variable and adds matrix tiling
void multiply1_2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
	int i,j,k;
 
  // Select device
//#ifdef GPU
  //cl::sycl::gpu_selector device;
//#else
  //cl::sycl::host_selector device;
//#endif
  // Declare a deviceQueue
  cl::sycl::default_selector device;
  cl::sycl::queue deviceQueue(device);
  // Declare a 2 dimensional range
  cl::sycl::range<2> matrix_range{NUM, NUM};
  cl::sycl::range<2> tile_range{MATRIX_TILE_SIZE, MATRIX_TILE_SIZE};

  // Declare 3 buffers and Initialize them
  cl::sycl::buffer<TYPE, 2> bufferA((TYPE*)a, matrix_range);
  cl::sycl::buffer<TYPE, 2> bufferB((TYPE*)b, matrix_range);
  cl::sycl::buffer<TYPE, 2> bufferC((TYPE*)c, matrix_range);

  // Submit our job to the queue
  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    // Declare 3 accessors to our buffers. The first 2 read and the last read_write
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_read_write>(cgh);

	//Create matrix tiles
    cl::sycl::accessor<TYPE, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> aTile(cl::sycl::range<2>(MATRIX_TILE_SIZE, MATRIX_TILE_SIZE), cgh);
    cl::sycl::accessor<TYPE, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> bTile(cl::sycl::range<2>(MATRIX_TILE_SIZE, MATRIX_TILE_SIZE), cgh);
    // Execute matrix multiply in parallel over our matrix_range
    // ind is an index into this range
    cgh.parallel_for<class Matrix2<TYPE>>(cl::sycl::nd_range<2>(matrix_range,tile_range),
		[=](cl::sycl::nd_item<2> it) {
            int k;
            const int numTiles = NUM/MATRIX_TILE_SIZE;
            const int row = it.get_local_id(0);
            const int col = it.get_local_id(1);
            const int globalRow = MATRIX_TILE_SIZE*it.get_group(0) + row;
            const int globalCol = MATRIX_TILE_SIZE*it.get_group(1) + col;
            TYPE acc = 0.0;
            for (int t=0; t<numTiles; t++) {
                 const int tiledRow = MATRIX_TILE_SIZE*t + row;
                 const int tiledCol = MATRIX_TILE_SIZE*t + col;
                 aTile[row][col] = accessorA[globalRow][tiledCol];
                 bTile[row][col] = accessorB[tiledRow][globalCol];
                 it.barrier(cl::sycl::access::fence_space::local_space);
                 for(k=0; k<MATRIX_TILE_SIZE; k++) {
                     // Perform computation ind[0] is row, ind[1] is col
                     acc += aTile[row][k] * bTile[k][col];
                 }
                 it.barrier(cl::sycl::access::fence_space::local_space);
            }
            accessorC[globalRow][globalCol] = acc;
     });
  });

}

//Cache-blocked matrix multiply using sub-ranges
void multiply2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k,i0,j0,k0,ibeg,ibound,istep,mblock;
  // Select device
//#ifdef GPU
  //cl::sycl::gpu_selector device;
//#else
  //cl::sycl::host_selector device;
//#endif
  // Declare a deviceQueue
  cl::sycl::default_selector device;
  cl::sycl::queue deviceQueue(device);
// Step3: Cache blocking
// Add current platform optimization for Windows: /QxHost Linux: -xHost
// Define the ALIGNED in the preprocessor definitions and compile option Windows: /Oa Linux: -fno-alias
    mblock = MATRIX_BLOCK_SIZE;
    cl::sycl::range<2> matrix_range{NUM, NUM};
    //Declare a 2 dimensional range which is a block of our matrix
    cl::sycl::range<2> sub_range{MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE};

    // Declare 3 buffers which will access 1 block of our matrix
    cl::sycl::buffer<TYPE, 2> bufferA((TYPE*)&(a[i0][j0]), matrix_range);
    cl::sycl::buffer<TYPE, 2> bufferB((TYPE*)&b[i0][j0], matrix_range);
    cl::sycl::buffer<TYPE, 2> bufferC((TYPE*)&c[i0][j0], matrix_range);
    /// Declare 3 outer loops which break down our matrix into blocks
    for (i0 = 0; i0 < msize; i0 +=mblock) {
        for (j0 =0; j0 < msize; j0 += mblock) {
            for (k0 = 0; k0 < msize; k0 += mblock) {
		    cl::sycl::buffer<TYPE, 2> subbufa(bufferA, cl::sycl::id<2>(i0,k0), cl::sycl::range<2>(MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE));    
		    cl::sycl::buffer<TYPE, 2> subbufb(bufferB, cl::sycl::id<2>(k0,j0), cl::sycl::range<2>(MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE));    
		    cl::sycl::buffer<TYPE, 2> subbufc(bufferC, cl::sycl::id<2>(i0,j0), cl::sycl::range<2>(MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE));    
                deviceQueue.submit([&](cl::sycl::handler& cgh) {
                   auto accessorA = subbufa.template get_access<sycl_read>(cgh);
                   auto accessorB = subbufb.template get_access<sycl_read>(cgh);
                   auto accessorC = subbufc.template get_access<sycl_read_write>(cgh);

		   // Execute a matrix multiply in parallel
                   cgh.parallel_for<class CMatrix<TYPE>>(sub_range,
		       [=](cl::sycl::id<2> ind) {
                           int k;
    	                   for(k=0; k<MATRIX_BLOCK_SIZE; k++) {
			      // Perform computation ind[0] is row, ind[1] is col
                              accessorC[ind[0]][ind[1]] += accessorA[ind[0]][k] * accessorB[k][ind[1]];
                           }
                       });
                });
			}
		}
	}
}

//Cache-blocked matrix multiply using sub-ranges. Also replaces accessorC reference with a local variable
void multiply2_1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k,i0,j0,k0,ibeg,ibound,istep,mblock;
  // Select device
//#ifdef GPU
  //cl::sycl::gpu_selector device;
//#else
  //cl::sycl::host_selector device;
//#endif
  // Declare a deviceQueue
  cl::sycl::default_selector device;
  cl::sycl::queue deviceQueue(device);

// Step3: Cache blocking
// Add current platform optimization for Windows: /QxHost Linux: -xHost
// Define the ALIGNED in the preprocessor definitions and compile option Windows: /Oa Linux: -fno-alias
    mblock = MATRIX_BLOCK_SIZE;
    cl::sycl::range<2> matrix_range{NUM, NUM};
    //Declare a 2 dimensional range which is a block of our matrix
    cl::sycl::range<2> sub_range{MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE};

    // Declare 3 buffers which will access 1 block of our matrix
    cl::sycl::buffer<TYPE, 2> bufferA((TYPE*)&(a[i0][j0]), matrix_range);
    cl::sycl::buffer<TYPE, 2> bufferB((TYPE*)&b[i0][j0], matrix_range);
    cl::sycl::buffer<TYPE, 2> bufferC((TYPE*)&c[i0][j0], matrix_range);
    /// Declare 3 outer loops which break down our matrix into blocks
    for (i0 = 0; i0 < msize; i0 +=mblock) {
        for (j0 =0; j0 < msize; j0 += mblock) {
            for (k0 = 0; k0 < msize; k0 += mblock) {
		    cl::sycl::buffer<TYPE, 2> subbufa(bufferA, cl::sycl::id<2>(i0,k0), cl::sycl::range<2>(MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE));    
		    cl::sycl::buffer<TYPE, 2> subbufb(bufferB, cl::sycl::id<2>(k0,j0), cl::sycl::range<2>(MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE));    
		    cl::sycl::buffer<TYPE, 2> subbufc(bufferC, cl::sycl::id<2>(i0,j0), cl::sycl::range<2>(MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE));    
                deviceQueue.submit([&](cl::sycl::handler& cgh) {
                   auto accessorA = subbufa.template get_access<sycl_read>(cgh);
                   auto accessorB = subbufb.template get_access<sycl_read>(cgh);
                   auto accessorC = subbufc.template get_access<sycl_read_write>(cgh);

		   // Execute a matrix multiply in parallel
                   cgh.parallel_for<class CMatrix1<TYPE>>(sub_range,
		       [=](cl::sycl::id<2> ind) {
                           int k;
                           TYPE acc = accessorC[ind[0]][ind[1]];
    	                   for(k=0; k<MATRIX_BLOCK_SIZE; k++) {
			      // Perform computation ind[0] is row, ind[1] is col
                              acc += accessorA[ind[0]][k] * accessorB[k][ind[1]];
                           }
                           accessorC[ind[0]][ind[1]] = acc;
                       });
                });
			}
		}
	}
}

#endif


#ifdef USE_OMP

void multiply0(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
	int i,j,k;

// Basic serial implementation
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
    	    for(k=0; k<msize; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}

void multiply1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
	int i,j,k;

	// Basic parallel implementation
	#pragma omp parallel for
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
    	    for(k=0; k<msize; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}

void multiply2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k;

	// Parallel with merged outer loops
	#pragma omp parallel for collapse (2)
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
    	    for(k=0; k<msize; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}
void multiply3(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k;

	#pragma omp parallel for collapse (2)
	for(i=0; i<msize; i++) {
		for(k=0; k<msize; k++) {
#pragma ivdep
			for(j=0; j<msize; j++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	}
}
void multiply4(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k;

	#pragma omp parallel for collapse (2)
	for(i=0; i<msize; i++) {
		for(k=0; k<msize; k++) {
#pragma unroll(8)
#pragma ivdep
			for(j=0; j<msize; j++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	}
}

#endif // USE_OMP

#ifdef USE_MKL
// DGEMM way of matrix multiply using Intel MKL
// Link with Intel MKL library: With MSFT VS and Intel Composer integration: Select build components in the Project context menu.
// For command line - check out the Intel Math Kernel Library Link Line Advisor 
void multiply5(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{

	double alpha = 1.0, beta = 0.;
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NUM,NUM,NUM,alpha,(const double *)b,NUM,(const double *)a,NUM,beta,(double *)c,NUM);
}
#endif //USE_MKL
