//==============================================================
// Copyright  2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifdef __MIC__
#define MAXTHREADS 240
#define NUM 3840
#define MATRIX_BLOCK_SIZE 16
#else
#define MAXTHREADS 16
#define NUM 1024
#define MATRIX_BLOCK_SIZE 64
#define MATRIX_TILE_SIZE 16
#define WPT 8
#endif

typedef float TYPE;
typedef TYPE array[NUM];

// Select which multiply kernel to use via the following macro so that the
// kernel being used can be reported when the test is run.
#define MULTIPLY multiply1

extern void multiply0(int msize, int tidx, int numt, TYPE a[][NUM],
                      TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1(int msize, int tidx, int numt, TYPE a[][NUM],
                      TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1_1(int msize, int tidx, int numt, TYPE a[][NUM],
                        TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1_2(int msize, int tidx, int numt, TYPE a[][NUM],
                        TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1_3(int msize, int tidx, int numt, TYPE a[][NUM],
                        TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply1_4(int msize, int tidx, int numt, TYPE a[][NUM],
                        TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply2(int msize, int tidx, int numt, TYPE a[][NUM],
                      TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply2_1(int msize, int tidx, int numt, TYPE a[][NUM],
                        TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply3(int msize, int tidx, int numt, TYPE a[][NUM],
                      TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply4(int msize, int tidx, int numt, TYPE a[][NUM],
                      TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);
extern void multiply5(int msize, int tidx, int numt, TYPE a[][NUM],
                      TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM]);

void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM],
                      TYPE t[][NUM]);
void GetModelParams(int* nthreads, int* msize, int print);
