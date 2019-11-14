# Copyright Intel Corporation 2019
#
# To compile with the GNU* C/C++ compiler, creating an execution file with the
# extension ".gcc" for binary instrumentation, issue:
#
#   > make
#
# To compile with the Intel(R) C++ Compiler, creating an execution
# file with the extension ".icc":
#
#   > make icc
#
# To compile with the Intel(R) C++ Compiler with Intel(R) MKL library
# creating an execution file with the extension ".mkl":
#
#   > make mkl
#
# To compile them all:
#
#   > make all

SHELL = /bin/sh

PARAMODEL = -DUSE_THR	# Default parallelism using pthreads/Win threads
#PARAMODEL = -DUSE_OMP -fopenmp	

GCC     = gcc
CLANG   = clang++
ICC     = icc
CFLAGS  = -g -O3 -fpermissive -w
OPTFLAGS = -xSSE3 
#OPTFLAGS = -xHost -fno-alias
# add -DALIGNED to the multiply.cpp and matrix.cpp
LDFLAGS = -lpthread -lm
LDCLFLAGS = -lOpenCL -lsycl -g  -lm

GCFLAGS = $(CFLAGS) $(PARAMODEL) 
CLFLAGS = $(CFLAGS) -fsycl -mavx2 -std=c++11 -DDPCPP
ICFLAGS = $(CFLAGS) $(PARAMODEL)-DICC -debug inline-debug-info #-vec-report3 -qopt-report -qopt-report-phase=vec
MKFLAGS = $(CFLAGS) -DUSE_MKL   -DICC -mkl -debug inline-debug-info


GCC_EXE = matrix.gcc
ICC_EXE = matrix.icc
CLANG_EXE = matrix.clang
MKL_EXE = matrix.mkl


srcdir = ../src

gcc: $(GCC_EXE)
icc: $(ICC_EXE)
clang: $(CLANG_EXE)
mkl: $(MKL_EXE)
all: $(GCC_EXE) $(ICC_EXE) $(CLANG_EXE) $(MKL_EXE)


OBJS = util.o thrmodel.o multiply.o matrix.o 


matrix.gcc: $(srcdir)/matrix.cpp $(srcdir)/multiply.cpp $(srcdir)/multiply.h $(srcdir)/util.cpp $(srcdir)/thrmodel.cpp
	$(GCC) $(GCFLAGS) -c $(srcdir)/util.cpp -D_LINUX
	$(GCC) $(GCFLAGS) -c $(srcdir)/thrmodel.cpp -D_LINUX
	$(GCC) $(GCFLAGS) -c $(srcdir)/multiply.cpp -D_LINUX
	$(GCC) $(GCFLAGS) -c $(srcdir)/matrix.cpp -D_LINUX
	$(GCC) $(GCFLAGS) -g $(OBJS) -o $@ $(LDFLAGS)

matrix.clang: $(srcdir)/matrix.cpp $(srcdir)/multiply.cpp $(srcdir)/multiply.h $(srcdir)/util.cpp $(srcdir)/thrmodel.cpp
	$(CLANG) $(CLFLAGS) -c $(srcdir)/util.cpp -D_LINUX
	$(CLANG) $(CLFLAGS) -c $(srcdir)/thrmodel.cpp -D_LINUX
	$(CLANG) $(CLFLAGS) -c $(srcdir)/multiply.cpp -D_LINUX 
	$(CLANG) $(CLFLAGS) -c $(srcdir)/matrix.cpp -D_LINUX
	$(CLANG) $(CLFLAGS) -g $(OBJS) -o $@ $(LDCLFLAGS)

matrix.icc: $(srcdir)/matrix.cpp $(srcdir)/multiply.cpp $(srcdir)/multiply.h $(srcdir)/util.cpp $(srcdir)/thrmodel.cpp
	$(ICC) $(ICFLAGS) -c $(srcdir)/util.cpp -D_LINUX
	$(ICC) $(ICFLAGS) -c $(srcdir)/thrmodel.cpp -D_LINUX
	$(ICC) $(ICFLAGS) $(OPTFLAGS) -c $(srcdir)/multiply.cpp -D_LINUX
	$(ICC) $(ICFLAGS) $(OPTFLAGS) -c $(srcdir)/matrix.cpp -D_LINUX
	$(ICC) $(ICFLAGS) $(OBJS) -o $@ $(LDFLAGS)

matrix.mkl:     $(srcdir)/matrix.cpp $(srcdir)/multiply.cpp $(srcdir)/multiply.h $(srcdir)/util.cpp $(srcdir)/thrmodel.cpp
	$(ICC) $(MKFLAGS) -c $(srcdir)/util.cpp -D_LINUX
	$(ICC) $(MKFLAGS) -c $(srcdir)/thrmodel.cpp -D_LINUX
	$(ICC) $(MKFLAGS) $(OPTFLAGS) -c $(srcdir)/multiply.cpp -D_LINUX
	$(ICC) $(MKFLAGS) $(OPTFLAGS) -c $(srcdir)/matrix.cpp -D_LINUX
	$(ICC) $(MKFLAGS) $(OBJS) -o $@ $(LDFLAGS)

clean:
	@rm -rf $(OBJS) $(GCC_EXE) $(ICC_EXE) $(MKL_EXE) *.o

# * Other names and brands may be claimed as the property of others.