CXX = dpcpp
CXXFLAGS = -g -o
LDFLAGS = -lOpenCL -lsycl
EXE_NAME = simple-vector-incr
SOURCES = src/simple-vector-incr.cpp
BINDIR = bin

all: main

main:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(CXXFLAGS) $(BINDIR)/$(EXE_NAME) $(SOURCES) $(LDFLAGS)

run: 
	$(BINDIR)/$(EXE_NAME)

.PHONY: clean
clean: 
	rm -rf $(BINDIR)/$(EXE_NAME)
