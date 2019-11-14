CXX = dpcpp
CXXFLAGS = -o
LDFLAGS = -lOpenCL -lsycl
EXE_NAME = bootstrapping
SOURCES = src/bootstrapping.cpp
BINDIR = bin

all: main

main:
	mkdir $(BINDIR)
	$(CXX) $(CXXFLAGS) $(BINDIR)/$(EXE_NAME) $(SOURCES) $(LDFLAGS)

run: 
	$(BINDIR)/$(EXE_NAME)

clean: 
	rm -rf $(BINDIR)/$(EXE_NAME)
