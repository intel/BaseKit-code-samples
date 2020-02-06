CXX = dpcpp
CXXFLAGS = -g -o
LDFLAGS = -lOpenCL -lsycl
EXE_NAME = bootstrapping
SOURCES = src/bootstrapping.cpp
BINDIR = bin

.SILENT: main
all: main

main:
	if [ ! -d "bin" ]; then \
        mkdir $(BINDIR); \
    fi
	$(CXX) $(CXXFLAGS) $(BINDIR)/$(EXE_NAME) $(SOURCES) $(LDFLAGS)

run: 
	$(BINDIR)/$(EXE_NAME)

.PHONY: clean
clean: 
	rm -rf $(BINDIR)/$(EXE_NAME)
