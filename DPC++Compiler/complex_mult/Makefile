CXX = dpcpp
CXXFLAGS = -g -o
LDFLAGS = -lOpenCL -lsycl
EXE_NAME = complex_mult.exe
SOURCES = src/complex_mult.cpp

all: main

main:
	$(CXX) $(CXXFLAGS) $(EXE_NAME) $(SOURCES) $(LDFLAGS)

run: 
	./$(EXE_NAME)

clean: 
	rm -rf $(EXE_NAME)
