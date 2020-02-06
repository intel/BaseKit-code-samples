CXX = dpcpp
CXXFLAGS = -g -o
LDFLAGS = -lOpenCL -lsycl
EXE_NAME = poly_integral.exe
SOURCES = src/Polynomial_Integral.cpp

all: main

main:
	$(CXX) $(CXXFLAGS) $(EXE_NAME) $(SOURCES) $(LDFLAGS)

run: 
	./$(EXE_NAME)

clean: 
	rm -rf $(EXE_NAME)
