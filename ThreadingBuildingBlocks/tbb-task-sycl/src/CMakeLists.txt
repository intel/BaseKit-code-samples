set(CMAKE_CXX_COMPILER "dpcpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsycl -std=c++11 -O2 -cl-opt-disable")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -ltbb -lOpenCL -lsycl")
add_executable (tbb-task-sycl tbb-task-sycl.cpp)
add_custom_target (run ./tbb-task-sycl)

