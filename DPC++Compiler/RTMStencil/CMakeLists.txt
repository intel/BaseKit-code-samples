if(WIN32)
set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
set(CMAKE_CXX_COMPILER "dpcpp")
endif()
cmake_minimum_required (VERSION 3.0)
project (RTMStencil)
add_subdirectory (src)

