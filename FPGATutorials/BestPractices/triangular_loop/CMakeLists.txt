if(WIN32)
  set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
  set(CMAKE_CXX_COMPILER "dpcpp")
endif()

cmake_minimum_required (VERSION 2.8)

project(TriangularLoop)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory (src)
