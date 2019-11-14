cmake_minimum_required(VERSION 3.5)
project(decode_sycl_invert)

set(TARGET decode_sycl_invert)
set(SRCS src/decode_sycl_invert.cpp)
set(RUNARGS
    avc
    "$ENV{VPL_DIR}/samples/content/cars_1920x1080.h264"
    screen
    1920
    1080
    gpu
    100)

if(WIN32)
  set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
  set(CMAKE_CXX_COMPILER "dpcpp")
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

message(STATUS "Check for required environment variables")
if(NOT DEFINED ENV{VPLLIB})
  message(FATAL_ERROR "Expected environment variable not defined: VPLLIB")
endif()
if(NOT DEFINED ENV{VPLINCLUDE})
  message(FATAL_ERROR "Expected environment variable not defined: VPLINCLUDE")
endif()

message(STATUS "Check for presence of vpl library")
find_library(VPL_LIB vpl HINTS $ENV{VPLLIB})
if(NOT VPL_LIB)
  message(FATAL_ERROR "vpl library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using vpl library: ${VPL_LIB}")
endif()

message(STATUS "Check for presence of vplmemory library")
find_library(VPLMEMORY_LIB vplmemory HINTS $ENV{VPLLIB})
if(NOT VPLMEMORY_LIB)
  message(FATAL_ERROR "vplmemory library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using vplmemory library: ${VPLMEMORY_LIB}")
endif()

message(STATUS "Check for presence of OpenCV library")
if(WIN32)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    find_library(VPL_OPENCV_LIB opencv_world opencv_world412d HINTS $ENV{VPLLIB})
  else()
    find_library(VPL_OPENCV_LIB opencv_world opencv_world412 HINTS $ENV{VPLLIB})
  endif()
else()
  find_library(VPL_OPENCV_LIB opencv_world opencv_world412 HINTS $ENV{VPLLIB})
endif()

if(NOT VPL_OPENCV_LIB)
  message(FATAL_ERROR "opencv_world library not found. Looked in $ENV{VPLLIB}")
else()
  message(STATUS "Using opencv_world library: ${VPL_OPENCV_LIB}")
endif()

add_executable(${TARGET} ${SRCS})

target_include_directories(${TARGET} PRIVATE $ENV{VPLINCLUDE})
target_link_libraries(${TARGET} ${VPL_LIB} ${VPLMEMORY_LIB} ${VPL_OPENCV_LIB})

add_custom_target(run ${TARGET} ${RUNARGS})
