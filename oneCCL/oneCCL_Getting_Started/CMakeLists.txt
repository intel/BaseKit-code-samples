cmake_minimum_required (VERSION 2.8)
if("${CMAKE_CXX_COMPILER}" STREQUAL "")
	set(CMAKE_C_COMPILER "clang")
	set(CMAKE_CXX_COMPILER "dpcpp")
endif()
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()
project (oneCCL_Getting_Started)
add_subdirectory (src out)
