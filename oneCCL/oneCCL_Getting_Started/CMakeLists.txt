cmake_minimum_required (VERSION 2.8)
project (oneCCL_Getting_Started)

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

set(GCC_BFP16_MIN_SUPPORTED "4.9.0")

if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel" OR ${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS ${GCC_BFP16_MIN_SUPPORTED}))
    add_definitions(-DCCL_BFP16_COMPILER)
    set(CCL_BFP16_COMPILER ON)
    message(STATUS "BFP16 compiler: yes")
else()
    set(CCL_BFP16_COMPILER OFF)
    message(STATUS "BFP16 compiler: no")
endif()

if (CCL_BFP16_COMPILER)
    if ((${CMAKE_C_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
        add_definitions(-DCCL_BFP16_TARGET_ATTRIBUTES)
        message(STATUS "BFP16 target attributes: yes")
    else()
        message(STATUS "BFP16 target attributes: no")
    endif()
endif()

if("$ENV{EXAMPLE_ROOT}" STREQUAL "")
	message(" - use default examples")
	add_subdirectory (src out)
else()
	add_subdirectory ($ENV{EXAMPLE_ROOT} out)
endif()
