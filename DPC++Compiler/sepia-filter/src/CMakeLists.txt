set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_executable (sepia sepia_dpcpp.cpp)
target_link_libraries(sepia OpenCL sycl)
if(WIN32)
add_custom_target (run sepia.exe)
else()
add_custom_target (run ./sepia)
endif()


