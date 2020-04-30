cmake_minimum_required(VERSION 3.5)
project(py_decode_vpp)

set(SCRIPT ${CMAKE_SOURCE_DIR}/src/decode_vpp.py)
set(CMAKE_MSVCIDE_RUN_PATH $ENV{PATH})
set(RUNARGS $ENV{VPL_DIR}/samples/content/cars_1280x720.h264)

find_package (Python3 COMPONENTS Interpreter)
add_custom_target(run ${Python3_EXECUTABLE} ${SCRIPT} ${RUNARGS})
