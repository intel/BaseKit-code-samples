cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_COMPILER "dpcpp")
project (array-transform LANGUAGES CXX)

if (NOT CMAKE_CXX_STANDARD)
  set (CMAKE_CXX_STANDARD 11)
endif ()

# This project requires the "Debug" build type.
if (NOT CMAKE_BUILD_TYPE MATCHES Debug)
  set (CMAKE_BUILD_TYPE Debug)
  message (WARNING "Build Type is changed to 'Debug'.")
endif ()

# Check for the clBuildProgram flags.
if (NOT $ENV{SYCL_PROGRAM_COMPILE_OPTIONS} MATCHES "-g -cl-opt-disable")
  message (WARNING "Set the SYCL_PROGRAM_COMPILE_OPTIONS environment "
    "variable to '-g -cl-opt-disable' for a proper debug session.")
endif ()

# Check that debugger executables exist in PATH
find_program(GDB gdb-oneapi)
if (NOT GDB)
  message (FATAL_ERROR "'gdb-oneapi' not found!")
endif ()

find_program(GDBSERVER gdbserver-gt)
if (NOT GDBSERVER)
  message (FATAL_ERROR "'gdbserver-gt' not found!")
endif ()

add_executable (${PROJECT_NAME} src/array-transform.cpp)
target_compile_options (${PROJECT_NAME} PRIVATE -fsycl -g -O0)

# Add custom target for starting a debug session
add_custom_target (debug-session
  ${CMAKE_COMMAND} -E env SYCL_PROGRAM_COMPILE_OPTIONS='-g -cl-opt-disable' ${GDB} ./${PROJECT_NAME}
  DEPENDS ${PROJECT_NAME})

# Add custom target for running
add_custom_target(run ./${PROJECT_NAME} cpu
  DEPENDS ${PROJECT_NAME})

add_custom_target(run-cpu ./${PROJECT_NAME} cpu
  DEPENDS ${PROJECT_NAME})

add_custom_target(run-gpu ./${PROJECT_NAME} gpu
  DEPENDS ${PROJECT_NAME})
