set(SOURCE_FILE use_library.cpp)
set(TARGET_NAME use_library)
set(EMULATOR_TARGET ${TARGET_NAME}_emu.fpga)
set(CPU_HOST_TARGET ${TARGET_NAME}.cpu_host)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

set(LIBRARY_DEVICE_LINK_FLAGS "-XsL${CMAKE_CURRENT_SOURCE_DIR} -Xsllib.fpgalib -Xsoverride-library-version")
set(LIBRARY_HOST_LINK_FLAGS "${CMAKE_CURRENT_SOURCE_DIR}/lib.o")

set(HARDWARE_COMPILE_FLAGS "-fintelfpga")

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware ${LIBRARY_DEVICE_LINK_FLAGS} ${LIBRARY_HOST_LINK_FLAGS} ${USER_HARDWARE_FLAGS}")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga ${LIBRARY_DEVICE_LINK_FLAGS} ${LIBRARY_HOST_LINK_FLAGS}")

set(CPU_HOST_COMPILE_FLAGS "-DCPU_HOST")
set(CPU_HOST_LINK_FLAGS "${LIBRARY_HOST_LINK_FLAGS}")

# fpga emulator
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})

# cpu host
add_executable(${CPU_HOST_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(cpu_host DEPENDS ${CPU_HOST_TARGET})
set_target_properties(${CPU_HOST_TARGET} PROPERTIES COMPILE_FLAGS ${CPU_HOST_COMPILE_FLAGS})
set_target_properties(${CPU_HOST_TARGET} PROPERTIES LINK_FLAGS ${CPU_HOST_LINK_FLAGS})

# fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})

# report
set(DEVICE_OBJ_FILE ${TARGET_NAME}_report.a)
add_custom_target(report DEPENDS ${DEVICE_OBJ_FILE})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lib.h lib.h COPYONLY)

separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
add_custom_command(OUTPUT ${DEVICE_OBJ_FILE} 
                   COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${DEVICE_OBJ_FILE}
                   DEPENDS ${SOURCE_FILE} lib.h)

# run
add_custom_target(run
                  COMMAND ../${TARGET_NAME}_emu.fpga
                  DEPENDS ${TARGET_NAME}_emu.fpga)
