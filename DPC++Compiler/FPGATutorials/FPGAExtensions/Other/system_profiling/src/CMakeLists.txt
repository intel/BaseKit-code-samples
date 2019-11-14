set(SOURCE_FILE double_buffering.cpp)
set(TARGET_NAME double_buffering)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(CPU_HOST_TARGET ${TARGET_NAME}.cpu_host)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

set(HARDWARE_COMPILE_FLAGS "-fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsno-accessor-aliasing")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")

set(CPU_HOST_COMPILE_FLAGS "-DCPU_HOST")

# fpga emulator
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})

# cpu host
add_executable(${CPU_HOST_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(cpu_host DEPENDS ${CPU_HOST_TARGET})
set_target_properties(${CPU_HOST_TARGET} PROPERTIES COMPILE_FLAGS ${CPU_HOST_COMPILE_FLAGS})

# fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})


# run
add_custom_target(run
                  COMMAND ../${TARGET_NAME}.fpga_emu
                  DEPENDS ${TARGET_NAME}.fpga_emu)