set(SOURCE_FILE pipe_array.cpp)
set(TARGET_NAME pipe_array)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

set(HARDWARE_COMPILE_FLAGS "-fintelfpga -std=c++14")

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware ${USER_HARDWARE_FLAGS}")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -std=c++14 -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")

# fpga emulator
if(WIN32)
      set(WIN_EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
      add_custom_target(fpga_emu DEPENDS ${WIN_EMULATOR_TARGET})
      separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
      add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET} 
                   COMMAND ${CMAKE_CXX_COMPILER} ${WIN_EMULATOR_COMPILE_FLAGS} /GX ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET}
                   DEPENDS ${SOURCE_FILE})

else()
      add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
      add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
      set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
      set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
endif()

# fpga
if(WIN32)
     add_custom_target(fpga
                  COMMAND echo "FPGA hardware flow is not supported in Windows")
else()
    add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
    set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
endif()

# report
if(WIN32)
    set(DEVICE_OBJ_FILE ${TARGET_NAME}_report.a)
    add_custom_target(report DEPENDS ${DEVICE_OBJ_FILE})

    separate_arguments(HARDWARE_LINK_FLAGS_LIST WINDOWS_COMMAND "${HARDWARE_LINK_FLAGS}")
    add_custom_command(OUTPUT ${DEVICE_OBJ_FILE} 
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -std=c++14 -fsycl-link ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${DEVICE_OBJ_FILE}
                       DEPENDS ${SOURCE_FILE})

else()
    set(DEVICE_OBJ_FILE ${TARGET_NAME}_report.a)
    add_custom_target(report DEPENDS ${DEVICE_OBJ_FILE})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_array.h pipe_array.h COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/unroller.h unroller.h COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pipe_array_internal.h pipe_array_internal.h COPYONLY)

    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    add_custom_command(OUTPUT ${DEVICE_OBJ_FILE} 
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -std=c++14 -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${DEVICE_OBJ_FILE}
                       DEPENDS ${SOURCE_FILE} pipe_array.h unroller.h pipe_array_internal.h)
endif()

# run
add_custom_target(run
                  COMMAND ../${TARGET_NAME}.fpga_emu
                  DEPENDS ${TARGET_NAME}.fpga_emu)
