set(SOURCE_FILE fpga_reg.cpp)
set(TARGET_NAME fpga_reg)
set(TARGET_NAME_REG fpga_reg_registered)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(CPU_HOST_TARGET ${TARGET_NAME}.cpu_host)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(FPGA_TARGET_REG ${TARGET_NAME_REG}.fpga)


set(HARDWARE_COMPILE_FLAGS "-fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware -Xsno-accessor-aliasing")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")

set(CPU_HOST_COMPILE_FLAGS "-DCPU_HOST")

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

# cpu host
if(WIN32)
    set(WIN_HOST_TARGET ${CPU_HOST_TARGET}.exe)
    add_custom_target(cpu_host DEPENDS ${WIN_HOST_TARGET})
    separate_arguments(WIN_HOST_COMPILE_FLAGS WINDOWS_COMMAND "${CPU_HOST_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_HOST_TARGET} 
             COMMAND ${CMAKE_CXX_COMPILER} ${WIN_HOST_COMPILE_FLAGS} /GX ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${WIN_HOST_TARGET}
             DEPENDS ${SOURCE_FILE})      

else()
    add_executable(${CPU_HOST_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_custom_target(cpu_host DEPENDS ${CPU_HOST_TARGET})
    set_target_properties(${CPU_HOST_TARGET} PROPERTIES COMPILE_FLAGS ${CPU_HOST_COMPILE_FLAGS})
endif()


# fpga
if(WIN32)
    add_custom_target(fpga
            COMMAND echo "FPGA hardware flow is not supported in Windows")
else()
    add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_executable(${FPGA_TARGET_REG} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_custom_target(fpga DEPENDS ${FPGA_TARGET} ${FPGA_TARGET_REG})

    set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
    set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})

    set_target_properties(${FPGA_TARGET_REG} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DUSE_FPGA_REG")
    set_target_properties(${FPGA_TARGET_REG} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
endif()

# report
if(WIN32)
    set(REPORT ${TARGET_NAME}_report.a)
    set(REPORT_REG ${TARGET_NAME_REG}_report.a)

    add_custom_target(report DEPENDS ${REPORT} ${REPORT_REG})

    separate_arguments(HARDWARE_LINK_FLAGS_LIST WINDOWS_COMMAND "${HARDWARE_LINK_FLAGS}")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${CMAKE_BINARY_DIR}/${TARGET_NAME_REG}/${SOURCE_FILE} COPYONLY)

    add_custom_command(OUTPUT ${REPORT}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT}
                 DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${REPORT_REG}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -DUSE_FPGA_REG -fsycl-link ${CMAKE_BINARY_DIR}/${TARGET_NAME_REG}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT_REG}
                 DEPENDS ${SOURCE_FILE})

else()
    set(REPORT ${TARGET_NAME}_report.a)
    set(REPORT_REG ${TARGET_NAME_REG}_report.a)

    add_custom_target(report DEPENDS ${REPORT} ${REPORT_REG})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)

    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    add_custom_command(OUTPUT ${REPORT}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT}
                 DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${REPORT_REG}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -DUSE_FPGA_REG -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT_REG}
                 DEPENDS ${SOURCE_FILE})
endif()

# run
add_custom_target(run
            COMMAND ../${TARGET_NAME}.fpga_emu
            DEPENDS ${TARGET_NAME}.fpga_emu)
