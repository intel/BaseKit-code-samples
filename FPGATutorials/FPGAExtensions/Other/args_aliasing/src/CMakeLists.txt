set(SOURCE_FILE no_accessor_aliasing.cpp)
set(TARGET_NAME args_aliasing)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET_ALIAS ${TARGET_NAME}_alias.fpga)
set(FPGA_TARGET_NOALIAS ${TARGET_NAME}_noalias.fpga)


set(HARDWARE_COMPILE_FLAGS "-fintelfpga")

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS "-fintelfpga -Xshardware ${USER_HARDWARE_FLAGS}")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
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
    add_executable(${FPGA_TARGET_ALIAS} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_executable(${FPGA_TARGET_NOALIAS} EXCLUDE_FROM_ALL ${SOURCE_FILE})
    add_custom_target(fpga DEPENDS ${FPGA_TARGET_ALIAS} ${FPGA_TARGET_NOALIAS})
    set_target_properties(${FPGA_TARGET_ALIAS} PROPERTIES COMPILE_FLAGS ${HARDWARE_COMPILE_FLAGS})
    set_target_properties(${FPGA_TARGET_NOALIAS} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS} -DFLAG=1")
    set_target_properties(${FPGA_TARGET_ALIAS} PROPERTIES LINK_FLAGS ${HARDWARE_LINK_FLAGS})
    set_target_properties(${FPGA_TARGET_NOALIAS} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -Xsno-accessor-aliasing")
endif()

# report

if(WIN32)
    set(REPORT_ALIAS ${TARGET_NAME}_alias_report.a)
    set(REPORT_NOALIAS ${TARGET_NAME}_noalias_report.a)
    add_custom_target(report DEPENDS ${REPORT_ALIAS} ${REPORT_NOALIAS})

    separate_arguments(HARDWARE_LINK_FLAGS_LIST WINDOWS_COMMAND "${HARDWARE_LINK_FLAGS}")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${CMAKE_BINARY_DIR}/${TARGET_NAME}_alias_report/${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${CMAKE_BINARY_DIR}/${TARGET_NAME}_noalias_report/${SOURCE_FILE} COPYONLY)

    add_custom_command(OUTPUT ${REPORT_ALIAS}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${CMAKE_BINARY_DIR}/${TARGET_NAME}_alias_report/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT_ALIAS}
                 DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${REPORT_NOALIAS}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${CMAKE_BINARY_DIR}/${TARGET_NAME}_noalias_report/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT_NOALIAS} -Xsno-accessor-aliasing
                 DEPENDS ${SOURCE_FILE})

else()
    set(REPORT_ALIAS ${TARGET_NAME}_alias_report.a)
    set(REPORT_NOALIAS ${TARGET_NAME}_noalias_report.a)
    add_custom_target(report DEPENDS ${REPORT_ALIAS} ${REPORT_NOALIAS})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)

    separate_arguments(HARDWARE_LINK_FLAGS_LIST UNIX_COMMAND "${HARDWARE_LINK_FLAGS}")
    add_custom_command(OUTPUT ${REPORT_ALIAS}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT_ALIAS}
                 DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${REPORT_NOALIAS}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS_LIST} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORT_NOALIAS} -Xsno-accessor-aliasing
                 DEPENDS ${SOURCE_FILE})
endif()


# run
add_custom_target(run
            COMMAND ../${TARGET_NAME}.fpga_emu
            DEPENDS ${TARGET_NAME}.fpga_emu)
