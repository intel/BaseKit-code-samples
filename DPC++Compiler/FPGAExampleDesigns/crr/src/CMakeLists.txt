set(SOURCE_FILE main.cpp)
set(TARGET_NAME crr)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(CPU_HOST_TARGET ${TARGET_NAME}.cpu_host)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)

set(HARDWARE_COMPILE_FLAGS -fintelfpga -c)

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware -Xsno-accessor-aliasing -Xsdaz -Xsrounding=faithful -Xsparallel=2 -Xsseed=2 ${USER_HARDWARE_FLAGS})

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")

set(CPU_HOST_COMPILE_FLAGS "-DCPU_HOST")

#copy input data
configure_file("data/ordered_inputs.csv" "data/ordered_inputs.csv" COPYONLY)

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

# fpgas
if(WIN32)
    add_custom_target(fpga
                  COMMAND echo "FPGA hardware flow is not supported in Windows")
else()
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set(DEVICE_FPGA_OBJ "crr_fpga.o")

    add_custom_command(OUTPUT ${DEVICE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${DEVICE_FPGA_OBJ}
                       DEPENDS ${SOURCE_FILE})

    add_custom_command(OUTPUT ${FPGA_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS} ${DEVICE_FPGA_OBJ} -o  ${CMAKE_BINARY_DIR}/${FPGA_TARGET}
                   DEPENDS ${DEVICE_FPGA_OBJ})
endif()
                   
# fpga report
if(WIN32)
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )

    separate_arguments(WIN_FLAGS WINDOWS_COMMAND)
    add_custom_command(OUTPUT ${REPORTS_TARGET}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${CMAKE_CXX_FLAGS} ${WIN_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${SOURCE_FILE})

else()
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} ${SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CRR_common.h CRR_common.h COPYONLY)

    add_custom_command(OUTPUT ${REPORTS_TARGET}
                      COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link ${SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${SOURCE_FILE} CRR_common.h)
endif()

# run
add_custom_target(run
                  COMMAND ../${TARGET_NAME}.fpga_emu data/ordered_inputs.csv -o=data/ordered_output.csv
                  DEPENDS ${TARGET_NAME}.fpga_emu)

