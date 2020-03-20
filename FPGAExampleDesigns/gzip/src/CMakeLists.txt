set(DEVICE_SOURCE_FILE gzipkernel.cpp)
set(DEVICE_HEADER_FILE gzipkernel.h) 
set(HOST_SOURCE_FILE gzip.cpp crc32.cpp WriteGzip.cpp CompareGzip.cpp)

set(TARGET_NAME gzip)
set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)
set(REPORTS_TARGET ${TARGET_NAME}_report)


#specify -MMD -fsycl-link-targets=... instead of -fintelfpga to workaround known issue; lower report quality 
set(HARDWARE_COMPILE_FLAGS -MMD -fsycl-link-targets=spir64_fpga-unknown-unknown-sycldevice -c)

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
separate_arguments(USER_HARDWARE_FLAGS)
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware -Xsno-accessor-aliasing -Xsfmax=280 -Xsparallel=2 -Xsseed=1 ${USER_HARDWARE_FLAGS})
set(FINAL_LINK_FLAGS -fintelfpga)

set(EMULATOR_COMPILE_FLAGS "-g0 -fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS -fintelfpga)

# fpga emulator
if(WIN32)
    set(WIN_EMULATOR_TARGET ${EMULATOR_TARGET}.exe)
    add_custom_target(fpga_emu DEPENDS ${WIN_EMULATOR_TARGET})
    separate_arguments(WIN_EMULATOR_COMPILE_FLAGS WINDOWS_COMMAND "${EMULATOR_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_EMULATOR_TARGET} 
                   COMMAND ${CMAKE_CXX_COMPILER} ${WIN_EMULATOR_COMPILE_FLAGS} /GX ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/gzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/crc32.cpp ${CMAKE_CURRENT_SOURCE_DIR}/WriteGzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/CompareGzip.cpp  -o ${CMAKE_BINARY_DIR}/${WIN_EMULATOR_TARGET}
                   DEPENDS ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
else()
    add_executable(${EMULATOR_TARGET} ${DEVICE_SOURCE_FILE} ${HOST_SOURCE_FILE})
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS ${EMULATOR_COMPILE_FLAGS})
    set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS ${EMULATOR_LINK_FLAGS})
endif()

# fpga
if(WIN32)
    add_custom_target(fpga
                  COMMAND echo "FPGA hardware flow is not supported in Windows")
else()
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set(DEVICE_FPGA_OBJ "gzipkernel_fpga.o")
    set(DEVICE_IMAGE_FPGA_OBJ "gzipkernel_fpga.a")

    add_custom_command(OUTPUT ${DEVICE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_FPGA_OBJ}
                       DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_IMAGE_FPGA_OBJ}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link=image ${DEVICE_FPGA_OBJ} -o ${DEVICE_IMAGE_FPGA_OBJ}
                       DEPENDS ${DEVICE_FPGA_OBJ})

    add_custom_command(OUTPUT ${FPGA_TARGET}
                       COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${FINAL_LINK_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/gzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/crc32.cpp ${CMAKE_CURRENT_SOURCE_DIR}/WriteGzip.cpp ${CMAKE_CURRENT_SOURCE_DIR}/CompareGzip.cpp ${DEVICE_IMAGE_FPGA_OBJ} -o  ${CMAKE_BINARY_DIR}/${FPGA_TARGET}
                       DEPENDS ${DEVICE_IMAGE_FPGA_OBJ} ${HOST_SOURCE_FILE})
endif()

# fpga report
if(WIN32)
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )

    separate_arguments(WIN_FLAGS WINDOWS_COMMAND)
    add_custom_command(OUTPUT ${REPORTS_TARGET}
        COMMAND ${CMAKE_CXX_COMPILER} /EHsc ${WIN_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE})

else()
    add_custom_target(report DEPENDS ${REPORTS_TARGET} )

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} ${DEVICE_SOURCE_FILE} COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kernels.h kernels.h COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_HEADER_FILE} ${DEVICE_HEADER_FILE} COPYONLY)

    add_custom_command(OUTPUT ${REPORTS_TARGET}
                      COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link ${DEVICE_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${REPORTS_TARGET}
                      DEPENDS ${DEVICE_SOURCE_FILE} ${DEVICE_HEADER_FILE} kernels.h)
endif()

# run
add_custom_target(run
                  COMMAND ../${TARGET_NAME}.fpga_emu Makefile -o=test.gz
                  DEPENDS ${TARGET_NAME}.fpga_emu)

