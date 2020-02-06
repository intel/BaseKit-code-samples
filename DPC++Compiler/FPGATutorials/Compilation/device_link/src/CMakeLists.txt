set(DEVICE_SOURCE_FILE kernel.cpp)
set(KERNEL_HEADER_FILE kernel.h)
set(HOST_SOURCE_FILE main.cpp)
set(TARGET_NAME device_link)

set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(CPU_HOST_TARGET ${TARGET_NAME}.cpu_host)
set(FPGA_TARGET ${TARGET_NAME}.fpga)

set(HARDWARE_COMPILE_FLAGS -fintelfpga -c)

# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation
set(HARDWARE_LINK_FLAGS -fintelfpga -Xshardware ${USER_HARDWARE_FLAGS})

set(EMULATOR_COMPILE_FLAGS -fintelfpga -DFPGA_EMULATOR -c)
set(EMULATOR_LINK_FLAGS -fintelfpga)

set(CPU_HOST_COMPILE_FLAGS "-DCPU_HOST")

# cpu host
if(WIN32)
    set(WIN_HOST_TARGET ${CPU_HOST_TARGET}.exe)
    add_custom_target(cpu_host DEPENDS ${WIN_HOST_TARGET})
    separate_arguments(WIN_HOST_COMPILE_FLAGS WINDOWS_COMMAND "${CPU_HOST_COMPILE_FLAGS}")
    add_custom_command(OUTPUT ${WIN_HOST_TARGET} 
             COMMAND ${CMAKE_CXX_COMPILER} /GX ${WIN_HOST_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${CMAKE_BINARY_DIR}/${WIN_HOST_TARGET}
             DEPENDS ${SOURCE_FILE})
else()
    add_executable(${CPU_HOST_TARGET} ${HOST_SOURCE_FILE} ${DEVICE_SOURCE_FILE})
    add_custom_target(cpu_host DEPENDS ${CPU_HOST_TARGET})
    set_target_properties(${CPU_HOST_TARGET} PROPERTIES COMPILE_FLAGS ${CPU_HOST_COMPILE_FLAGS})
endif()

# fpga emulator
if(WIN32)
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set(HOST_EMU_OBJ "host_emu.o")
    set(DEVICE_EMU_OBJ "dev_emu.o")
    set(DEVICE_IMAGE_EMU_OBJ "dev_image_emu.a")

    add_custom_command(OUTPUT ${HOST_EMU_OBJ} 
                 COMMAND dpcpp ${EMULATOR_COMPILE_FLAGS}
                 ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_EMU_OBJ} 
                 DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_EMU_OBJ} 
                 COMMAND dpcpp ${EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_EMU_OBJ}
                 DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_IMAGE_EMU_OBJ} 
                 COMMAND dpcpp ${EMULATOR_LINK_FLAGS} -fsycl-link=image ${DEVICE_EMU_OBJ} -o ${DEVICE_IMAGE_EMU_OBJ} 
                 DEPENDS ${DEVICE_EMU_OBJ})

    add_custom_command(OUTPUT ${EMULATOR_TARGET}
                 COMMAND dpcpp -fintelfpga ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ} -o ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET} 
                 DEPENDS ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ})
else()
    add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})
    set(HOST_EMU_OBJ "host_emu.o")
    set(DEVICE_EMU_OBJ "dev_emu.o")
    set(DEVICE_IMAGE_EMU_OBJ "dev_image_emu.a")

    add_custom_command(OUTPUT ${HOST_EMU_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS}
                 ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_EMU_OBJ} 
                 DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_EMU_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_EMU_OBJ}
                 DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_IMAGE_EMU_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${EMULATOR_LINK_FLAGS} -fsycl-link=image ${DEVICE_EMU_OBJ} -o ${DEVICE_IMAGE_EMU_OBJ} 
                 DEPENDS ${DEVICE_EMU_OBJ})

    add_custom_command(OUTPUT ${EMULATOR_TARGET}
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} -fintelfpga ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ} -o ${CMAKE_BINARY_DIR}/${EMULATOR_TARGET} 
                 DEPENDS ${HOST_EMU_OBJ} ${DEVICE_IMAGE_EMU_OBJ})
endif()

# fpga
if(WIN32)
    add_custom_target(fpga
            COMMAND echo "FPGA hardware flow is not supported in Windows")
else()
    add_custom_target(fpga DEPENDS ${FPGA_TARGET})
    set(HOST_OBJ "host.o")
    set(DEVICE_OBJ "dev.o")
    set(DEVICE_IMAGE_OBJ "dev_image.a")

    add_custom_command(OUTPUT ${HOST_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SOURCE_FILE} -o ${HOST_OBJ} 
                 DEPENDS ${HOST_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_COMPILE_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${DEVICE_SOURCE_FILE} -o ${DEVICE_OBJ} 
                 DEPENDS ${DEVICE_SOURCE_FILE} ${KERNEL_HEADER_FILE})

    add_custom_command(OUTPUT ${DEVICE_IMAGE_OBJ} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${HARDWARE_LINK_FLAGS} -fsycl-link=image ${DEVICE_OBJ} -o ${DEVICE_IMAGE_OBJ} 
                 DEPENDS ${DEVICE_OBJ})

    add_custom_command(OUTPUT ${FPGA_TARGET} 
                 COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS} -fintelfpga ${HOST_OBJ} ${DEVICE_IMAGE_OBJ} -o ${CMAKE_BINARY_DIR}/${FPGA_TARGET} 
                 DEPENDS ${HOST_OBJ} ${DEVICE_IMAGE_OBJ})
endif()

# run
add_custom_target(run
            COMMAND ../${TARGET_NAME}.cpu_host
            DEPENDS ${TARGET_NAME}.cpu_host)
