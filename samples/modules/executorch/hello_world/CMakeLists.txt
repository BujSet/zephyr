cmake_minimum_required(VERSION 3.24)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_hello_world)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -Wno-double-promotion")

# Use EXECUTORCH_DIR if available, otherwise use Zephyr's module discovery
if(NOT DEFINED EXECUTORCH_DIR)
    if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
        set(EXECUTORCH_DIR ${ZEPHYR_EXECUTORCH_MODULE_DIR})
        message(STATUS "Using Zephyr module discovery: EXECUTORCH_DIR=${EXECUTORCH_DIR}")
    else()
        message(FATAL_ERROR "ExecutorTorch module not found. Ensure it's properly configured in your Zephyr workspace.")
    endif()
else()
    message(STATUS "Using predefined EXECUTORCH_DIR=${EXECUTORCH_DIR}")
endif()

# Get model filename from Kconfig
if(NOT DEFINED CONFIG_EXECUTORCH_MODEL_NAME)
    set(CONFIG_EXECUTORCH_MODEL_NAME "model_add.pte")
endif()

# Detect target architecture and configure backend accordingly
if(CONFIG_ARM OR CONFIG_ARM64)
    set(EXECUTORCH_USE_ARM_BACKEND TRUE)
    set(EXECUTORCH_BACKEND_TYPE "ARM")
    message(STATUS "Using ARM backend for ExecuTorch")
elseif(CONFIG_ARCH_POSIX OR "${BOARD}" MATCHES "native")
    set(EXECUTORCH_USE_PORTABLE_BACKEND TRUE)
    set(EXECUTORCH_BACKEND_TYPE "Portable")
    message(STATUS "Using Portable backend for ExecuTorch (native/POSIX target)")
else()
    # Default to portable backend for other architectures
    set(EXECUTORCH_USE_PORTABLE_BACKEND TRUE)
    set(EXECUTORCH_BACKEND_TYPE "Portable")
    message(STATUS "Using Portable backend for ExecuTorch (default)")
endif()

# Define output files that will be generated in the build directory
set(GENERATED_OPS_DEF_FILE "${CMAKE_CURRENT_BINARY_DIR}/gen_ops_def.yml")  
set(GENERATED_MODEL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/model_pte.h")

# Build script location
set(BUILD_MODEL_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build_model.py")

# Add custom command to generate header files from existing .pte model
add_custom_command(
    OUTPUT 
        ${GENERATED_OPS_DEF_FILE}
        ${GENERATED_MODEL_HEADER}
    COMMAND 
        ${Python3_EXECUTABLE} ${BUILD_MODEL_SCRIPT}
        --executorch-root "${EXECUTORCH_DIR}"
        --model-file "${CONFIG_EXECUTORCH_MODEL_NAME}"
        --sample-dir "${CMAKE_CURRENT_SOURCE_DIR}"
        --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${BUILD_MODEL_SCRIPT}
    COMMENT "Generating ExecuTorch headers from ${CONFIG_EXECUTORCH_MODEL_NAME}"
    VERBATIM
)

# Create a custom target that depends on the generated files
add_custom_target(generate_model_files ALL
    DEPENDS 
        ${GENERATED_OPS_DEF_FILE}
        ${GENERATED_MODEL_HEADER}
)

# Set EXECUTORCH_ROOT for the Codegen.cmake file 
set(EXECUTORCH_ROOT ${EXECUTORCH_DIR})

include(${EXECUTORCH_DIR}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_DIR}/tools/cmake/Codegen.cmake)

# Generate selective operators library
gen_selected_ops(
    LIB_NAME "hello_world_ops_lib" 
    OPS_SCHEMA_YAML "${GENERATED_OPS_DEF_FILE}"
)

# Generate bindings for the selected kernels
generate_bindings_for_kernels(
    LIB_NAME "hello_world_ops_lib" 
    FUNCTIONS_YAML "${GENERATED_OPS_DEF_FILE}"
)

# Create the actual library target with portable kernels
gen_operators_lib(
    LIB_NAME "hello_world_ops_lib" 
    KERNEL_LIBS portable_kernels
    DEPS executorch_core
)

message(STATUS "Using ExecuTorch ${EXECUTORCH_BACKEND_TYPE} backend with libraries: ${EXECUTORCH_LIBS}")

set(app_sources
    src/main.cpp
    src/program_loader.cpp
    src/pal_zephyr.cpp
)
target_sources(app PRIVATE ${app_sources})

# Make sure the model generation happens before compiling the app
add_dependencies(app generate_model_files)

# Link with ExecuTorch and our operators library
target_link_libraries(app PRIVATE 
    libexecutorch
    hello_world_ops_lib
)

# Add include directories for generated headers and source files
target_include_directories(app PRIVATE 
    src
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated model_pte.h
)

# Force include local warning suppressions header to prevent compilation warnings
target_compile_options(app PRIVATE 
    -include ${CMAKE_CURRENT_SOURCE_DIR}/warning_suppressions.h
) 