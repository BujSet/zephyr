cmake_minimum_required(VERSION 3.24)

# Use configuration overlay to disable default portable ops to enable selective operator build
set(CONF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/prj.conf")

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_hello_world)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -Wno-double-promotion -ffunction-sections -fdata-sections")

# Always use selective build since embedded devices are resource constrained
    
# Include ExecuTorch selective build utilities
# Use EXECUTORCH_DIR if available, otherwise use Zephyr's module discovery
if(NOT DEFINED EXECUTORCH_DIR)
    message(STATUS "ZEPHYR_EXECUTORCH_MODULE_DIR set to : ${ZEPHYR_EXECUTORCH_MODULE_DIR}")
    if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
        set(EXECUTORCH_DIR ${ZEPHYR_EXECUTORCH_MODULE_DIR})
        message(STATUS "Using Zephyr module discovery: EXECUTORCH_DIR=${EXECUTORCH_DIR}")
    else()
        message(FATAL_ERROR "ExecutorTorch module not found. Ensure it's properly configured in your Zephyr workspace.")
    endif()
else()
    message(STATUS "Using predefined EXECUTORCH_DIR=${EXECUTORCH_DIR}")
endif()
    
# Set EXECUTORCH_ROOT for the Codegen.cmake file 
set(EXECUTORCH_ROOT ${EXECUTORCH_DIR})
    
include(${EXECUTORCH_DIR}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_DIR}/tools/cmake/Codegen.cmake)
    
set(EXECUTORCH_OPS_LIB selective_portable_ops_lib)
set(EXECUTORCH_SRCS_FILE "${CMAKE_CURRENT_BINARY_DIR}/modules/executorch/executorch/executorch_srcs.cmake")
include(${EXECUTORCH_SRCS_FILE})

if(DEFINED ET_PTE_FILE_PATH_FOR_SELECTIVE_BUILD)
    gen_selected_ops(
        LIB_NAME "${EXECUTORCH_OPS_LIB}" 
	OPS_FROM_MODEL "${ET_PTE_FILE_PATH_FOR_SELECTIVE_BUILD}"
        DTYPE_SELECTIVE_BUILD "ON"
    )
    generate_bindings_for_kernels(
        LIB_NAME "${EXECUTORCH_OPS_LIB}" 
        FUNCTIONS_YAML "${EXECUTORCH_ROOT}/kernels/portable/functions.yaml"
    )
    gen_operators_lib(
        LIB_NAME "${EXECUTORCH_OPS_LIB}" 
        KERNEL_LIBS portable_kernels
        DEPS executorch_core
        DTYPE_SELECTIVE_BUILD "ON"
    )
else()
    gen_selected_ops(
        LIB_NAME "${EXECUTORCH_OPS_LIB}" 
        ROOT_OPS "aten::_softmax.out"
    )

    generate_bindings_for_kernels(
        LIB_NAME "${EXECUTORCH_OPS_LIB}" 
        FUNCTIONS_YAML "${EXECUTORCH_ROOT}/kernels/portable/functions.yaml"
    )
   
    gen_operators_lib(
        LIB_NAME "${EXECUTORCH_OPS_LIB}" 
        KERNEL_LIBS portable_kernels
        DEPS executorch_core
    )
endif()
    
# Use our custom selective operators library
message(STATUS "Using selective operators library: ${EXECUTORCH_OPS_LIB}")

set(app_sources
    src/arm_executor_runner.cpp
    ${EXECUTORCH_DIR}/examples/arm/executor_runner/arm_memory_allocator.cpp
    ${EXECUTORCH_DIR}/examples/arm/executor_runner/arm_perf_monitor.cpp
    ${EXECUTORCH_DIR}/runtime/platform/default/arm_zephyr.cpp
)
target_sources(app PRIVATE ${app_sources})

if(DEFINED CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE)
    target_compile_definitions(app PRIVATE ET_ARM_METHOD_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE})
endif()

zephyr_linker_sources(SECTIONS "${CMAKE_CURRENT_SOURCE_DIR}/../mps3_corstone300_fvp_relocation.ld")
if (CONFIG_CODE_DATA_RELOCATION)
    # Relocation the Executorch portable operator library to partition S2 of DD4
    file(GLOB_RECURSE ET_OP_SRC_FILES ${EXECUTORCH_ROOT}/*.cpp )
    zephyr_code_relocate(FILES ${ET_OP_SRC_FILES} LOCATION DDR4S2_TEXT_RODATA NOCOPY)
endif()

if(CONFIG_EXECUTORCH_INCLUDE_ARM_QUANTIZE_LIBS)
    kernel_link_options(quantized_kernels)
    kernel_link_options(quantized_ops_lib)
    kernel_link_options(cortex_m_kernels)
    kernel_link_options(cortex_m_ops_lib)
endif()

# Link with ExecuTorch and our operators library
target_link_libraries(app PRIVATE 
    libexecutorch
    ${EXECUTORCH_OPS_LIB}
)

if(CONFIG_EXECUTORCH_INCLUDE_ARM_QUANTIZE_LIBS)
    target_link_libraries(app PRIVATE 
        quantized_kernels
        quantized_ops_lib
        cortex_m_kernels
        cortex_m_ops_lib
    )
endif()

# Add include directory for generated headers
target_include_directories(app PRIVATE src) 
get_target_property(OUT app LINK_LIBRARIES)
message(STATUS ${OUT})
