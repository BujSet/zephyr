cmake_minimum_required(VERSION 3.24)

# Use configuration overlay to disable default portable ops to enable selective operator build
set(CONF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/prj.conf;${CMAKE_CURRENT_SOURCE_DIR}/prj_selective.conf")

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
    
# Generate selective operators library
gen_selected_ops(
    LIB_NAME "hello_world_ops_lib" 
    ROOT_OPS "aten::add.out"
)
    #ROOT_OPS "aten::_softmax.out,aten::add.out,aten::addmm.out,aten::any.out,aten::bmm.out,aten::clone.out,aten::convolution.out,aten::eq.Scalar_out,aten::expand_copy.out,aten::full_like.out,aten::gelu.out,aten::logical_not.out,aten::mm.out,aten::mul.Scalar_out,aten::native_layer_norm.out,aten::permute_copy.out,aten::where.self_out"
    #OPS_FROM_MODEL "/home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/whisper/encoder.pte"
	#OPS_SCHEMA_YAML "${OPS_DEF_FILE}"
	#INCLUDE_ALL_OPS "ON"
#	OPS_FROM_MODEL "/home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/add.pte"
#	DTYPE_SELECTIVE_BUILD "ON"
# Generate bindings for the selected kernels
generate_bindings_for_kernels(
    LIB_NAME "hello_world_ops_lib" 
    FUNCTIONS_YAML "${EXECUTORCH_ROOT}/kernels/portable/functions.yaml"
)
#	DTYPE_SELECTIVE_BUILD "OFF"
   
# Create the actual library target with portable kernels
# portable_kernels provides the actual kernel implementations (add_out, etc.)
gen_operators_lib(
    LIB_NAME "hello_world_ops_lib" 
    KERNEL_LIBS portable_kernels
    DEPS executorch_core
)
#	DTYPE_SELECTIVE_BUILD "OFF"
    
# Use our custom selective operators library
set(EXECUTORCH_OPS_LIB hello_world_ops_lib)
message(STATUS "Using selective operators library: ${EXECUTORCH_OPS_LIB}")

set(app_sources
	src/arm_executor_runner.cpp
        ${EXECUTORCH_DIR}/examples/arm/executor_runner/arm_memory_allocator.cpp
        ${EXECUTORCH_DIR}/examples/arm/executor_runner/arm_perf_monitor.cpp
)
	#${EXECUTORCH_DIR}/runtime/platform/default/arm_zephyr.cpp
target_sources(app PRIVATE ${app_sources})

zephyr_linker_sources(SECTIONS linker.ld)
if (CONFIG_CODE_DATA_RELOCATION)
    file(GLOB_RECURSE ET_OP_SRC_FILES ${EXECUTORCH_ROOT}/op_*.cpp )
    zephyr_code_relocate(FILES ${ET_OP_SRC_FILES} LOCATION DDR4S2_TEXT_RODATA NOCOPY)
endif()

# Link with ExecuTorch and our operators library
target_link_libraries(app PRIVATE 
    libexecutorch
    ${EXECUTORCH_OPS_LIB}
)

# Add include directory for generated headers
target_include_directories(app PRIVATE src) 
get_target_property(OUT app LINK_LIBRARIES)
message(STATUS ${OUT})
