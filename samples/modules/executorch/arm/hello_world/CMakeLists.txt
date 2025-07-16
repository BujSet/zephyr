cmake_minimum_required(VERSION 3.24)

# Use configuration overlay to disable default portable ops to enable selective operator build
set(CONF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/prj.conf;${CMAKE_CURRENT_SOURCE_DIR}/prj_selective.conf")

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_hello_world)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -Wno-double-promotion")

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
    ROOT_OPS "aten::add.out,aten::native_layer_norm.out,aten::gelu.out,aten::clone.out,aten::addmm.out,aten::permute_copy.out,aten::mm.out,aten::mul.Scalar_out,aten::expand_copy.out,aten::bmm.out,aten::_softmax.out"
)
    #OPS_FROM_MODEL "/home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/whisper/encoder.pte"
    #DTYPE_SELECTIVE_BUILD "ON"
    #OPS_SCHEMA_YAML "/home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/whisper_dtype_operators.yaml"
    #INCLUDE_ALL_OPS "ON"
    #ROOT_OPS "aten::add.out"

    # Below uses about 93% of ITCM (512KB) and is an incomplete list of operators
    #ROOT_OPS "aten::add.out,aten::native_layer_norm.out,aten::gelu.out,aten::clone.out,aten::addmm.out,aten::permute_copy.out,aten::mm.out,aten::mul.Scalar_out"

    # Complete operator list for whisper, but exceeds ITCM memory limit of 512 KB
    #ROOT_OPS "aten::add.out,aten::native_layer_norm.out,aten::gelu.out,aten::clone.out,aten::addmm.out,aten::permute_copy.out,aten::mm.out,aten::mul.Scalar_out,aten::expand_copy.out,aten::bmm.out,aten::_softmax.out,aten::eq.Scalar_out,aten::logical_not.out,aten::any.out,aten::full_like.out,aten::where.self_out"
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
	src/arm_perf_monitor.cpp
)
target_sources(app PRIVATE ${app_sources})

zephyr_linker_sources(SECTIONS linker.ld)
if (CONFIG_CODE_DATA_RELOCATION)
    #zephyr_code_relocate(LIBRARY ${EXECUTORCH_OPS_LIB} LOCATION DDR4S2)
    #zephyr_code_relocate(FILES "build/hello_world_ops_lib/RegisterCodegenUnboxedKernelsEverything.cpp" LOCATION DDR4S2)
    #zephyr_code_relocate(FILES ${app_sources} LOCATION DDR4S2)
    #file(GLOB ops_lib_cpp_sources "build/hello_world_ops_lib/*.cpp")
    #file(GLOB ops_lib_h_sources "build/hello_world_ops_lib/*.h")
    #file(GLOB ops_lib_yaml_sources "build/hello_world_ops_lib/*.yaml")
    #zephyr_code_relocate(FILES ${ops_lib_cpp_sources} LOCATION DDR4S2)
    #zephyr_code_relocate(FILES ${ops_lib_h_sources} LOCATION DDR4S2)
    #zephyr_code_relocate(FILES ${ops_lib_yaml_sources} LOCATION DDR4S2)
    #zephyr_code_relocate(LIBRARY libexecutorch LOCATION DDR4S3)
    #zephyr_code_relocate(FILES ${app_sources} LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(LIBRARY app LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(LIBRARY ${EXECUTORCH_OPS_LIB} LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    zephyr_code_relocate(LIBRARY libexecutorch LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(LIBRARY zephyr LOCATION ISRAM)

    file(GLOB_RECURSE ET_SRC_FILES ${EXECUTORCH_ROOT}/*.cpp ${EXECUTORCH_ROOT}/*.hpp ${EXECUTORCH_ROOT}/*.c)
    zephyr_code_relocate(FILES ${ET_SRC_FILES} LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    file(GLOB_RECURSE APP_SRC_FILES src/*.cpp)
    zephyr_code_relocate(FILES ${APP_SRC_FILES} LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #file(GLOB_RECURSE Z_SRC_FILES $ENV{ZEPHYR_BASE}/*.cpp $ENV{ZEPHYR_BASE}/*.hpp $ENV{ZEPHYR_BASE}/*.c)
    #zephyr_code_relocate(FILES ${Z_SRC_FILES} LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #file(GLOB op_source_files "${EXECUTORCH_ROOT}/kernels/portable/cpu/*.cpp")
    #zephyr_code_relocate(FILES ${op_source_files} LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #file(GLOB op_util_source_files "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/*.cpp")
    #zephyr_code_relocate(FILES ${op_util_source_files} LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)

    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/*.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_mm.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_permute_copy.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_mul.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_add.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_clone.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_gelu.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_native_layer_norm.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_softmax.cpp LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)

    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/sched.c LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/kheap.c LOCATION ISRAM_TEXT_RODATA_DATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/mem_slab.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/msg_q.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/mutex.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/queue.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/sem.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/system_work_q.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/work.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/thread.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/timeslicing.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/thread_monitor.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/timeout.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/poll.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/usage.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/mempool.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/busy_wait.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/device.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/idle.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/kernel/timer.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
    #zephyr_code_relocate(FILES $ENV{ZEPHYR_BASE}/heap.c LOCATION ISRAM_TEXT_RODATA NOKEEP)
endif()

# Link with ExecuTorch and our operators library
target_link_libraries(app PRIVATE 
    libexecutorch
    ${EXECUTORCH_OPS_LIB}
)

# Add include directory for generated headers
target_include_directories(app PRIVATE src) 

