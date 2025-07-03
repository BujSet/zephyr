cmake_minimum_required(VERSION 3.24)

# Check if we have selective operators and apply configuration overlay
set(OPS_DEF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/gen_ops_def.yml")
if(EXISTS ${OPS_DEF_FILE})
    # Use configuration overlay to disable default portable ops
    set(CONF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/prj.conf;${CMAKE_CURRENT_SOURCE_DIR}/prj_selective.conf")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_hello_world)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -Wno-double-promotion")

# Check if we have generated operator definitions for selective build
set(OPS_DEF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/gen_ops_def.yml")
if(EXISTS ${OPS_DEF_FILE})
    message(STATUS "Using selective operators from: ${OPS_DEF_FILE}")
    message(STATUS "EXECUTORCH_DIR set to : ${EXECUTORCH_DIR}")
    
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
	OPS_SCHEMA_YAML "${OPS_DEF_FILE}"
    )
    
#	ROOT OPS ""
#	INCLUDE_ALL_OPS ""
#	OPS_FROM_MODEL ""
#	DTYPE_SELECTIVE_BUILD "OFF"
#	OPS_FROM_MODEL "/home/zephyruser/zephyr/samples/modules/executorch/arm/hello_world/add.pte"
#	DTYPE_SELECTIVE_BUILD "ON"
    # Generate bindings for the selected kernels
    generate_bindings_for_kernels(
        LIB_NAME "hello_world_ops_lib" 
        FUNCTIONS_YAML "${OPS_DEF_FILE}"
    )
#	CUSTOM_OPS_YAML ""
#	ADD_EXCEPTION_BOUNDARY ""
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
else()
    message(STATUS "No selective operators file found, using default portable_ops_lib")
    message(STATUS "Run 'python build_model.py' to generate selective operators")
    
    # Use default portable operators
    set(EXECUTORCH_OPS_LIB portable_ops_lib)
endif()

set(app_sources
	#src/main.cpp
	#src/program_loader.cpp
	src/arm_executor_runner.cpp
	src/arm_perf_monitor.cpp
)
target_sources(app PRIVATE ${app_sources})

# Link with ExecuTorch and our operators library
target_link_libraries(app PRIVATE 
    libexecutorch
    ${EXECUTORCH_OPS_LIB}
)

# Add include directory for generated headers
target_include_directories(app PRIVATE src) 
