#include "program_loader.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <cstddef>
#include <memory>
#include <cstring>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/executor/program.h>

#include "model_pte.h"
#include "et_memory_allocator.hpp"

LOG_MODULE_REGISTER(program_loader, LOG_LEVEL_INF);

using namespace torch::executor;

// Memory pools for ExecuTorch execution
// Sizes are configurable via Kconfig
#ifndef CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE
#define CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE (16 * 1024)  // 16KB default
#endif
#ifndef CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE
#define CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE (2 * 1024)     // 2KB default
#endif

static unsigned char method_allocation_pool[CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE];
static unsigned char temp_allocation_pool[CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE];

ProgramLoader::ProgramLoader() 
    : initialized_(false),
      method_(nullptr) {
    // Constructor is now minimal - no complex initialization
}

void ProgramLoader::initialize() {
    if (initialized_) {
        return; // Already initialized
    }
    
    LOG_INF("Initializing ProgramLoader...");
    
    // Initialize ExecuTorch runtime
    executorch::runtime::runtime_init();
    
    // Create memory allocators
    method_allocator_ = std::make_unique<ETMemoryAllocator>(
        CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE, 
        method_allocation_pool
    );
    
    temp_allocator_ = std::make_unique<ETMemoryAllocator>(
        CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE, 
        temp_allocation_pool
    );
    
    initialized_ = true;
    LOG_INF("ProgramLoader initialized successfully");
}

Error ProgramLoader::loadProgram() {
    // Ensure we're initialized
    if (!initialized_) {
        LOG_ERR("ProgramLoader not initialized - call initialize() first");
        return Error::InvalidState;
    }
    
    // Calculate model size from the array
    size_t model_size = sizeof(model_pte);
    
    LOG_INF("Loading embedded program, size: %zu bytes", model_size);
    
    // Create data loader for embedded model
    auto loader = std::make_unique<BufferDataLoader>(model_pte, model_size);
    
    // Load program
    auto program_result = Program::load(loader.get());
    if (!program_result.ok()) {
        LOG_ERR("Failed to load program: %d", (int)program_result.error());
        return program_result.error();
    }
    
    program_ = std::make_unique<Program>(std::move(program_result.get()));
    LOG_INF("Program loaded successfully");
    
    // Get method metadata to determine memory planning requirements
    auto method_meta_result = program_->method_meta("forward");
    if (!method_meta_result.ok()) {
        LOG_ERR("Failed to get method metadata: %d", (int)method_meta_result.error());
        return method_meta_result.error();
    }
    
    MethodMeta method_meta = method_meta_result.get();
    
    // Allocate memory-planned buffers
    size_t num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
    LOG_INF("Method requires %zu memory-planned buffers", num_memory_planned_buffers);
    
    planned_buffers_.clear();
    planned_spans_.clear();
    
    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
        auto buffer_size_result = method_meta.memory_planned_buffer_size(id);
        if (!buffer_size_result.ok()) {
            LOG_ERR("Failed to get buffer size for buffer %zu: %d", id, (int)buffer_size_result.error());
            return buffer_size_result.error();
        }
        
        size_t buffer_size = static_cast<size_t>(buffer_size_result.get());
        LOG_INF("Allocating planned buffer %zu, size %zu bytes", id, buffer_size);
        
        uint8_t* buffer = reinterpret_cast<uint8_t*>(method_allocator_->allocate(buffer_size));
        if (!buffer) {
            LOG_ERR("Failed to allocate memory-planned buffer %zu of size %zu", id, buffer_size);
            return Error::MemoryAllocationFailed;
        }
        
        planned_buffers_.push_back(buffer);
        planned_spans_.push_back({planned_buffers_.back(), buffer_size});
    }
    
    // Create hierarchical allocator for planned memory
    planned_memory_ = std::make_unique<HierarchicalAllocator>(
        HierarchicalAllocator({planned_spans_.data(), planned_spans_.size()})
    );
    
    // Update memory manager with planned memory
    memory_manager_ = std::make_unique<MemoryManager>(
        method_allocator_.get(),         // method_allocator
        planned_memory_.get(),           // planned_memory 
        temp_allocator_.get()           // temp_allocator
    );
    
    // Load method with updated memory manager
    auto method_result = program_->load_method("forward", memory_manager_.get());
    if (!method_result.ok()) {
        LOG_ERR("Failed to load method 'forward': %d", (int)method_result.error());
        return method_result.error();
    }
    
    method_ = std::make_unique<Result<Method>>(std::move(method_result));
    LOG_INF("Method 'forward' loaded successfully");
    
    return Error::Ok;
}

Error ProgramLoader::createInputTensor(const float* data, size_t size, size_t input_index) {
    if (!isLoaded()) {
        LOG_ERR("Program not loaded");
        return Error::InvalidState;
    }
    
    Method& method = method_->get();
    
    // Get input tensor metadata
    auto tensor_meta_result = method.method_meta().input_tensor_meta(input_index);
    if (!tensor_meta_result.ok()) {
        LOG_ERR("Failed to get input tensor meta for index %zu: %d", 
                input_index, (int)tensor_meta_result.error());
        return tensor_meta_result.error();
    }
    
    TensorInfo tensor_meta = tensor_meta_result.get();
    
    // Validate input size
    size_t expected_bytes = tensor_meta.nbytes();
    size_t provided_bytes = size * sizeof(float);
    
    if (provided_bytes != expected_bytes) {
        LOG_ERR("Input size mismatch for tensor %zu: expected %zu bytes, got %zu bytes",
                input_index, expected_bytes, provided_bytes);
        return Error::InvalidArgument;
    }
    
    // Create metadata arrays with persistent storage
    static std::vector<int32_t> sizes_storage[2];  // Support up to 2 input tensors
    static std::vector<uint8_t> dim_order_storage[2];
    static std::unique_ptr<TensorImpl> tensor_impls[2];  // Persistent storage for TensorImpl objects
    
    if (input_index >= 2) {
        LOG_ERR("Input index %zu exceeds supported maximum of 2", input_index);
        return Error::InvalidArgument;
    }
    
    // Copy metadata to persistent storage
    sizes_storage[input_index].assign(tensor_meta.sizes().begin(), tensor_meta.sizes().end());
    dim_order_storage[input_index].assign(tensor_meta.dim_order().begin(), tensor_meta.dim_order().end());
    
    // Create TensorImpl with persistent metadata using unique_ptr for proper lifecycle
    tensor_impls[input_index] = std::make_unique<TensorImpl>(
        ScalarType::Float,
        sizes_storage[input_index].size(),
        sizes_storage[input_index].data(),
        const_cast<void*>(static_cast<const void*>(data)),
        dim_order_storage[input_index].data()
    );
    
    // Create Tensor and EValue
    Tensor tensor(tensor_impls[input_index].get());
    EValue tensor_evalue(tensor);
    
    // Set input
    Error error = method.set_input(tensor_evalue, input_index);
    if (error != Error::Ok) {
        LOG_ERR("Failed to set input tensor %zu: %d", input_index, (int)error);
        return error;
    }
    
    LOG_DBG("Input tensor %zu set successfully", input_index);
    return Error::Ok;
}

Error ProgramLoader::runInference(const float* input1, const float* input2, size_t input_size,
                                 float* output, size_t output_size) {
    if (!isLoaded()) {
        LOG_ERR("Program not loaded");
        return Error::InvalidState;
    }
    
    Method& method = method_->get();
    
    LOG_INF("Running inference with inputs of size %zu", input_size);
    
    // Set input tensors
    Error error = createInputTensor(input1, input_size, 0);
    if (error != Error::Ok) {
        return error;
    }
    
    error = createInputTensor(input2, input_size, 1);
    if (error != Error::Ok) {
        return error;
    }
    
    // Execute the method
    error = method.execute();
    if (error != Error::Ok) {
        LOG_ERR("Method execution failed: %d", (int)error);
        return error;
    }
    
    LOG_INF("Method executed successfully");
    
    // Get output
    EValue output_evalue = method.get_output(0);
    if (!output_evalue.isTensor()) {
        LOG_ERR("Output is not a tensor");
        return Error::InvalidArgument;
    }
    
    Tensor output_tensor = output_evalue.toTensor();
    
    // Validate output size
    size_t expected_elements = output_tensor.numel();
    if (output_size < expected_elements) {
        LOG_ERR("Output buffer too small: need %zu elements, got %zu",
                expected_elements, output_size);
        return Error::InvalidArgument;
    }
    
    // Copy output data
    const float* output_data = output_tensor.const_data_ptr<float>();
    std::memcpy(output, output_data, expected_elements * sizeof(float));
    
    LOG_INF("Inference completed, output size: %zu elements", expected_elements);
    return Error::Ok;
} 