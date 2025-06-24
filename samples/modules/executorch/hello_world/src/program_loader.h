#ifndef PROGRAM_LOADER_H
#define PROGRAM_LOADER_H

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#include <memory>
#include <vector>

#include "et_memory_allocator.hpp"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

/**
 * Simple program loader for ExecuTorch ARM Hello World sample
 * Loads embedded model and provides simple inference interface
 */
class ProgramLoader {
public:
    /**
     * Get singleton instance
     */
    static ProgramLoader& getInstance() {
        static ProgramLoader instance;
        return instance;
    }
    
    /**
     * Initialize the program loader (call once before first use)
     */
    void initialize();

    /**
     * Load the embedded program from model_pte.h
     * @return Error::Ok on success
     */
    Error loadProgram();

    /**
     * Run inference on the loaded model
     * @param input1 First input tensor data
     * @param input2 Second input tensor data  
     * @param input_size Size of input tensors
     * @param output Output tensor data
     * @param output_size Size of output tensor
     * @return Error::Ok on success
     */
    Error runInference(const float* input1, const float* input2, size_t input_size,
                      float* output, size_t output_size);

    /**
     * Check if program is loaded
     */
    bool isLoaded() const { return method_ != nullptr && method_->ok(); }

private:
    ProgramLoader();
    ~ProgramLoader() = default;
    
    // Disable copy/move
    ProgramLoader(const ProgramLoader&) = delete;
    ProgramLoader& operator=(const ProgramLoader&) = delete;
    ProgramLoader(ProgramLoader&&) = delete;
    ProgramLoader& operator=(ProgramLoader&&) = delete;

    /**
     * Create input tensor
     */
    Error createInputTensor(const float* data, size_t size, size_t input_index);

    // Initialization state
    bool initialized_;
    
    // Memory allocators
    std::unique_ptr<ETMemoryAllocator> method_allocator_;
    std::unique_ptr<ETMemoryAllocator> temp_allocator_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<HierarchicalAllocator> planned_memory_;
    
    // ExecuTorch objects
    std::unique_ptr<Program> program_;
    std::unique_ptr<Result<Method>> method_;
    
    // Memory management
    std::vector<uint8_t*> planned_buffers_;
    std::vector<Span<uint8_t>> planned_spans_;
};

#endif // PROGRAM_LOADER_H
