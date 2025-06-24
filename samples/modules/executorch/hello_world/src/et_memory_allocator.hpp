#ifndef ET_MEMORY_ALLOCATOR_HPP
#define ET_MEMORY_ALLOCATOR_HPP

#include <executorch/runtime/core/memory_allocator.h>

using executorch::runtime::MemoryAllocator;

class ETMemoryAllocator : public MemoryAllocator {
public:
    ETMemoryAllocator(uint32_t size, uint8_t* base_address) : MemoryAllocator(size, base_address), used_(0) {}

    void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
        void* ret = MemoryAllocator::allocate(size, alignment);
        if (ret != nullptr) {
            if ((size & (alignment - 1)) == 0) {
                used_ += size;
            } else {
                used_ = (used_ & ~(alignment - 1)) + alignment + size;
            }
        }
        return ret;
    }

    size_t used_size() const { return used_; }
    size_t free_size() const { return MemoryAllocator::size() - used_; }

private:
    size_t used_;
};

#endif  // ET_MEMORY_ALLOCATOR_HPP
