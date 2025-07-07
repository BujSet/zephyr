
#ifndef _ARM_MEMORY_ALLOCATOR_HPP
#define _ARM_MEMORY_ALLOCATOR_HPP

#include <executorch/runtime/core/memory_allocator.h>

using executorch::runtime::MemoryAllocator;

// Setup our own allocator that can show some extra stuff like used and free
// memory info
class ArmMemoryAllocator : public MemoryAllocator {
 public:
  ArmMemoryAllocator(uint32_t size, uint8_t* base_address)
      : MemoryAllocator(size, base_address), used_(0), peak_used_(0) {}

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    void* ret = MemoryAllocator::allocate(size, alignment);
    if (ret != nullptr) {
      // Align with the same code as in MemoryAllocator::allocate() to keep
      // used_ "in sync" As alignment is expected to be power of 2 (checked by
      // MemoryAllocator::allocate()) we can check it the lower bits
      // (same as alignment - 1) is zero or not.
      if ((size & (alignment - 1)) == 0) {
        // Already aligned.
        used_ += size;
      } else {
        used_ = (used_ | (alignment - 1)) + 1 + size;
      }
      if (used_ > peak_used_)
        peak_used_ = used_;
    }
    return ret;
  }

  // Returns the used size of the allocator's memory buffer.
  size_t used_size() const {
    return used_;
  }

  // Returns the peak memory usage of the allocator's memory buffer
  // Peak usage is useful when doing multiple allocations & resets
  size_t peak_used() const {
    return peak_used_;
  }

  // Returns the free size of the allocator's memory buffer.
  size_t free_size() const {
    return MemoryAllocator::size() - used_;
  }

  void reset() {
    MemoryAllocator::reset();
    used_ = 0;
  }

 private:
  size_t used_;
  size_t peak_used_;
};

#endif  // _ARM_MEMORY_ALLOCATOR_HPP
