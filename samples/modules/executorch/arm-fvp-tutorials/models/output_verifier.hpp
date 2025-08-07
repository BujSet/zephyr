#ifndef __OUTPUT_VERIFICATION_H__
#define __OUTPUT_VERIFICATION_H__

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/log.h>
#include <vector>
class OutputVerifier {
    public:
        OutputVerifier(void);
        int verify(std::vector<executorch::runtime::EValue>& outputs);
};
#endif //__OUTPUT_VERIFICATION_H__
