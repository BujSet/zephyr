#ifndef __MV2_OUTPUT_VERIFICATION_H__
#define __MV2_OUTPUT_VERIFICATION_H__

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/log.h>
#include <vector>

#include "output_verifier.hpp"

class MV2OutputVerifier : public OutputVerifier {
    public:
        MV2OutputVerifier(void);
        int verify(std::vector<executorch::runtime::EValue>& outputs);
    private:
	float _max(std::vector<executorch::runtime::EValue>& outputs);
	void _print(std::vector<executorch::runtime::EValue>& outputs);
	void _softmax_in_place(std::vector<executorch::runtime::EValue>& outputs, float max_val);
	void _top_k(std::vector<executorch::runtime::EValue>& in_probs, size_t k, std::vector<float>& out_probs, std::vector<size_t>& out_idxs);
	void _get_class(size_t idx, char *name_buf, size_t buf_len);

};
#endif //__MV2_OUTPUT_VERIFICATION_H__
