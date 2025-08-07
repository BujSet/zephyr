#include "output_verifier.hpp" 

#include <cmath>

OutputVerifier::OutputVerifier(void) {
}

float
OutputVerifier:: _max(std::vector<executorch::runtime::EValue>& outputs) {
    float max_val = 0;
    bool max_val_set = false;
    for (int i = 0; i < outputs.size(); ++i) {
        assert(outputs[i].isTensor());
        executorch::aten::Tensor tensor = outputs[i].toTensor();
        for (int j = 0; j < tensor.numel(); ++j) {
            assert(tensor.scalar_type() == executorch::aten::ScalarType::Float);
            float value  = tensor.const_data_ptr<float>()[j];
	    if (!max_val_set || value > max_val) {
                max_val = value;
		max_val_set = true;
	    }
        }
    }
    return max_val;
}

int
OutputVerifier::verify(std::vector<executorch::runtime::EValue>& outputs) {
    float max_val = _max(outputs);

    if (max_val > 0.0) {
	    return 1;
    }

    return 0;
}
