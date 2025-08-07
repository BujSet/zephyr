#include "mv2_output_verifier.hpp" 

#include <cmath>
#include <string.h>

#include "imagenet_classes.h"

MV2OutputVerifier::MV2OutputVerifier(void) {
}

float
MV2OutputVerifier:: _max(std::vector<executorch::runtime::EValue>& outputs) {
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

void 
MV2OutputVerifier::_softmax_in_place(std::vector<executorch::runtime::EValue>& outputs, float max_val) {
    double sum_exp = 0.0;
    for (int i = 0; i < outputs.size(); ++i) {
        assert(outputs[i].isTensor());
        executorch::aten::Tensor tensor = outputs[i].toTensor();
	// First pass to determine sum_exp
        for (int j = 0; j < tensor.numel(); ++j) {
            assert(tensor.scalar_type() == executorch::aten::ScalarType::Float);
            float value  = tensor.const_data_ptr<float>()[j];
            tensor.data_ptr<float>()[j] = exp(value - max_val);
	    sum_exp += tensor.const_data_ptr<float>()[j];
        }
	// Second pass to normalize to sum_exp
        for (int j = 0; j < tensor.numel(); ++j) {
            float value  = tensor.const_data_ptr<float>()[j];
            tensor.data_ptr<float>()[j] = value / sum_exp;
        }
    }
}

void 
MV2OutputVerifier::_top_k(std::vector<executorch::runtime::EValue>& in_probs, 
		size_t k, std::vector<float>& out_probs, std::vector<size_t>& out_idxs) {
    assert(out_probs.size() == 0);
    assert(out_idxs.size() == 0);
    for (size_t i = 0; i < in_probs.size(); ++i) {
        executorch::aten::Tensor tensor = in_probs[i].toTensor();
        for (int j = 0; j < tensor.numel(); ++j) {
            float value  = tensor.const_data_ptr<float>()[j];
	    if (out_probs.size() == 0) {
	        out_probs.push_back(value);
		out_idxs.push_back((i * tensor.numel()) + j);
	    } else if (out_probs.size() == 1) {
	        if (value > out_probs[0]) {
		    out_probs.insert(out_probs.begin(), value);
		    out_idxs.insert(out_idxs.begin(), (i * tensor.numel()) + j);
		} else {
	            out_probs.push_back(value);
		    out_idxs.push_back((i * tensor.numel()) + j);
		}
	    } else if (out_probs.size() == 2) {
	        if (value > out_probs[0]) {
		    out_probs.insert(out_probs.begin(), value);
		    out_idxs.insert(out_idxs.begin(), (i * tensor.numel()) + j);
		} else if (value > out_probs[1]){
		    out_probs.insert(out_probs.begin() + 1, value);
		    out_idxs.insert(out_idxs.begin() + 1, (i * tensor.numel()) + j);
		} else {
	            out_probs.push_back(value);
		    out_idxs.push_back((i * tensor.numel()) + j);
		}
	    } else {
                assert(out_probs.size() == 3);
	        if (value > out_probs[0]) {
		    out_probs.insert(out_probs.begin(), value);
		    out_idxs.insert(out_idxs.begin(), (i * tensor.numel()) + j);
		} else if (value > out_probs[1]){
		    out_probs.insert(out_probs.begin() + 1, value);
		    out_idxs.insert(out_idxs.begin() + 1, (i * tensor.numel()) + j);
		} else if (value > out_probs[2]){
		    out_probs.insert(out_probs.begin() + 2, value);
		    out_idxs.insert(out_idxs.begin() + 2, (i * tensor.numel()) + j);
		} else {
	            out_probs.push_back(value);
		    out_idxs.push_back((i * tensor.numel()) + j);
		}
		out_probs.pop_back();
		out_idxs.pop_back();
	    }
        }
    }
}

void 
MV2OutputVerifier::_get_class(size_t idx, char *name_buf, size_t buf_len) {
    unsigned int name_idx = 0;
    unsigned int name_char_idx = 0;
    for(unsigned int i = 0; i < imagenet_classes_txt_len; i++) {
        if (name_idx == idx) {
	    assert(name_char_idx < buf_len);
            if (imagenet_classes_txt[i] == '\n') {
                name_buf[name_char_idx++] = '\0';
	        break;
	    } else {
                name_buf[name_char_idx++] = imagenet_classes_txt[i];
	    }
        }
        if(imagenet_classes_txt[i] == '\n') {
            name_idx++;
        }
    }
}

int
MV2OutputVerifier::verify(std::vector<executorch::runtime::EValue>& outputs) {
    if (outputs.size() != 1) {
        ET_LOG(Error, "ERROR: Incorrect top level dim of output size (%zu != 1)", outputs.size());
	return 1;
    }
    if (!outputs[0].isTensor()) {
        ET_LOG(Error, "ERROR: Expected output[0] to return a tesnor but got something else");
	return 1;
    }
    executorch::aten::Tensor tensor = outputs[0].toTensor();
    if (tensor.numel() != 1000) {
        ET_LOG(Error, "ERROR: Incorrect lower level dim of output (%zu != 1000)", tensor.numel());
	return 1;
    }
    size_t top_k = 3;
    std::vector<float> top_k_probs;
    std::vector<size_t> top_k_idxs;
    top_k_probs.reserve(top_k + 1);
    top_k_idxs.reserve(top_k + 1);

    float max_val = _max(outputs);
    ET_LOG(Info, "max_val from verifier is %f", max_val);
    _softmax_in_place(outputs, max_val);
    _top_k(outputs, top_k, top_k_probs, top_k_idxs);

    char class_name[256];
    memset(class_name, 0, sizeof(class_name));
    for (size_t i = 0; i < top_k; i++) {
	_get_class(top_k_idxs[i], class_name, 256);
        ET_LOG(Info, "Predicted top %u class (%d,%f)->%s", 
			i, 
			top_k_idxs[i], 
			top_k_probs[i],
			class_name);
        memset(class_name, 0, sizeof(class_name));
    }
    float expected_probs[3] = {0.397523,0.043148,0.035979};
    int expected_idxs[3] = {258,259,279};
    for (int i = 0; i < 3; i++) {
        float prob = top_k_probs[i];
        int idx = top_k_idxs[i];

        if (fabs(prob - expected_probs[i]) >= 0.00001f) {
            ET_LOG(Error, "ERROR: Incorrect probability top %zu predicted label (%f != %f)", i, prob, expected_probs[i]);
	    return 1;
        }
        if (idx != expected_idxs[i]) {
            ET_LOG(Error, "ERROR: Incorrect label choice for top %zu predicted label (%d != %d)", i, idx, expected_idxs[i]);
	    return 1;
        }
    }
    return 0;
}
