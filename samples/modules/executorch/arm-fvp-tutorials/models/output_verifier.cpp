#include "output_verifier.hpp" 

OutputVerifier::OutputVerifier(void) {
}

void
OutputVerifier::_print(std::vector<executorch::runtime::EValue>& outputs) {
  // Print the outputs.
  ET_LOG(Info, "Printing outputs.");
  for (int i = 0; i < outputs.size(); ++i) {
    if (outputs[i].isTensor()) {
      executorch::aten::Tensor tensor = outputs[i].toTensor();
      // The output might be collected and parsed so printf() is used instead
      // of ET_LOG() here
      for (int j = 0; j < tensor.numel(); ++j) {
        if (tensor.scalar_type() == executorch::aten::ScalarType::Int) {
          printf(
              "Output[%d][%d]: (int) %d\n",
              i,
              j,
              tensor.const_data_ptr<int>()[j]);
        } else if (tensor.scalar_type() == executorch::aten::ScalarType::Float) {
          printf(
              "Output[%d][%d]: (float) %f\n",
              i,
              j,
              tensor.const_data_ptr<float>()[j]);
        } else if (tensor.scalar_type() == executorch::aten::ScalarType::Char) {
          printf(
              "Output[%d][%d]: (char) %d\n",
              i,
              j,
              tensor.const_data_ptr<int8_t>()[j]);
        } else if (tensor.scalar_type() == executorch::aten::ScalarType::Bool) {
          printf(
              "Output[%d][%d]: (bool) %s (0x%x)\n",
              i,
              j,
              tensor.const_data_ptr<int8_t>()[j] ? "true " : "false",
              tensor.const_data_ptr<int8_t>()[j]);
        }
      }
    } else {
      printf("Output[%d]: Not Tensor\n", i);
    }
  }
}
