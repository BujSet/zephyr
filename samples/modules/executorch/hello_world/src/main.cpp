/*
 * Copyright (c) 2025 Petri Oksanen
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "program_loader.h"

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);

int main(void) {
	LOG_INF("ExecuTorch Hello World Sample");
	
	// Get program loader instance and initialize it
	ProgramLoader& loader = ProgramLoader::getInstance();
	loader.initialize();
	
	// Load the embedded model
	auto error = loader.loadProgram();
	if (error != executorch::runtime::Error::Ok) {
		LOG_ERR("Failed to load program: %d", (int)error);
		return -1;
	}
	
	LOG_INF("Program loaded successfully");
	
	// Test data: simple addition
	float input1[] = {2.0f};
	float input2[] = {3.0f};
	float output[1];
	
	// Run inference
	error = loader.runInference(input1, input2, 1, output, 1);
	if (error != executorch::runtime::Error::Ok) {
		LOG_ERR("Inference failed: %d", (int)error);
		return -1;
	}
	
	LOG_INF("Inference result: %.2f + %.2f = %.2f", 
			input1[0], input2[0], output[0]);
	
	// Expected result should be 5.0
	if (output[0] == 5.0f) {
		LOG_INF("✓ Test PASSED: Addition worked correctly!");
	} else {
		LOG_ERR("✗ Test FAILED: Expected 5.0, got %.2f", output[0]);
		return -1;
	}
	
	return 0;
} 