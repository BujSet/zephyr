/*
 * Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#if DT_NODE_HAS_STATUS(DT_NODELABEL(ddr4s0), okay)
GROUP_START(DDR4S0)

	SECTION_DATA_PROLOGUE(_DDR4S0_SECTION_NAME,,SUBALIGN(16))
	{
		. = ALIGN(16);
		*(network_model_sec)
	} GROUP_LINK_IN(DDR4S0)

GROUP_END(DDR4S0)
#endif

#if DT_NODE_HAS_STATUS(DT_NODELABEL(ddr4s1), okay)
GROUP_START(DDR4S1)

	SECTION_DATA_PROLOGUE(_DDR4S1_SECTION_NAME,,SUBALIGN(16))
	{
		. = ALIGN(16);
		*(input_data_sec)
	} GROUP_LINK_IN(DDR4S1)

GROUP_END(DDR4S1)
#endif

#if DT_NODE_HAS_STATUS(DT_NODELABEL(isram), okay)
GROUP_START(ISRAM)

	SECTION_DATA_PROLOGUE(_ISRAM_SECTION_NAME,,SUBALIGN(16))
	{
		. = ALIGN(16);
		*(kernels_portable_cpu_op_addmm_out)
	        *(kernels_portable_cpu_op_add_out) 
	        *(kernels_portable_cpu_op_bmm_out) 
	        *(kernels_portable_cpu_op_eq_scalar_out) 
	        *(kernels_portable_cpu_op_expand_copy_out) 
	        *(kernels_portable_cpu_op_gelu_out) 
	        *(kernels_portable_cpu_op_mm_out) 
	        *(kernels_portable_cpu_op_mul_out) 
	        *(kernels_portable_cpu_op_native_layer_norm_out) 
	        *(kernels_portable_cpu_op_permute_copy_out) 
	        *(kernels_portable_cpu_op_softmax_out) 
		*(registered_kernels_data_sec)
	} GROUP_LINK_IN(ISRAM)

GROUP_END(ISRAM)
#endif
