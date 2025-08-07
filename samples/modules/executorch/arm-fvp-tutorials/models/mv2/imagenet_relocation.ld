/*
 * Copyright 2025 Meta
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#if DT_NODE_HAS_STATUS(DT_NODELABEL(ddr4s0), okay)
GROUP_START(DDR4S0)

	SECTION_DATA_PROLOGUE(_DDR4S0_SECTION_NAME,,SUBALIGN(16))
	{
		. = ALIGN(16);
		*(image_net_classes input_image)
	} GROUP_LINK_IN(DDR4S0)

GROUP_END(DDR4S0)
#endif
