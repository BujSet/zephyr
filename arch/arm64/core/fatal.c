/*
 * Copyright (c) 2019 Carlo Caione <ccaione@baylibre.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Kernel fatal error handler for ARM64 Cortex-A
 *
 * This module provides the z_arm64_fatal_error() routine for ARM64 Cortex-A
 * CPUs and z_arm64_do_kernel_oops() routine to manage software-generated fatal
 * exceptions
 */

#include <zephyr/debug/symtab.h>
#include <zephyr/drivers/pm_cpu_ops.h>
#include <zephyr/arch/common/exc_handle.h>
#include <zephyr/kernel.h>
#include <zephyr/linker/linker-defs.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/poweroff.h>
#include <kernel_arch_func.h>
#include <kernel_arch_interface.h>
#include <zephyr/arch/exception.h>

#include "paging.h"

LOG_MODULE_DECLARE(os, CONFIG_KERNEL_LOG_LEVEL);

#ifdef CONFIG_ARM64_SAFE_EXCEPTION_STACK
K_KERNEL_PINNED_STACK_ARRAY_DEFINE(z_arm64_safe_exception_stacks,
				   CONFIG_MP_MAX_NUM_CPUS,
				   CONFIG_ARM64_SAFE_EXCEPTION_STACK_SIZE);

void z_arm64_safe_exception_stack_init(void)
{
	int cpu_id;
	char *safe_exc_sp;

	cpu_id = arch_curr_cpu()->id;
	safe_exc_sp = K_KERNEL_STACK_BUFFER(z_arm64_safe_exception_stacks[cpu_id]) +
		      CONFIG_ARM64_SAFE_EXCEPTION_STACK_SIZE;
	arch_curr_cpu()->arch.safe_exception_stack = (uint64_t)safe_exc_sp;
	write_sp_el0((uint64_t)safe_exc_sp);

	arch_curr_cpu()->arch.current_stack_limit = 0UL;
	arch_curr_cpu()->arch.corrupted_sp = 0UL;
}
#endif

#ifdef CONFIG_USERSPACE
Z_EXC_DECLARE(z_arm64_user_string_nlen);

static const struct z_exc_handle exceptions[] = {
	Z_EXC_HANDLE(z_arm64_user_string_nlen),
};
#endif /* CONFIG_USERSPACE */

#ifdef CONFIG_EXCEPTION_DEBUG
static void dump_esr(uint64_t esr, bool *dump_far)
{
	const char *err;

	switch (GET_ESR_EC(esr)) {
	case 0b000000: /* 0x00 */
		err = "Unknown reason";
		break;
	case 0b000001: /* 0x01 */
		err = "Trapped WFI or WFE instruction execution";
		break;
	case 0b000011: /* 0x03 */
		err = "Trapped MCR or MRC access with (coproc==0b1111) that "
		      "is not reported using EC 0b000000";
		break;
	case 0b000100: /* 0x04 */
		err = "Trapped MCRR or MRRC access with (coproc==0b1111) "
		      "that is not reported using EC 0b000000";
		break;
	case 0b000101: /* 0x05 */
		err = "Trapped MCR or MRC access with (coproc==0b1110)";
		break;
	case 0b000110: /* 0x06 */
		err = "Trapped LDC or STC access";
		break;
	case 0b000111: /* 0x07 */
		err = "Trapped access to SVE, Advanced SIMD, or "
		      "floating-point functionality";
		break;
	case 0b001100: /* 0x0c */
		err = "Trapped MRRC access with (coproc==0b1110)";
		break;
	case 0b001101: /* 0x0d */
		err = "Branch Target Exception";
		break;
	case 0b001110: /* 0x0e */
		err = "Illegal Execution state";
		break;
	case 0b010001: /* 0x11 */
		err = "SVC instruction execution in AArch32 state";
		break;
	case 0b011000: /* 0x18 */
		err = "Trapped MSR, MRS or System instruction execution in "
		      "AArch64 state, that is not reported using EC "
		      "0b000000, 0b000001 or 0b000111";
		break;
	case 0b011001: /* 0x19 */
		err = "Trapped access to SVE functionality";
		break;
	case 0b100000: /* 0x20 */
		*dump_far = true;
		err = "Instruction Abort from a lower Exception level, that "
		      "might be using AArch32 or AArch64";
		break;
	case 0b100001: /* 0x21 */
		*dump_far = true;
		err = "Instruction Abort taken without a change in Exception "
		      "level.";
		break;
	case 0b100010: /* 0x22 */
		*dump_far = true;
		err = "PC alignment fault exception.";
		break;
	case 0b100100: /* 0x24 */
		*dump_far = true;
		err = "Data Abort from a lower Exception level, that might "
		      "be using AArch32 or AArch64";
		break;
	case 0b100101: /* 0x25 */
		*dump_far = true;
		err = "Data Abort taken without a change in Exception level";
		break;
	case 0b100110: /* 0x26 */
		err = "SP alignment fault exception";
		break;
	case 0b101000: /* 0x28 */
		err = "Trapped floating-point exception taken from AArch32 "
		      "state";
		break;
	case 0b101100: /* 0x2c */
		err = "Trapped floating-point exception taken from AArch64 "
		      "state.";
		break;
	case 0b101111: /* 0x2f */
		err = "SError interrupt";
		break;
	case 0b110000: /* 0x30 */
		err = "Breakpoint exception from a lower Exception level, "
		      "that might be using AArch32 or AArch64";
		break;
	case 0b110001: /* 0x31 */
		err = "Breakpoint exception taken without a change in "
		      "Exception level";
		break;
	case 0b110010: /* 0x32 */
		err = "Software Step exception from a lower Exception level, "
		      "that might be using AArch32 or AArch64";
		break;
	case 0b110011: /* 0x33 */
		err = "Software Step exception taken without a change in "
		      "Exception level";
		break;
	case 0b110100: /* 0x34 */
		*dump_far = true;
		err = "Watchpoint exception from a lower Exception level, "
		      "that might be using AArch32 or AArch64";
		break;
	case 0b110101: /* 0x35 */
		*dump_far = true;
		err = "Watchpoint exception taken without a change in "
		      "Exception level.";
		break;
	case 0b111000: /* 0x38 */
		err = "BKPT instruction execution in AArch32 state";
		break;
	case 0b111100: /* 0x3c */
		err = "BRK instruction execution in AArch64 state.";
		break;
	default:
		err = "Unknown";
	}

	EXCEPTION_DUMP("ESR_ELn: 0x%016llx", esr);
	EXCEPTION_DUMP("  EC:  0x%llx (%s)", GET_ESR_EC(esr), err);
	EXCEPTION_DUMP("  IL:  0x%llx", GET_ESR_IL(esr));
	EXCEPTION_DUMP("  ISS: 0x%llx", GET_ESR_ISS(esr));
}

static void esf_dump(const struct arch_esf *esf)
{
	EXCEPTION_DUMP("x0:  0x%016llx  x1:  0x%016llx", esf->x0, esf->x1);
	EXCEPTION_DUMP("x2:  0x%016llx  x3:  0x%016llx", esf->x2, esf->x3);
	EXCEPTION_DUMP("x4:  0x%016llx  x5:  0x%016llx", esf->x4, esf->x5);
	EXCEPTION_DUMP("x6:  0x%016llx  x7:  0x%016llx", esf->x6, esf->x7);
	EXCEPTION_DUMP("x8:  0x%016llx  x9:  0x%016llx", esf->x8, esf->x9);
	EXCEPTION_DUMP("x10: 0x%016llx  x11: 0x%016llx", esf->x10, esf->x11);
	EXCEPTION_DUMP("x12: 0x%016llx  x13: 0x%016llx", esf->x12, esf->x13);
	EXCEPTION_DUMP("x14: 0x%016llx  x15: 0x%016llx", esf->x14, esf->x15);
	EXCEPTION_DUMP("x16: 0x%016llx  x17: 0x%016llx", esf->x16, esf->x17);
	EXCEPTION_DUMP("x18: 0x%016llx  lr:  0x%016llx", esf->x18, esf->lr);
}
#endif /* CONFIG_EXCEPTION_DEBUG */

#ifdef CONFIG_ARCH_STACKWALK
typedef bool (*arm64_stacktrace_cb)(void *cookie, unsigned long addr, void *fp);

static bool is_address_mapped(uint64_t *addr)
{
	uintptr_t *phys = NULL;

	if (*addr == 0U) {
		return false;
	}

	/* Check alignment. */
	if ((*addr & (sizeof(uint32_t) - 1U)) != 0U) {
		return false;
	}

	return !arch_page_phys_get((void *) addr, phys);
}

static bool is_valid_jump_address(uint64_t *addr)
{
	if (*addr == 0U) {
		return false;
	}

	/* Check alignment. */
	if ((*addr & (sizeof(uint32_t) - 1U)) != 0U) {
		return false;
	}

	return ((*addr >= (uint64_t)__text_region_start) &&
		(*addr <= (uint64_t)(__text_region_end)));
}

static void walk_stackframe(arm64_stacktrace_cb cb, void *cookie, const struct arch_esf *esf,
			    int max_frames)
{
	/*
	 * For GCC:
	 *
	 *  ^  +-----------------+
	 *  |  |                 |
	 *  |  |                 |
	 *  |  |                 |
	 *  |  |                 |
	 *  |  | function stack  |
	 *  |  |                 |
	 *  |  |                 |
	 *  |  |                 |
	 *  |  |                 |
	 *  |  +-----------------+
	 *  |  |       LR        |
	 *  |  +-----------------+
	 *  |  |   previous FP   | <---+ FP
	 *  +  +-----------------+
	 */

	uint64_t *fp;
	uint64_t lr;

	if (esf != NULL) {
		fp = (uint64_t *) esf->fp;
	} else {
		return;
	}

	for (int i = 0; (fp != NULL) && (i < max_frames); i++) {
		if (!is_address_mapped(fp))
			break;
		lr = fp[1];
		if (!is_valid_jump_address(&lr)) {
			break;
		}
		if (!cb(cookie, lr, fp)) {
			break;
		}
		fp = (uint64_t *) fp[0];
	}
}

void arch_stack_walk(stack_trace_callback_fn callback_fn, void *cookie,
		     const struct k_thread *thread, const struct arch_esf *esf)
{
	ARG_UNUSED(thread);

	walk_stackframe((arm64_stacktrace_cb)callback_fn, cookie, esf,
			CONFIG_ARCH_STACKWALK_MAX_FRAMES);
}
#endif /* CONFIG_ARCH_STACKWALK */

#ifdef CONFIG_EXCEPTION_STACK_TRACE
static bool print_trace_address(void *arg, unsigned long lr, void *fp)
{
	int *i = arg;
#ifdef CONFIG_SYMTAB
	uint32_t offset = 0;
	const char *name = symtab_find_symbol_name(lr, &offset);

	EXCEPTION_DUMP("     %d: fp: 0x%016llx lr: 0x%016lx [%s+0x%x]",
			(*i)++, (uint64_t)fp, lr, name, offset);
#else
	EXCEPTION_DUMP("     %d: fp: 0x%016llx lr: 0x%016lx",
			(*i)++, (uint64_t)fp, lr);
#endif /* CONFIG_SYMTAB */

	return true;
}

static void esf_unwind(const struct arch_esf *esf)
{
	int i = 0;

	EXCEPTION_DUMP("");
	EXCEPTION_DUMP("call trace:");
	walk_stackframe(print_trace_address, &i, esf, CONFIG_ARCH_STACKWALK_MAX_FRAMES);
	EXCEPTION_DUMP("");
}
#endif /* CONFIG_EXCEPTION_STACK_TRACE */

#ifdef CONFIG_ARM64_STACK_PROTECTION
static bool z_arm64_stack_corruption_check(struct arch_esf *esf, uint64_t esr, uint64_t far)
{
	uint64_t sp, sp_limit, guard_start;
	/* 0x25 means data abort from current EL */
	if (GET_ESR_EC(esr) == 0x25) {
		sp_limit = arch_curr_cpu()->arch.current_stack_limit;
		guard_start = sp_limit - Z_ARM64_STACK_GUARD_SIZE;
		sp = arch_curr_cpu()->arch.corrupted_sp;
		if ((sp != 0 && sp <= sp_limit) || (guard_start <= far && far <= sp_limit)) {
#ifdef CONFIG_FPU_SHARING
			/*
			 * We are in exception stack, and now we are sure the stack does overflow,
			 * so flush the fpu context to its owner, and then set no fpu trap to avoid
			 * a new nested exception triggered by FPU accessing (var_args).
			 */
			arch_flush_local_fpu();
			write_cpacr_el1(read_cpacr_el1() | CPACR_EL1_FPEN_NOTRAP);
#endif
			arch_curr_cpu()->arch.corrupted_sp = 0UL;
			EXCEPTION_DUMP("STACK OVERFLOW FROM KERNEL,"
				" SP: 0x%llx OR FAR: 0x%llx INVALID,"
				" SP LIMIT: 0x%llx", sp, far, sp_limit);
			return true;
		}
	}
#ifdef CONFIG_USERSPACE
	else if ((_current->base.user_options & K_USER) != 0 && GET_ESR_EC(esr) == 0x24) {
		sp_limit = (uint64_t)_current->stack_info.start;
		guard_start = sp_limit - Z_ARM64_STACK_GUARD_SIZE;
		sp = esf->sp;
		if (sp <= sp_limit || (guard_start <= far && far <= sp_limit)) {
			EXCEPTION_DUMP("STACK OVERFLOW FROM USERSPACE,"
					"SP: 0x%llx OR FAR: 0x%llx INVALID,"
					" SP LIMIT: 0x%llx", sp, far, sp_limit);
			return true;
		}
	}
#endif
	return false;
}
#endif

static bool is_recoverable(struct arch_esf *esf, uint64_t esr, uint64_t far,
			   uint64_t elr)
{
	ARG_UNUSED(esr);
	ARG_UNUSED(far);
	ARG_UNUSED(elr);

	if (!esf) {
		return false;
	}

#ifdef CONFIG_USERSPACE
	for (int i = 0; i < ARRAY_SIZE(exceptions); i++) {
		/* Mask out instruction mode */
		uint64_t start = (uint64_t)exceptions[i].start;
		uint64_t end = (uint64_t)exceptions[i].end;

		if (esf->elr >= start && esf->elr < end) {
			esf->elr = (uint64_t)(exceptions[i].fixup);
			return true;
		}
	}
#endif

	return false;
}

void z_arm64_fatal_error(unsigned int reason, struct arch_esf *esf)
{
	uint64_t esr = 0;
	uint64_t elr = 0;
	uint64_t far = 0;
	uint64_t el;

	if (reason != K_ERR_SPURIOUS_IRQ) {
		el = read_currentel();

		switch (GET_EL(el)) {
		case MODE_EL1:
			esr = read_esr_el1();
			far = read_far_el1();
			elr = read_elr_el1();
			break;
#if !defined(CONFIG_ARMV8_R)
		case MODE_EL3:
			esr = read_esr_el3();
			far = read_far_el3();
			elr = read_elr_el3();
			break;
#endif /* CONFIG_ARMV8_R */
		}

#ifdef CONFIG_ARM64_STACK_PROTECTION
		if (z_arm64_stack_corruption_check(esf, esr, far)) {
			reason = K_ERR_STACK_CHK_FAIL;
		}
#endif

		if (IS_ENABLED(CONFIG_DEMAND_PAGING) &&
		    reason != K_ERR_STACK_CHK_FAIL &&
		    z_arm64_do_demand_paging(esf, esr, far)) {
			return;
		}

		if (GET_EL(el) != MODE_EL0) {
#ifdef CONFIG_EXCEPTION_DEBUG
			bool dump_far = false;

			EXCEPTION_DUMP("ELR_ELn: 0x%016llx", elr);

			dump_esr(esr, &dump_far);

			if (dump_far) {
				EXCEPTION_DUMP("FAR_ELn: 0x%016llx", far);
			}

			EXCEPTION_DUMP("TPIDRRO: 0x%016llx", read_tpidrro_el0());
#endif /* CONFIG_EXCEPTION_DEBUG */

			if (is_recoverable(esf, esr, far, elr) &&
			    reason != K_ERR_STACK_CHK_FAIL) {
				return;
			}
		}
	}

#ifdef CONFIG_EXCEPTION_DEBUG
	if (esf != NULL) {
		esf_dump(esf);
	}

#ifdef CONFIG_EXCEPTION_STACK_TRACE
	esf_unwind(esf);
#endif /* CONFIG_EXCEPTION_STACK_TRACE */
#endif /* CONFIG_EXCEPTION_DEBUG */

	z_fatal_error(reason, esf);
}

/**
 * @brief Handle a software-generated fatal exception
 * (e.g. kernel oops, panic, etc.).
 *
 * @param esf exception frame
 */
void z_arm64_do_kernel_oops(struct arch_esf *esf)
{
	/* x8 holds the exception reason */
	unsigned int reason = esf->x8;

#if defined(CONFIG_USERSPACE)
	/*
	 * User mode is only allowed to induce oopses and stack check
	 * failures via software-triggered system fatal exceptions.
	 */
	if (((_current->base.user_options & K_USER) != 0) &&
		reason != K_ERR_STACK_CHK_FAIL) {
		reason = K_ERR_KERNEL_OOPS;
	}
#endif

	z_arm64_fatal_error(reason, esf);
}

#ifdef CONFIG_USERSPACE
FUNC_NORETURN void arch_syscall_oops(void *ssf_ptr)
{
	z_arm64_fatal_error(K_ERR_KERNEL_OOPS, ssf_ptr);
	CODE_UNREACHABLE;
}
#endif

#if defined(CONFIG_PM_CPU_OPS_PSCI)
FUNC_NORETURN void arch_system_halt(unsigned int reason)
{
	ARG_UNUSED(reason);

	(void)arch_irq_lock();

#ifdef CONFIG_POWEROFF
	sys_poweroff();
#endif /* CONFIG_POWEROFF */

	for (;;) {
		/* Spin endlessly as fallback */
	}
}
#endif
