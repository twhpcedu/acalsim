/*
 * Copyright 2023-2025 Playlab/ACAL
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * crt0.S - RISC-V C Runtime Startup Code
 *
 * This file provides the minimal C runtime environment for bare-metal
 * RISC-V programs. It initializes the CPU, sets up the stack, initializes
 * .data and .bss sections, and calls main().
 */

.section .text.start
.global _start

_start:
    # Disable interrupts during initialization
    csrci mstatus, 0x8

    # Set up trap vector (direct mode)
    la t0, trap_handler
    csrw mtvec, t0

    # Initialize global pointer (gp)
    # The global pointer is used for relaxed addressing of global variables
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop

    # Set up stack pointer
    # Stack grows downward from _stack_top
    la sp, _stack_top

    # Initialize .data section (copy from ROM to RAM)
    # This is needed if program is loaded from ROM
    la t0, __data_start_rom    # Source (in ROM)
    la t1, __data_start        # Destination (in RAM)
    la t2, __data_end          # End of .data

copy_data:
    bgeu t1, t2, clear_bss     # Skip if no data to copy
    lw t3, 0(t0)              # Load word from ROM
    sw t3, 0(t1)              # Store word to RAM
    addi t0, t0, 4
    addi t1, t1, 4
    j copy_data

clear_bss:
    # Clear .bss section (uninitialized data)
    la t0, __bss_start
    la t1, __bss_end

clear_bss_loop:
    bgeu t0, t1, init_done
    sw zero, 0(t0)
    addi t0, t0, 4
    j clear_bss_loop

init_done:
    # Call global constructors (if any)
    # For simple bare-metal programs, this can be omitted
    # call __libc_init_array

    # Call main function with argc=0, argv=NULL
    li a0, 0                   # argc = 0
    li a1, 0                   # argv = NULL
    call main

    # If main returns, call exit with return value
    # Save return value in s0
    mv s0, a0

    # Call global destructors (if any)
    # call __libc_fini_array

    # Exit: infinite loop with WFI (wait for interrupt)
exit_loop:
    wfi
    j exit_loop

# =============================================================================
# Trap Handler
# =============================================================================

# Trap vector - handles exceptions and interrupts
.align 4
.global trap_handler
trap_handler:
    # Save all registers to stack
    # This is a minimal save - a full handler would save more
    addi sp, sp, -128

    # Save x registers
    sw x1,  0(sp)   # ra
    sw x5,  4(sp)   # t0
    sw x6,  8(sp)   # t1
    sw x7,  12(sp)  # t2
    sw x10, 16(sp)  # a0
    sw x11, 20(sp)  # a1
    sw x12, 24(sp)  # a2
    sw x13, 28(sp)  # a3
    sw x14, 32(sp)  # a4
    sw x15, 36(sp)  # a5
    sw x16, 40(sp)  # a6
    sw x17, 44(sp)  # a7
    sw x28, 48(sp)  # t3
    sw x29, 52(sp)  # t4
    sw x30, 56(sp)  # t5
    sw x31, 60(sp)  # t6

    # Save CSRs
    csrr t0, mepc
    sw t0, 64(sp)
    csrr t0, mcause
    sw t0, 68(sp)
    csrr t0, mtval
    sw t0, 72(sp)

    # Call C trap handler
    # Pass mcause as argument
    csrr a0, mcause
    csrr a1, mepc
    csrr a2, mtval
    call trap_handler_c

    # Restore CSRs
    lw t0, 64(sp)
    csrw mepc, t0

    # Restore x registers
    lw x1,  0(sp)   # ra
    lw x5,  4(sp)   # t0
    lw x6,  8(sp)   # t1
    lw x7,  12(sp)  # t2
    lw x10, 16(sp)  # a0
    lw x11, 20(sp)  # a1
    lw x12, 24(sp)  # a2
    lw x13, 28(sp)  # a3
    lw x14, 32(sp)  # a4
    lw x15, 36(sp)  # a5
    lw x16, 40(sp)  # a6
    lw x17, 44(sp)  # a7
    lw x28, 48(sp)  # t3
    lw x29, 52(sp)  # t4
    lw x30, 56(sp)  # t5
    lw x31, 60(sp)  # t6

    addi sp, sp, 128

    # Return from trap
    mret

# =============================================================================
# Default Trap Handler (weak symbol - can be overridden)
# =============================================================================

.weak trap_handler_c
trap_handler_c:
    # Default trap handler - just return
    # Real implementation should be provided in C code
    ret

# =============================================================================
# Weak symbol for main (in case no main is provided)
# =============================================================================

.weak main
main:
    # If no main is provided, just return 0
    li a0, 0
    ret
