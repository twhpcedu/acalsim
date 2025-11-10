/*
 * RISC-V Startup Code
 */

.section .text.start
.global _start

_start:
    # Set up stack pointer
    la sp, _stack_top

    # Clear BSS section
    la t0, __bss_start
    la t1, __bss_end
1:
    bgeu t0, t1, 2f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 1b

2:
    # Call main C function
    call _start_c

    # If we return, loop forever
3:
    wfi
    j 3b

# Trap/exception handler
.align 4
.global trap_handler
trap_handler:
    # Save context (simplified - just call C handler)
    call trap_handler_c
    mret
