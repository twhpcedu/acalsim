/* Auto-generated linker script for ACALSim kernel */

ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 1048576
}

SECTIONS
{
    .text : {
        _start = .;
        *(.text.init)
        *(.text .text.*)
    } > RAM

    .rodata : {
        *(.rodata .rodata.*)
    } > RAM

    .data : {
        *(.data .data.*)
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    .stack : {
        . = ALIGN(16);
        . = . + 0x4000;  /* 16KB stack */
        __stack_top = .;
    } > RAM
}
