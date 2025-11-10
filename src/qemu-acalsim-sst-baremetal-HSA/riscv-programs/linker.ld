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
 * RISC-V Bare-Metal Linker Script
 * For QEMU virt machine
 *
 * Memory Layout:
 *   0x80000000: Start of RAM
 *   - .text section (code)
 *   - .rodata section (read-only data)
 *   - .data section (initialized data)
 *   - .bss section (uninitialized data)
 *   - Heap (if needed)
 *   - Stack (grows downward from top of RAM)
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
    /* Code section */
    .text : {
        /* Startup code comes first */
        KEEP(*(.text.start))
        *(.text*)
        *(.gnu.linkonce.t.*)
    } > RAM

    /* Read-only data section */
    .rodata : ALIGN(4) {
        *(.rodata*)
        *(.gnu.linkonce.r.*)
        . = ALIGN(4);
    } > RAM

    /* Data section (initialized data) */
    .data : ALIGN(4) {
        __data_start = .;
        *(.data*)
        *(.gnu.linkonce.d.*)
        . = ALIGN(4);
        __data_end = .;
    } > RAM

    /* For ROM-based systems, this would point to the ROM copy */
    /* For RAM-based systems (like QEMU), it's the same as __data_start */
    __data_start_rom = LOADADDR(.data);

    /* Global pointer - used for relaxed addressing */
    /* Points to middle of .sdata for optimal reach */
    __global_pointer$ = MIN(__data_start + 0x800, __data_end);

    /* BSS section (uninitialized data) */
    .bss : ALIGN(4) {
        __bss_start = .;
        *(.bss*)
        *(.gnu.linkonce.b.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > RAM

    /* End of used memory */
    _end = .;
    PROVIDE(end = .);

    /* Heap starts after BSS (if needed) */
    . = ALIGN(16);
    __heap_start = .;
    __heap_end = ORIGIN(RAM) + LENGTH(RAM) - 0x10000;  /* Reserve 64KB for stack */

    /* Stack grows down from end of RAM */
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
    __stack_size = 0x10000;  /* 64KB stack */

    /* Discard sections we don't need */
    /DISCARD/ : {
        *(.comment)
        *(.eh_frame)
        *(.note*)
    }
}
