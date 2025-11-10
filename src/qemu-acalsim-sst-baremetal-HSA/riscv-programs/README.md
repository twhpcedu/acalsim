# RISC-V Bare-Metal Programs

This directory contains RISC-V bare-metal programs for testing QEMU integration with SST.

## Overview

These programs demonstrate:
- Bare-metal RISC-V execution in QEMU
- UART output for debugging
- Basic C programming without standard library
- Foundation for SST device communication (Phase 2B+)

## Prerequisites

- RISC-V toolchain: `riscv64-unknown-elf-gcc`
- QEMU RISC-V: `qemu-system-riscv32`

### Installation (Docker Container)

```bash
# Install RISC-V toolchain
sudo apt-get install gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf

# Install QEMU
sudo apt-get install qemu-system-misc
```

## Building

```bash
# Build all programs
make

# Clean build artifacts
make clean
```

### Build Output

- `simple_test.elf` - Executable ELF file
- `simple_test.bin` - Raw binary
- `simple_test.dump` - Disassembly listing

## Running

### Run in QEMU

```bash
make test
```

Or manually:
```bash
qemu-system-riscv32 -M virt -bios none -nographic -kernel simple_test.elf
```

Press `Ctrl-A` then `X` to exit QEMU.

### Debug with GDB

Terminal 1:
```bash
make debug
```

Terminal 2:
```bash
riscv64-unknown-elf-gdb simple_test.elf
(gdb) target remote localhost:1234
(gdb) break _start
(gdb) continue
```

## Program Structure

### simple_test.c

Basic test program that demonstrates:

1. **UART Output**: Text output via memory-mapped UART
2. **Arithmetic**: Simple calculations
3. **Loops**: Iteration execution
4. **Function Calls**: C function invocation
5. **Exit**: Clean exit via QEMU test device

### start.S

Assembly startup code:
- Sets up stack pointer
- Clears BSS section
- Calls C entry point
- Provides trap handler

### linker.ld

Linker script for QEMU virt machine:
- Maps code to 0x80000000 (RAM start)
- Defines sections: .text, .rodata, .data, .bss
- Sets stack at end of RAM

## Memory Map

| Address Range | Description |
|--------------|-------------|
| 0x80000000 | RAM start (128MB) |
| 0x10000000 | UART base |
| 0x20000000 | SST device base (future) |
| 0x100000   | QEMU test device (exit) |

## Example Output

```
================================
RISC-V Bare-Metal Test Program
================================
Running in QEMU...

Test 1: Simple arithmetic
  a = 0xDEADBEEF
  b = 0xCAFEBABE
  c = a + b = 0xA9AC79AD

Test 2: Loop execution
  Iteration 0
  Iteration 1
  Iteration 2
  Iteration 3
  Iteration 4

Test 3: Function call
  test_function(10, 20) = 0x00000050

================================
All tests completed successfully!
================================
```

## Adding New Programs

1. Create new C file (e.g., `my_program.c`)
2. Update Makefile with new target:

```makefile
my_program.elf: start.o my_program.o linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) start.o my_program.o -o $@

my_program.o: my_program.c
	$(CC) $(CFLAGS) -c $< -o $@
```

3. Build and test:

```bash
make my_program.elf
qemu-system-riscv32 -M virt -bios none -nographic -kernel my_program.elf
```

## SST Device Integration (Phase 2B)

Future programs will communicate with SST devices:

```c
#define SST_DEVICE_BASE 0x20000000
#define SST_DATA_IN  ((volatile uint32_t*)(SST_DEVICE_BASE + 0x00))
#define SST_DATA_OUT ((volatile uint32_t*)(SST_DEVICE_BASE + 0x04))
#define SST_STATUS   ((volatile uint32_t*)(SST_DEVICE_BASE + 0x08))

// Write to SST device
*SST_DATA_IN = 0xDEADBEEF;

// Wait for device
while (*SST_STATUS & STATUS_BUSY);

// Read response
uint32_t result = *SST_DATA_OUT;
```

This will be integrated with SST's distributed simulation framework, allowing QEMU to communicate with SST components across MPI ranks.

## Troubleshooting

### QEMU fails with "Unable to load firmware"

Solution: Use `-bios none` flag to skip firmware loading.

### Program doesn't output anything

Check:
1. UART address is correct (0x10000000)
2. Using `-nographic` flag
3. Entry point is `_start` (check with `objdump`)

### Linking errors

Make sure:
1. Toolchain is installed: `riscv64-unknown-elf-gcc --version`
2. All object files are built
3. Linker script path is correct

## Architecture Notes

### RV32IMAC

- **RV32I**: Base integer instruction set
- **M**: Integer multiplication and division
- **A**: Atomic instructions
- **C**: Compressed instructions (16-bit)

### ABI: ilp32

- int, long, pointer = 32 bits
- Used for RV32 targets

### QEMU virt Machine

- Generic RISC-V virtual platform
- UART at 0x10000000
- RAM at 0x80000000
- Test device at 0x100000

## Next Steps

**Phase 2A** (Current): ✓ QEMU running bare-metal programs
**Phase 2B**: Connect QEMU to SST via custom device backend
**Phase 2C**: Full integration with distributed SST simulation
**Phase 3**: U-Boot support
**Phase 4**: Linux kernel support

## References

- [RISC-V Specifications](https://riscv.org/specifications/)
- [QEMU Documentation](https://www.qemu.org/docs/master/)
- [QEMU RISC-V](https://www.qemu.org/docs/master/system/target-riscv.html)
- [RISC-V Assembly Programmer's Manual](https://github.com/riscv-non-isa/riscv-asm-manual)
