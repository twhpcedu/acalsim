# Copyright 2023-2025 Playlab/ACAL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RISC-V Bare-Metal Programs Makefile

# Toolchain
PREFIX = riscv64-unknown-elf-
CC = $(PREFIX)gcc
AS = $(PREFIX)as
LD = $(PREFIX)ld
OBJCOPY = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump

# Flags
ARCH = rv32imac
ABI = ilp32
CFLAGS = -march=$(ARCH) -mabi=$(ABI) -nostdlib -ffreestanding -O2 -Wall -Wextra
ASFLAGS = -march=$(ARCH) -mabi=$(ABI)
LDFLAGS = -T linker.ld -nostdlib

# Targets
all: simple_test.elf simple_test.bin simple_test.dump sst_device_test.elf sst_device_test.bin sst_device_test.dump mmio_test.elf mmio_test.bin mmio_test.dump

simple_test.elf: start.o simple_test.o linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) start.o simple_test.o -o $@

simple_test.bin: simple_test.elf
	$(OBJCOPY) -O binary $< $@

simple_test.dump: simple_test.elf
	$(OBJDUMP) -d $< > $@

start.o: start.S
	$(CC) $(CFLAGS) -c $< -o $@

simple_test.o: simple_test.c
	$(CC) $(CFLAGS) -c $< -o $@

# SST device test
sst_device_test.elf: start.o sst_device_test.o sst_device.o linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) start.o sst_device_test.o sst_device.o -o $@

sst_device_test.bin: sst_device_test.elf
	$(OBJCOPY) -O binary $< $@

sst_device_test.dump: sst_device_test.elf
	$(OBJDUMP) -d $< > $@

sst_device_test.o: sst_device_test.c sst_device.h
	$(CC) $(CFLAGS) -c $< -o $@

sst_device.o: sst_device.c sst_device.h
	$(CC) $(CFLAGS) -c $< -o $@

# MMIO test (Phase 2C)
mmio_test.elf: start.o mmio_test.o linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) start.o mmio_test.o -o $@

mmio_test.bin: mmio_test.elf
	$(OBJCOPY) -O binary $< $@

mmio_test.dump: mmio_test.elf
	$(OBJDUMP) -d $< > $@

mmio_test.o: mmio_test.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.elf *.bin *.dump

# Test with QEMU
test: simple_test.elf
	@echo "Running RISC-V program in QEMU..."
	@echo "Press Ctrl-A then X to exit QEMU"
	qemu-system-riscv32 -M virt -bios none -nographic -kernel simple_test.elf

# Test SST device program with QEMU
test-sst: sst_device_test.elf
	@echo "Running SST device test program in QEMU..."
	@echo "Press Ctrl-A then X to exit QEMU"
	qemu-system-riscv32 -M virt -bios none -nographic -kernel sst_device_test.elf

# Test MMIO program with QEMU
test-mmio: mmio_test.elf
	@echo "Running MMIO test program in QEMU..."
	@echo "Press Ctrl-A then X to exit QEMU"
	qemu-system-riscv32 -M virt -bios none -nographic -kernel mmio_test.elf

# Debug with QEMU
debug: simple_test.elf
	@echo "Starting QEMU in debug mode..."
	@echo "Connect with: riscv64-unknown-elf-gdb simple_test.elf"
	@echo "Then run: target remote localhost:1234"
	qemu-system-riscv32 -M virt -bios none -nographic -kernel simple_test.elf -s -S

.PHONY: all clean test test-sst test-mmio debug
