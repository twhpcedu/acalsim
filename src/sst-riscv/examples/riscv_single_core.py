#!/usr/bin/env python3
# Copyright 2023-2026 Playlab/ACAL
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
"""
SST Configuration for Single RISC-V RV32I Processor

This example demonstrates a standalone RISC-V processor running
an assembly program within the SST simulation framework.

Features:
- Complete RV32I ISA implementation
- Event-driven timing model
- Pipeline visualization (IF, EXE, WB stages)
- Data and control hazard handling
- Assembly program execution

Supported Instructions:
- Arithmetic: ADD, SUB, ADDI
- Logical: AND, OR, XOR, ANDI, ORI, XORI
- Shift: SLL, SRL, SRA, SLLI, SRLI, SRAI
- Comparison: SLT, SLTU, SLTI, SLTIU
- Memory: LB, LBU, LH, LHU, LW, SB, SH, SW
- Branch: BEQ, BNE, BLT, BLTU, BGE, BGEU
- Jump: JAL, JALR
- Upper Immediate: LUI, AUIPC
- Halt: HCF (Halt and Catch Fire)
"""

import sst
import os

# ========== Configuration Parameters ==========

# Processor configuration
CLOCK_FREQ = "1GHz"
MAX_CYCLES = 100000  # Maximum simulation cycles (0 = unlimited)

# Memory configuration
MEMORY_SIZE = 65536  # 64KB
TEXT_OFFSET = 0  # Text segment starts at address 0
DATA_OFFSET = 8192  # Data segment starts at address 8KB

# Assembly program selection
# Available programs in src/riscv/asm/:
# - branch_simple.txt  : Simple branch test
# - full_test.txt      : Comprehensive ISA test
# - load_store_simple.txt : Memory access test
# - test.txt           : Basic arithmetic test

ASM_FILE = "/tmp/simple_test.txt"

# Verbosity (0-5, higher = more detailed)
VERBOSE = 2

# ========== Validate Assembly File ==========

if not os.path.exists(ASM_FILE):
	print(f"Error: Assembly file not found: {ASM_FILE}")
	print("Available assembly files:")
	asm_dir = "../../src/riscv/asm"
	if os.path.exists(asm_dir):
		for f in os.listdir(asm_dir):
			if f.endswith(".txt"):
				print(f"  - {asm_dir}/{f}")
	exit(1)

# ========== Component Creation ==========

print("Creating RISC-V RV32I Single-Core System...")

# Create RISC-V processor component
riscv_cpu = sst.Component("riscv_cpu", "acalsim.RISCVSoCStandalone")
riscv_cpu.addParams({
    "clock": CLOCK_FREQ,
    "asm_file": ASM_FILE,
    "memory_size": MEMORY_SIZE,
    "text_offset": TEXT_OFFSET,
    "data_offset": DATA_OFFSET,
    "max_cycles": MAX_CYCLES,
    "verbose": VERBOSE
})

# ========== Statistics Configuration ==========

# Enable statistics collection
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputConsole")

# Enable all statistics for the processor
# riscv_cpu.enableAllStatistics({
#     "type": "sst.AccumulatorStatistic",
#     "rate": "1us"
# })

# ========== Simulation Configuration ==========

# Optional: Set simulation end time
# sst.setProgramOption("stop-at", "1ms")

# ========== Configuration Summary ==========

print("\n" + "=" * 60)
print("RISC-V RV32I Single-Core Configuration")
print("=" * 60)
print(f"Clock frequency:     {CLOCK_FREQ}")
print(f"Max cycles:          {MAX_CYCLES if MAX_CYCLES > 0 else 'Unlimited'}")
print(f"Memory size:         {MEMORY_SIZE} bytes ({MEMORY_SIZE // 1024}KB)")
print(f"Text segment:        0x{TEXT_OFFSET:08X}")
print(f"Data segment:        0x{DATA_OFFSET:08X}")
print(f"Assembly program:    {ASM_FILE}")
print(f"Verbosity level:     {VERBOSE}")
print("=" * 60)
print()
print("Running RISC-V simulation...")
print("The simulator will execute instructions until:")
print("  1. HCF (Halt and Catch Fire) instruction is encountered")
print(f"  2. Maximum cycles ({MAX_CYCLES}) is reached")
print("  3. No more events to process")
print()
