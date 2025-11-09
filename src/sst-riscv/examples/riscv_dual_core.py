#!/usr/bin/env python3
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
"""
SST Configuration for Dual RISC-V RV32I Processors

This example demonstrates multiple RISC-V processors running in parallel
within SST, each executing different assembly programs.

System Architecture:
    [CPU0] --- running program A
    [CPU1] --- running program B

This showcases:
- Multiple ACALSim components in a single SST simulation
- Independent program execution on each core
- Parallel simulation of multiple processors
- Per-core statistics collection
"""

import sst
import os

# ========== Configuration Parameters ==========

# System configuration
NUM_CORES = 2
CLOCK_FREQ = "1GHz"
MAX_CYCLES = 100000

# Memory configuration (per core)
MEMORY_SIZE = 65536
TEXT_OFFSET = 0
DATA_OFFSET = 8192

# Assembly programs for each core
ASM_FILES = [
    "../../src/riscv/asm/branch_simple.txt",  # Core 0
    "../../src/riscv/asm/load_store_simple.txt"  # Core 1
]

VERBOSE = 1

# ========== Validation ==========

print("Validating assembly files...")
for i, asm_file in enumerate(ASM_FILES):
	if not os.path.exists(asm_file):
		print(f"Error: Assembly file for core {i} not found: {asm_file}")
		exit(1)
	print(f"  Core {i}: {asm_file} - OK")

# ========== Component Creation ==========

print(f"\nCreating {NUM_CORES}-core RISC-V system...")

cores = []
for i in range(NUM_CORES):
	core_name = f"riscv_cpu{i}"
	print(f"  Creating {core_name}...")

	cpu = sst.Component(core_name, "acalsim.RISCVSoCStandalone")
	cpu.addParams({
	    "clock": CLOCK_FREQ,
	    "asm_file": ASM_FILES[i],
	    "memory_size": MEMORY_SIZE,
	    "text_offset": TEXT_OFFSET,
	    "data_offset": DATA_OFFSET,
	    "max_cycles": MAX_CYCLES,
	    "verbose": VERBOSE
	})

	cores.append(cpu)

# ========== Statistics Configuration ==========

print("Configuring statistics...")

sst.setStatisticLoadLevel(7)
sst.setStatisticOutput(
    "sst.statOutputCSV", {
        "filepath": "riscv_dual_core_stats.csv",
        "separator": ","
    }
)

# Enable statistics for each core
# for i, cpu in enumerate(cores):
#     cpu.enableAllStatistics({
#         "type": "sst.AccumulatorStatistic",
#         "rate": "10us"
#     })

# ========== Configuration Summary ==========

print("\n" + "=" * 70)
print("RISC-V RV32I Dual-Core Configuration")
print("=" * 70)
print(f"Number of cores:     {NUM_CORES}")
print(f"Clock frequency:     {CLOCK_FREQ}")
print(f"Max cycles:          {MAX_CYCLES}")
print(f"Memory per core:     {MEMORY_SIZE} bytes ({MEMORY_SIZE // 1024}KB)")
print()
print("Core Programs:")
for i, asm_file in enumerate(ASM_FILES):
	print(f"  Core {i}: {os.path.basename(asm_file)}")
print()
print("Statistics output:   riscv_dual_core_stats.csv")
print("=" * 70)
print()
print("Running dual-core RISC-V simulation...")
print("Each core executes independently with its own:")
print("  - Program counter (PC)")
print("  - Register file (x0-x31)")
print("  - Instruction memory (IMEM)")
print("  - Data memory (DMEM)")
print("  - Pipeline stages (IF, EXE, WB)")
print()
