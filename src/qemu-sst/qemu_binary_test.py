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
QEMU-Binary SST Integration Test Configuration - Phase 2C

This SST Python configuration sets up a test environment for the
QEMUBinaryComponent which runs a RISC-V program in QEMU and
communicates with an SST device component via binary MMIO protocol.

Architecture:
  - QEMU subprocess running RISC-V bare-metal program
  - Binary MMIO communication via Unix socket
  - Direct memory-mapped I/O instead of serial text protocol
  - ACALSim echo device for testing

Components:
  - qemu0: QEMUBinaryComponent (runs QEMU subprocess with MMIO)
  - device0: ACALSimDeviceComponent (echo device)

Improvements over Phase 2B:
  - Binary protocol: No text parsing overhead
  - MMIO-based: Direct memory access instead of UART
  - ~10x performance improvement expected
"""

import sst
import os

# -----------------------------------------------------------------------------
# Simulation Parameters
# -----------------------------------------------------------------------------

# Get RISC-V binary path from environment or use default
binary_path = os.environ.get("RISCV_BINARY",
    "/home/user/projects/acalsim/src/qemu-sst/riscv-programs/mmio_test.elf")

# QEMU path
qemu_path = os.environ.get("QEMU_PATH", "qemu-system-riscv32")

# Unix socket path for MMIO
socket_path = "/tmp/qemu-sst-mmio.sock"

# Device base address (should match RISC-V program: SST_DEVICE_BASE in mmio_test.c)
device_base = 0x20000000

# Simulation time
sim_time_us = 1000  # 1ms should be enough for test

# -----------------------------------------------------------------------------
# SST Configuration
# -----------------------------------------------------------------------------

# Set global options
sst.setProgramOption("timebase", "1ps")
sst.setProgramOption("stop-at", f"{sim_time_us}us")

print("=" * 70)
print("QEMU-Binary SST Integration Test - Phase 2C")
print("=" * 70)
print(f"Binary:      {binary_path}")
print(f"QEMU:        {qemu_path}")
print(f"Socket:      {socket_path}")
print(f"Device base: 0x{device_base:08X}")
print(f"Sim time:    {sim_time_us} us")
print("=" * 70)
print()

# -----------------------------------------------------------------------------
# Component: QEMU Binary
# -----------------------------------------------------------------------------

qemu = sst.Component("qemu0", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock":       "1GHz",
    "verbose":     "2",
    "binary_path": binary_path,
    "qemu_path":   qemu_path,
    "socket_path": socket_path,
    "device_base": f"0x{device_base:08X}",
})

# -----------------------------------------------------------------------------
# Component: ACALSim Device (Echo Device)
# -----------------------------------------------------------------------------

device = sst.Component("device0", "acalsim.QEMUDevice")
device.addParams({
    "clock":        "1GHz",
    "base_addr":    str(device_base),
    "size":         "4096",
    "verbose":      "1",
    "echo_latency": "1",  # 1 cycle (minimal latency)
})

# -----------------------------------------------------------------------------
# Links
# -----------------------------------------------------------------------------

# Link between QEMU and device
qemu_device_link = sst.Link("qemu_device_link")
qemu_device_link.connect(
    (qemu, "device_port", "1ns"),
    (device, "cpu_port", "1ns")
)

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

sst.setStatisticLoadLevel(5)

print("Configuration complete!")
print()
print("Expected behavior:")
print("  1. QEMU launches with RISC-V MMIO test program")
print("  2. Program accesses SST device via MMIO (0x20000000)")
print("  3. QEMUBinaryComponent receives binary MMIO requests")
print("  4. Requests translated to MemoryTransactionEvent")
print("  5. Device processes and returns MemoryResponseEvent")
print("  6. Binary MMIO responses sent back to QEMU")
print("  7. Program validates results and reports pass/fail")
print()
print("Watch for:")
print("  - QEMU startup messages")
print("  - MMIO read/write operations")
print("  - Binary protocol communication (no text parsing!)")
print("  - Device transaction logs")
print("  - Test pass/fail messages from RISC-V program")
print()
print("Phase 2C Status:")
print("  [READY] SST component framework")
print("  [READY] RISC-V test program")
print("  [TODO]  QEMU custom device (sst-device.so)")
print("  ")
print("  NOTE: QEMU device not yet implemented. Component will launch")
print("        QEMU but socket connection will timeout (expected).")
print("        This is Phase 2C.1 - component framework testing.")
print("=" * 70)
