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
QEMU-Real SST Integration Test Configuration

This SST Python configuration sets up a test environment for the
QEMURealComponent which runs an actual RISC-V program in QEMU and
communicates with an SST device component.

Architecture:
  - QEMU subprocess running RISC-V bare-metal program
  - Serial communication via Unix socket
  - SST protocol parsing and event translation
  - ACALSim echo device for testing

Components:
  - qemu0: QEMURealComponent (runs QEMU subprocess)
  - device0: ACALSimDeviceComponent (echo device)
"""

import sst
import os

# -----------------------------------------------------------------------------
# Simulation Parameters
# -----------------------------------------------------------------------------

# Get RISC-V binary path from environment or use default
binary_path = os.environ.get(
    "RISCV_BINARY", "/home/user/projects/acalsim/src/qemu-sst/riscv-programs/sst_device_test.elf"
)

# QEMU path
qemu_path = os.environ.get("QEMU_PATH", "qemu-system-riscv32")

# Unix socket path for QEMU serial
socket_path = "/tmp/qemu-sst-test.sock"

# Device base address (should match RISC-V program: SST_DEVICE_BASE in sst_device.h)
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
print("QEMU-Real SST Integration Test")
print("=" * 70)
print(f"Binary:      {binary_path}")
print(f"QEMU:        {qemu_path}")
print(f"Socket:      {socket_path}")
print(f"Device base: 0x{device_base:08X}")
print(f"Sim time:    {sim_time_us} us")
print("=" * 70)
print()

# -----------------------------------------------------------------------------
# Component: QEMU Real
# -----------------------------------------------------------------------------

qemu = sst.Component("qemu0", "qemureal.QEMUReal")
qemu.addParams({
    "clock": "1GHz",
    "verbose": "2",
    "binary_path": binary_path,
    "qemu_path": qemu_path,
    "socket_path": socket_path,
})

# -----------------------------------------------------------------------------
# Component: ACALSim Device (Echo Device)
# -----------------------------------------------------------------------------

device = sst.Component("device0", "acalsim.QEMUDevice")
device.addParams({
    "clock": "1GHz",
    "base_addr": str(device_base),
    "size": "4096",
    "verbose": "1",  # Reduce verbosity for cleaner output
    "echo_latency": "1",  # 1 cycle (minimal latency for testing)
})

# -----------------------------------------------------------------------------
# Links
# -----------------------------------------------------------------------------

# Link between QEMU and device
qemu_device_link = sst.Link("qemu_device_link")
qemu_device_link.connect((qemu, "device_port", "1ns"), (device, "cpu_port", "1ns"))

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

sst.setStatisticLoadLevel(5)

print("Configuration complete!")
print()
print("Expected behavior:")
print("  1. QEMU launches with RISC-V test program")
print("  2. Program sends SST protocol commands via serial")
print("  3. QEMURealComponent parses and translates to SST events")
print("  4. Device processes requests and sends responses")
print("  5. Responses forwarded back to QEMU via serial")
print("  6. Program validates echo operation and reports results")
print()
print("Watch for:")
print("  - QEMU startup messages")
print("  - Serial protocol commands (SST:WRITE:ADDR:DATA)")
print("  - Device transaction logs")
print("  - Test pass/fail messages from RISC-V program")
print("=" * 70)
