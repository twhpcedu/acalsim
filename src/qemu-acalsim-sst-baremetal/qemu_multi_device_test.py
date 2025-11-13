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
QEMU Multi-Device SST Integration Test Configuration

This SST Python configuration demonstrates advanced multi-device integration:
  - QEMU subprocess running RISC-V bare-metal program
  - Two SST device components (Echo and Compute)
  - QEMU communicates with both devices via binary MMIO protocol
  - Devices communicate with each other via SST links

Architecture:

  ┌─────────────────────────────────────────────────────────┐
  │                    SST Simulation                       │
  │                                                          │
  │  ┌───────────────┐                                      │
  │  │               │  device1_port   ┌───────────────┐   │
  │  │  QEMUBinary   ├────────────────►│ Echo Device   │   │
  │  │  Component    │                 │ (0x10200000)  │   │
  │  │               │                 └───────┬───────┘   │
  │  │               │                         │            │
  │  │               │  device2_port           │ peer_port  │
  │  │               ├──────────┐              │            │
  │  └───────────────┘          │              │            │
  │                              │              ▼            │
  │                              │     ┌────────────────┐   │
  │                              └────►│ Compute Device │   │
  │                                    │ (0x10300000)   │   │
  │                                    └────────────────┘   │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

Components:
  - qemu0: QEMUBinaryComponent (runs QEMU with multi_device_test.elf)
  - echo_dev: ACALSimDeviceComponent (echo device at 0x10200000)
  - compute_dev: ACALSimComputeDeviceComponent (compute device at 0x10300000)

Links:
  - qemu_echo_link: QEMU ↔ Echo Device (CPU transactions)
  - qemu_compute_link: QEMU ↔ Compute Device (CPU transactions)
  - device_peer_link: Echo Device ↔ Compute Device (peer communication)
"""

import sst
import os

# -----------------------------------------------------------------------------
# Simulation Parameters
# -----------------------------------------------------------------------------

# Get RISC-V binary path
binary_path = os.environ.get(
    "RISCV_BINARY",
    "/home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal/riscv-programs/multi_device_test.elf"
)

# QEMU path
qemu_path = os.environ.get("QEMU_PATH", "/home/user/qemu-build/install/bin/qemu-system-riscv32")

# Unix socket path for MMIO
socket_path = "/tmp/qemu-sst-mmio.sock"

# Device addresses (must match test program)
echo_device_base = 0x10200000
compute_device_base = 0x10300000

# Simulation time
sim_time_us = 5000  # 5ms for more complex test

# -----------------------------------------------------------------------------
# SST Configuration
# -----------------------------------------------------------------------------

# Set global options
sst.setProgramOption("timebase", "1ps")
sst.setProgramOption("stop-at", f"{sim_time_us}us")

print("=" * 70)
print("QEMU Multi-Device SST Integration Test")
print("=" * 70)
print(f"Binary:          {binary_path}")
print(f"QEMU:            {qemu_path}")
print(f"Socket:          {socket_path}")
print(f"Echo device:     0x{echo_device_base:08X}")
print(f"Compute device:  0x{compute_device_base:08X}")
print(f"Sim time:        {sim_time_us} us")
print("=" * 70)
print()

# -----------------------------------------------------------------------------
# Component: QEMU Binary
# -----------------------------------------------------------------------------

# Note: The standard QEMUBinary component has a single device_port.
# For multi-device support, the QEMU component should route based on address,
# or we need a modified version with multiple device ports.
#
# For this example, we'll use a simple approach where QEMU sends all requests
# to device1 (echo), and that device can forward to device2 if needed based
# on address ranges.
#
# A production implementation would modify QEMUBinaryComponent to support
# multiple device ports with address-based routing.

qemu = sst.Component("qemu0", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "1GHz",
    "verbose": "2",
    "binary_path": binary_path,
    "qemu_path": qemu_path,
    "socket_path": socket_path,
    "device_base": f"0x{echo_device_base:08X}",  # Base for routing
})

# -----------------------------------------------------------------------------
# Component: Echo Device (Device 1)
# -----------------------------------------------------------------------------

echo_dev = sst.Component("echo_dev", "acalsim.QEMUDevice")
echo_dev.addParams({
    "clock": "1GHz",
    "base_addr": str(echo_device_base),
    "size": "4096",
    "verbose": "2",
    "echo_latency": "10",  # 10 cycles
})

# -----------------------------------------------------------------------------
# Component: Compute Device (Device 2)
# -----------------------------------------------------------------------------

compute_dev = sst.Component("compute_dev", "acalsim.ComputeDevice")
compute_dev.addParams({
    "clock": "1GHz",
    "base_addr": str(compute_device_base),
    "size": "4096",
    "verbose": "2",
    "compute_latency": "100",  # 100 cycles for computation
})

# -----------------------------------------------------------------------------
# Links
# -----------------------------------------------------------------------------

# Link 1: QEMU to Echo Device (for echo device transactions)
# In this simplified configuration, all QEMU transactions go through echo device
# The echo device will handle its own addresses (0x10200000-0x10200FFF)
qemu_echo_link = sst.Link("qemu_echo_link")
qemu_echo_link.connect((qemu, "device_port", "1ns"), (echo_dev, "cpu_port", "1ns"))

# For proper multi-device support, we would need:
# - Modified QEMUBinaryComponent with device1_port and device2_port
# - Address-based routing in QEMUBinaryComponent
# - Separate links for each device

# Link 2: Peer communication between Echo and Compute devices
# This allows devices to exchange data directly
device_peer_link = sst.Link("device_peer_link")
device_peer_link.connect(
    (echo_dev, "peer_port", "10ns"),  # Echo device peer port
    (compute_dev, "peer_port", "10ns")  # Compute device peer port
)

# Note: The current implementation has a limitation - we need to either:
# 1. Extend QEMUBinaryComponent to support multiple device ports, OR
# 2. Create a router component that distributes transactions based on address, OR
# 3. Modify the echo device to forward compute device transactions
#
# For this demonstration, we'll create a simplified version that shows the
# architecture, even though it won't fully work without QEMUBinaryComponent
# modifications.

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

sst.setStatisticLoadLevel(5)

print("Configuration complete!")
print()
print("Architecture:")
print("  - QEMU launches with multi_device_test.elf")
print("  - Echo device at 0x10200000 (data echo operations)")
print("  - Compute device at 0x10300000 (arithmetic operations)")
print("  - Devices connected via peer link for inter-device communication")
print()
print("Note:")
print("  This configuration demonstrates the multi-device architecture.")
print("  For full functionality, QEMUBinaryComponent needs address-based")
print("  routing to multiple device ports. Current implementation routes")
print("  all transactions through echo device.")
print()
print("=" * 70)
