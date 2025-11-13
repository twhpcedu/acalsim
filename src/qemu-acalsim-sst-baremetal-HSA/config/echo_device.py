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

#!/usr/bin/env python3
"""
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
QEMU-ACALSim Echo Device Distributed Simulation Configuration

This SST configuration sets up a distributed simulation with two processes:
- Process 0 (MPI Rank 0): QEMU component simulating test program
- Process 1 (MPI Rank 1): ACALSim device component (echo device)

Usage:
    # Run with 2 MPI ranks on same node
    mpirun -n 2 sst echo_device.py

    # Run with 2 MPI ranks on different nodes
    mpirun -n 1 -host node0 sst echo_device.py : \
           -n 1 -host node1 sst echo_device.py

    # Run with verbose output
    mpirun -n 2 sst echo_device.py -- --verbose=2
"""

import sst
import sys

# Print immediately to verify both ranks execute this script
print(f"[STARTUP] Python script starting on process (before getting rank)", flush=True)
sys.stdout.flush()

# Configure SST for manual distributed simulation
sst.setProgramOption("partitioner", "self")

# Set timebase to prevent overflow with 1GHz clocks
sst.setProgramOption("timebase", "1ps")

# Set a reasonable stop-at time to prevent infinite simulation
sst.setProgramOption("stop-at", "100us")

# ==============================================================================
# Simulation Parameters
# ==============================================================================

# Clock frequency for both components
CLOCK_FREQ = "1GHz"

# Device memory-mapped region
DEVICE_BASE_ADDR = "0x10000000"  # 256MB base address
DEVICE_SIZE = "4096"  # 4KB device region

# Link latency (inter-process communication)
LINK_LATENCY = "1ns"

# Test parameters
TEST_PATTERN = "0xDEADBEEF"
NUM_ITERATIONS = "5"

# Verbosity levels (0=quiet, 1=normal, 2=verbose, 3=debug)
QEMU_VERBOSE = "2"
DEVICE_VERBOSE = "2"

# Echo operation latency (in device cycles)
ECHO_LATENCY = "10"

# ==============================================================================
# Component Configuration
# ==============================================================================

# Get MPI rank for informational purposes
rank = sst.getMyMPIRank()
nranks = sst.getMPIRankCount()

print(f"[DEBUG] SST Configuration: Rank {rank} of {nranks}", flush=True)

# Verify we have exactly 2 ranks
if nranks != 2:
	raise RuntimeError(f"This simulation requires exactly 2 MPI ranks, got {nranks}")

print(f"[DEBUG] Rank {rank}: Creating all components (will be distributed by SST)", flush=True)

# ==============================================================================
# Component Creation (All ranks create all components, SST distributes them)
# ==============================================================================

# ------------------------------------------------------------------------------
# QEMU Component (assigned to Rank 0)
# ------------------------------------------------------------------------------
print(f"[DEBUG] Rank {rank}: Creating QEMU component", flush=True)
qemu = sst.Component("qemu0", "qemu.RISCV")
qemu.setRank(0)  # This component runs on rank 0
qemu.addParams({
    "clock": CLOCK_FREQ,
    "device_base": DEVICE_BASE_ADDR,
    "device_size": DEVICE_SIZE,
    "verbose": QEMU_VERBOSE,
    "test_pattern": TEST_PATTERN,
    "num_iterations": NUM_ITERATIONS
})

# ------------------------------------------------------------------------------
# ACALSim Device Component (assigned to Rank 1)
# ------------------------------------------------------------------------------
print(f"[DEBUG] Rank {rank}: Creating ACALSim device component", flush=True)
device = sst.Component("device0", "acalsim.QEMUDevice")
device.setRank(1)  # This component runs on rank 1
device.addParams({
    "clock": CLOCK_FREQ,
    "base_addr": DEVICE_BASE_ADDR,
    "size": DEVICE_SIZE,
    "verbose": DEVICE_VERBOSE,
    "echo_latency": ECHO_LATENCY
})

# ------------------------------------------------------------------------------
# Link Connection
# ------------------------------------------------------------------------------
print(f"[DEBUG] Rank {rank}: Creating and connecting link", flush=True)
qemu_device_link = sst.Link("qemu_device_link")
qemu_device_link.setNoCut()  # Don't cut links for distributed simulation
qemu.addLink(qemu_device_link, "device_port", LINK_LATENCY)
device.addLink(qemu_device_link, "cpu_port", LINK_LATENCY)

print(f"[DEBUG] Rank {rank}: Finished configuration", flush=True)

# ==============================================================================
# Simulation Statistics
# ==============================================================================

# Enable statistics output
sst.setStatisticLoadLevel(7)

# ==============================================================================
# Print Configuration Summary
# ==============================================================================

if rank == 0:
	print("\n" + "=" * 70)
	print("QEMU-ACALSim Distributed Simulation Configuration")
	print("=" * 70)
	print(f"Clock Frequency:      {CLOCK_FREQ}")
	print(f"Device Base Address:  {DEVICE_BASE_ADDR}")
	print(f"Device Size:          {DEVICE_SIZE} bytes")
	print(f"Link Latency:         {LINK_LATENCY}")
	print(f"Test Pattern:         {TEST_PATTERN}")
	print(f"Test Iterations:      {NUM_ITERATIONS}")
	print(f"Echo Latency:         {ECHO_LATENCY} cycles")
	print("\nProcess Distribution:")
	print(f"  Rank 0: QEMU component (test program)")
	print(f"  Rank 1: ACALSim device (echo device)")
	print("=" * 70 + "\n")
