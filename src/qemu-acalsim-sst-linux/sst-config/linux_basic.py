#!/usr/bin/env python3
"""
SST Configuration for Linux Integration - Basic Test

Copyright 2023-2025 Playlab/ACAL

This configuration sets up a basic SST simulation with a single
VirtIO SST device for Linux integration testing.

Usage:
    sst linux_basic.py
"""

import sst
import os

# Configuration parameters
SOCKET_PATH = "/tmp/qemu-sst-linux.sock"
DEVICE_ID = 0

# Component library (local build)
SST_LIB = os.getenv("SST_LIB", "../acalsim-device/libacalsim.so")

print("=" * 60)
print("SST Linux Integration - Basic Configuration")
print("=" * 60)
print(f"Socket: {SOCKET_PATH}")
print(f"Device ID: {DEVICE_ID}")
print(f"SST Library: {SST_LIB}")
print("=" * 60)

# Clean up existing socket
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)
    print(f"Removed existing socket: {SOCKET_PATH}")

#
# Create SST Device Component
#
print("Creating SST device component...")

sst_device = sst.Component("sst_device_0", "acalsim.VirtIODevice")
sst_device.addParams({
    "socket_path": SOCKET_PATH,
    "device_id": DEVICE_ID,
    "verbose": "2",  # Increased verbosity to see detailed request processing
    "clock": "1GHz"
})

#
# Optional: Add Compute Device
#
# Uncomment this section to add a compute device that processes
# COMPUTE requests from Linux applications
#
"""
compute_device = sst.Component("compute_device_0", "acalsim.ACALSimComputeDeviceComponent")
compute_device.addParams({
    "device_id": DEVICE_ID,
    "compute_latency": "1000",  # Cycles per compute unit
    "verbose": "1"
})

# Connect devices via link
link = sst.Link("device_link_0")
link.connect(
    (sst_device, "compute_link", "1ns"),
    (compute_device, "device_link", "1ns")
)
"""

#
# Statistics and Configuration
#
sst.setStatisticLoadLevel(5)
sst.setStatisticOutput("sst.statOutputConsole")
sst.enableAllStatisticsForComponentType("acalsim.ACALSimDeviceComponent")

# Simulation end time (10 seconds)
# Note: QEMU will control actual simulation duration
# sst.setProgramOption("stopAtCycle", "10s")

print("SST configuration complete")
print("Waiting for QEMU to connect...")
print("")
print("Start QEMU with:")
print(f"  -device virtio-sst-device,socket={SOCKET_PATH}")
print("")
