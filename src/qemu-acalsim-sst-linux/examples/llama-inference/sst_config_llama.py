#!/usr/bin/env python3
"""
LLAMA 2 Inference SST Configuration

Creates a simulation with:
- 1 VirtIO device (connects to QEMU Linux)
- 4 AI accelerators (Attention, FFN, Embedding, General)

Copyright 2023-2025 Playlab/ACAL
Licensed under the Apache License, Version 2.0
"""

import sst
import os

# Configuration
SOCKET_PATH = "/tmp/qemu-sst-llama.sock"

# Accelerator Parameters
# These latencies model different neural network operations
ACCEL_CONFIG = {
    "attention": {
        "name": "Attention Accelerator",
        "latency": "1us",      # Self-attention computation
        "description": "Multi-head attention operations"
    },
    "ffn": {
        "name": "FFN Accelerator",
        "latency": "500ns",    # Feed-forward network
        "description": "Dense layer computations"
    },
    "embedding": {
        "name": "Embedding Accelerator",
        "latency": "100ns",    # Token embedding lookup
        "description": "Token embedding table lookups"
    },
    "general": {
        "name": "General Compute",
        "latency": "200ns",    # Other operations
        "description": "Normalization, activation functions"
    }
}

print("=" * 60)
print("LLAMA 2 Inference SST Configuration")
print("=" * 60)
print(f"Socket: {SOCKET_PATH}")
print("")
print("Accelerators:")
for idx, (accel_type, config) in enumerate(ACCEL_CONFIG.items()):
    print(f"  [{idx}] {config['name']:20s} - {config['latency']:8s} - {config['description']}")
print("=" * 60)

# Clean up existing socket
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)
    print(f"Removed existing socket: {SOCKET_PATH}")

print("\nCreating components...")

# VirtIO Device (connects to QEMU Linux)
print("VirtIO Device: virtio_llama")
virtio_dev = sst.Component("virtio_llama", "acalsim.VirtIODevice")
virtio_dev.addParams({
    "socket_path": SOCKET_PATH,
    "device_id": "0",
    "verbose": "1",
    "clock": "2GHz"
})

# Create AI Accelerator Components
accelerators = []
accel_list = list(ACCEL_CONFIG.items())

for idx, (accel_type, config) in enumerate(accel_list):
    comp_name = f"accel_{accel_type}"
    print(f"Accelerator {idx}: {comp_name} ({config['name']}, latency={config['latency']})")

    accel = sst.Component(comp_name, "acalsim.ComputeDevice")
    accel.addParams({
        "clock": "2GHz",
        "verbose": "1",
        "compute_latency": config["latency"],
        "device_id": str(idx)
    })
    accelerators.append((comp_name, accel))

print("\nConnecting components via SST links...")

# Connect VirtIO to each accelerator
for idx, (comp_name, accel) in enumerate(accelerators):
    link_name = f"link_virtio_{comp_name}"
    link = sst.Link(link_name)
    link.connect(
        (virtio_dev, f"device_port_{idx}", "10ns"),
        (accel, "cpu_port", "10ns")
    )
    print(f"  {link_name}: virtio_llama <-> {comp_name}")

# Statistics
sst.setStatisticLoadLevel(5)
sst.setStatisticOutput("sst.statOutputConsole")

# Simulation parameters
sst.setProgramOption("stop-at", "1s")

print("\nConfiguration complete!")
print("")
print("=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Start this SST simulation (already running)")
print("2. In another terminal, start QEMU with:")
print(f"   ./run_qemu.sh")
print("")
print("3. In Linux, mount model disk:")
print("   mount /dev/vda /mnt/models")
print("")
print("4. Run inference:")
print("   cd /apps/llama-inference")
print("   ./llama_inference.py \"Your prompt here\"")
print("=" * 60)
print("")
print("Waiting for QEMU to connect...")
print("")
