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
HSA Multi-Accelerator SST Configuration

Creates a simulation with:
- 1 VirtIO device (connects to QEMU Linux)
- 1 HSA host agent (manages accelerators)
- 4 HSA compute agents (AI accelerators)

Copyright 2023-2025 Playlab/ACAL
"""

import sst
import os

# Configuration
NUM_ACCELERATORS = 4
CORES_PER_ACCELERATOR = 64
ACCELERATOR_CLOCK = "2GHz"
SOCKET_PATH = "/tmp/qemu-sst-linux.sock"

print("=" * 60)
print("HSA Multi-Accelerator Configuration")
print("=" * 60)
print(f"Accelerators: {NUM_ACCELERATORS}")
print(f"Cores per accelerator: {CORES_PER_ACCELERATOR}")
print(f"Clock frequency: {ACCELERATOR_CLOCK}")
print(
    f"Total compute power: {NUM_ACCELERATORS * CORES_PER_ACCELERATOR} cores @ {ACCELERATOR_CLOCK}"
)
print("=" * 60)

# Clean up existing socket
if os.path.exists(SOCKET_PATH):
	os.remove(SOCKET_PATH)
	print(f"Removed existing socket: {SOCKET_PATH}")

print("\nCreating components...")

# VirtIO Device (connects to QEMU)
print("VirtIO Device: virtio_dev")
virtio_dev = sst.Component("virtio_dev", "acalsim.VirtIODevice")
virtio_dev.addParams({
    "socket_path": SOCKET_PATH,
    "device_id": "0",
    "verbose": "1",
    "clock": "1GHz"
})

# HSA Host Agent (manages compute devices)
print("HSA Host Agent: hsa_host")
hsa_host = sst.Component("hsa_host", "acalsim.HSAHost")
hsa_host.addParams({"num_agents": str(NUM_ACCELERATORS), "verbose": "1", "clock": "1GHz"})

# Connect VirtIO Device → HSA Host
link_virt_hsa = sst.Link("link_virtio_hsa")
link_virt_hsa.connect((virtio_dev, "hsa_link", "10ns"), (hsa_host, "host_link", "10ns"))

# Create HSA Compute Agents (accelerators)
compute_agents = []
for i in range(NUM_ACCELERATORS):
	print(
	    f"HSA Compute Agent {i}: hsa_compute_{i} ({CORES_PER_ACCELERATOR} cores, {ACCELERATOR_CLOCK})"
	)

	compute = sst.Component(f"hsa_compute_{i}", "acalsim.HSACompute")
	compute.addParams({
	    "device_id": str(i),
	    "cores": str(CORES_PER_ACCELERATOR),
	    "clock": ACCELERATOR_CLOCK,
	    "memory_size": "16GB",
	    "verbose": "1"
	})
	compute_agents.append(compute)

	# Connect Host → Compute
	link = sst.Link(f"link_host_compute_{i}")
	link.connect((hsa_host, f"compute_link_{i}", "10ns"), (compute, "host_link", "10ns"))

print("\nConnecting components via SST links...")

# Statistics
sst.setStatisticLoadLevel(5)
sst.setStatisticOutput("sst.statOutputConsole")

print("Configuration complete!")
print("Waiting for QEMU to connect...")
print("")
print("Start QEMU with:")
print(f"  -device virtio-sst-device,socket={SOCKET_PATH}")
print("")
