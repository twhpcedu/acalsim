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
LLAMA 2 Inference SST Configuration

Creates a simulation with:
- 1 VirtIO device (connects to QEMU Linux and processes compute requests)

The VirtIODevice handles compute requests from Linux applications
via /dev/sst0. The Python backend (llama_sst_backend.py) sends
compute requests with different latency models to simulate:
  - Attention operations (latency_model=0)
  - FFN operations (latency_model=1)
  - Embedding operations (latency_model=2)

Copyright 2023-2025 Playlab/ACAL
Licensed under the Apache License, Version 2.0
"""

import sst
import os

# Configuration
SOCKET_PATH = "/tmp/qemu-sst-llama.sock"
DEVICE_ID = 0

print("=" * 60)
print("LLAMA 2 Inference SST Configuration")
print("=" * 60)
print(f"Socket: {SOCKET_PATH}")
print(f"Device ID: {DEVICE_ID}")
print("")
print("The VirtIODevice will process compute requests from Linux.")
print("Different operations (attention, FFN, embedding) are modeled")
print("via the latency_model parameter in compute requests.")
print("=" * 60)

# Clean up existing socket
if os.path.exists(SOCKET_PATH):
	os.remove(SOCKET_PATH)
	print(f"Removed existing socket: {SOCKET_PATH}")

print("\nCreating components...")

# VirtIO Device (connects to QEMU Linux)
# This single device handles all compute requests from the LLAMA inference app
print("VirtIO Device: virtio_llama")
virtio_dev = sst.Component("virtio_llama", "acalsim.VirtIODevice")
virtio_dev.addParams({
    "socket_path": SOCKET_PATH,
    "device_id": str(DEVICE_ID),
    "verbose": "2",  # Increased verbosity to see request processing
    "clock": "2GHz"
})

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
