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
Distributed Multi-Device Simulation - QEMU with N Devices Across Servers

This configuration demonstrates running QEMU and multiple ACALSim devices
distributed across multiple physical servers using SST's MPI-based distributed
simulation.

Architecture (4 ranks across 4 servers):
  Server 1 (Rank 0):  QEMUBinary Component
  Server 2 (Rank 1):  Echo Device + Compute Device (2 devices on same rank)
  Server 3 (Rank 2):  MMIO Device
  Server 4 (Rank 3):  Second Echo Device

This demonstrates:
- Single server with QEMU (rank 0)
- Multiple devices on single rank (rank 1) - for tightly coupled devices
- Single device per rank (ranks 2, 3) - for load balancing

Communication:
  - QEMU Process <--Unix Socket--> QEMUBinary (co-located on rank 0)
  - QEMUBinary <--SST Links (MPI)--> All Devices (distributed across ranks 1-3)

Usage:
  # Four servers (production deployment):
  mpirun -np 4 -host server1:1,server2:1,server3:1,server4:1 \\
    sst distributed_multi_device_test.py

  # Single server testing (4 processes on localhost):
  mpirun -np 4 sst distributed_multi_device_test.py

  # Two servers (group devices on second server):
  mpirun -np 2 -host server1:1,server2:1 sst distributed_multi_device_test.py
  # Note: Modify rank assignments for 2-rank deployment
"""

import sst
import os

# Get MPI rank information
my_rank, num_ranks = sst.getMPIRankCount()

print(f"[Rank {my_rank}/{num_ranks}] Configuring distributed multi-device simulation...")

# Configuration parameters
base_dir = os.path.dirname(os.path.abspath(__file__))
qemu_binary = os.path.join(base_dir, "../../../qemu-build/qemu/build/qemu-system-riscv64")
kernel_image = os.path.join(base_dir, "../tests/multi_device_test")
bios_image = os.path.join(
    base_dir, "../../../qemu-build/opensbi/build/platform/generic/firmware/fw_jump.bin"
)

# Device base addresses (matching QEMU device tree)
echo_device1_base = 0x10200000
compute_device_base = 0x10300000
mmio_device_base = 0x10001000
echo_device2_base = 0x10400000

device_size = 4096

#
# Rank 0: QEMU Component (with co-located QEMU process)
#
if my_rank == 0:
	print(f"[Rank {my_rank}] Creating QEMUBinary component")

	qemu = sst.Component("qemu0", "acalsim.QEMUBinary")
	qemu.addParams({
	    "qemu_binary": qemu_binary,
	    "qemu_args": [
	        "-M",
	        "virt",
	        "-cpu",
	        "rv64",
	        "-m",
	        "128M",
	        "-nographic",
	        "-bios",
	        bios_image,
	        "-kernel",
	        kernel_image,
	        # Register all 4 devices
	        "-device",
	        f"acalsim-echo,addr={hex(echo_device1_base)},size={device_size}",
	        "-device",
	        f"acalsim-compute,addr={hex(compute_device_base)},size={device_size}",
	        "-device",
	        f"acalsim-mmio,addr={hex(mmio_device_base)},size={device_size},irq=1",
	        "-device",
	        f"acalsim-echo,addr={hex(echo_device2_base)},size={device_size}"
	    ],
	    "socket_path": "/tmp/qemu_sst_multi_distributed.sock",
	    "verbose": "2"
	})
	qemu.setRank(0)

	print(f"[Rank {my_rank}] QEMUBinary configured with 4 devices")
else:
	# Other ranks declare for link connections
	qemu = sst.Component("qemu0", "acalsim.QEMUBinary")
	qemu.setRank(0)

#
# Rank 1: Echo Device 1 + Compute Device (tightly coupled devices)
#
if my_rank == 1:
	print(f"[Rank {my_rank}] Creating Echo Device 1 + Compute Device (2 devices on this rank)")

	# Echo Device 1
	echo_dev1 = sst.Component("echo_device1", "acalsim.QEMUDevice")
	echo_dev1.addParams({
	    "clock": "1GHz",
	    "base_addr": str(echo_device1_base),
	    "size": str(device_size),
	    "verbose": "2",
	    "echo_latency": "10"
	})
	echo_dev1.setRank(1)

	# Compute Device (on same rank for efficient local communication)
	compute_dev = sst.Component("compute_device", "acalsim.ComputeDevice")
	compute_dev.addParams({
	    "clock": "1GHz",
	    "base_addr": str(compute_device_base),
	    "size": str(device_size),
	    "verbose": "2",
	    "compute_latency": "50"
	})
	compute_dev.setRank(1)

	print(f"[Rank {my_rank}] Both devices configured (efficient local coupling)")
else:
	# Other ranks declare for link connections
	echo_dev1 = sst.Component("echo_device1", "acalsim.QEMUDevice")
	echo_dev1.setRank(1)
	compute_dev = sst.Component("compute_device", "acalsim.ComputeDevice")
	compute_dev.setRank(1)

#
# Rank 2: MMIO Device (single device for isolation)
#
if my_rank == 2:
	print(f"[Rank {my_rank}] Creating MMIO Device")

	mmio_dev = sst.Component("mmio_device", "acalsim.MMIODevice")
	mmio_dev.addParams({
	    "clock": "1GHz",
	    "base_addr": str(mmio_device_base),
	    "size": str(device_size),
	    "irq_num": "1",
	    "verbose": "2",
	    "default_latency": "100"
	})
	mmio_dev.setRank(2)

	print(f"[Rank {my_rank}] MMIO Device configured")
else:
	mmio_dev = sst.Component("mmio_device", "acalsim.MMIODevice")
	mmio_dev.setRank(2)

#
# Rank 3: Echo Device 2 (single device for load balancing)
#
if my_rank == 3:
	print(f"[Rank {my_rank}] Creating Echo Device 2")

	echo_dev2 = sst.Component("echo_device2", "acalsim.QEMUDevice")
	echo_dev2.addParams({
	    "clock": "1GHz",
	    "base_addr": str(echo_device2_base),
	    "size": str(device_size),
	    "verbose": "2",
	    "echo_latency": "10"
	})
	echo_dev2.setRank(3)

	print(f"[Rank {my_rank}] Echo Device 2 configured")
else:
	echo_dev2 = sst.Component("echo_device2", "acalsim.QEMUDevice")
	echo_dev2.setRank(3)

#
# SST Links: QEMUBinary (rank 0) <-> All Devices (ranks 1-3)
#
print(f"[Rank {my_rank}] Configuring SST links for all devices")

# Echo Device 1 links (rank 0 <-> rank 1)
link_echo1 = sst.Link("qemu_echo1_link")
link_echo1.connect((qemu, "device_port_0", "1ns"), (echo_dev1, "cpu_port", "1ns"))

# Compute Device links (rank 0 <-> rank 1)
link_compute = sst.Link("qemu_compute_link")
link_compute.connect((qemu, "device_port_1", "1ns"), (compute_dev, "cpu_port", "1ns"))

# MMIO Device links (rank 0 <-> rank 2)
link_mmio_cpu = sst.Link("qemu_mmio_cpu_link")
link_mmio_cpu.connect((qemu, "device_port_2", "1ns"), (mmio_dev, "cpu_port", "1ns"))

link_mmio_irq = sst.Link("qemu_mmio_irq_link")
link_mmio_irq.connect((mmio_dev, "irq_port", "1ns"), (qemu, "irq_port_0", "1ns"))

# Echo Device 2 links (rank 0 <-> rank 3)
link_echo2 = sst.Link("qemu_echo2_link")
link_echo2.connect((qemu, "device_port_3", "1ns"), (echo_dev2, "cpu_port", "1ns"))

print(f"[Rank {my_rank}] All links configured")

#
# Optional: Peer links for inter-device communication (within same rank)
#
if my_rank == 1:
	# Echo Device 1 and Compute Device on same rank can communicate efficiently
	peer_link = sst.Link("echo_compute_peer_link")
	peer_link.connect((echo_dev1, "peer_port", "1ns"), (compute_dev, "peer_port", "1ns"))
	print(
	    f"[Rank {my_rank}] Peer link configured between Echo1 and Compute (efficient local communication)"
	)

#
# Simulation parameters
#
sst.setProgramOption("stop-at", "10ms")

print(f"[Rank {my_rank}] Configuration complete")
print(f"[Rank {my_rank}] ================================================")

# Print deployment summary
if my_rank == 0:
	print(f"[Rank 0] Components on this rank: QEMUBinary")
	print(f"[Rank 0]   - QEMU process co-located (Unix socket)")
	print(f"[Rank 0]   - 4 SST links to devices on ranks 1-3 (MPI)")
elif my_rank == 1:
	print(f"[Rank 1] Components on this rank: Echo Device 1, Compute Device")
	print(f"[Rank 1]   - 2 devices co-located for tight coupling")
	print(f"[Rank 1]   - Peer link for efficient inter-device communication")
elif my_rank == 2:
	print(f"[Rank 2] Components on this rank: MMIO Device")
	print(f"[Rank 2]   - Single device (isolated deployment)")
	print(f"[Rank 2]   - Supports interrupts to QEMU via MPI")
elif my_rank == 3:
	print(f"[Rank 3] Components on this rank: Echo Device 2")
	print(f"[Rank 3]   - Single device (load balancing)")

print(f"[Rank {my_rank}] ================================================")

# Print usage examples
if my_rank == 0:
	print(f"\n[Deployment Options]")
	print(f"1. Four servers (recommended):")
	print(f"   mpirun -np 4 -host server1:1,server2:1,server3:1,server4:1 sst {__file__}")
	print(f"\n2. Two servers (devices grouped):")
	print(f"   mpirun -np 4 -host server1:2,server2:2 sst {__file__}")
	print(f"\n3. Single server testing:")
	print(f"   mpirun -np 4 sst {__file__}")
	print(f"")
