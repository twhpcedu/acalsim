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
Distributed HSA Simulation - QEMU with HSA Host and Compute Agents

This configuration demonstrates running QEMU and HSA components distributed
across multiple physical servers using SST's MPI-based distributed simulation.

Architecture (3 ranks across 3 servers):
  Server 1 (Rank 0):  QEMUBinary Component + HSA Host Agent
  Server 2 (Rank 1):  HSA Compute Agent 1
  Server 3 (Rank 2):  HSA Compute Agent 2

This demonstrates HSA heterogeneous compute across distributed servers.

Usage:
  # Three servers (production):
  mpirun -np 3 -host server1:1,server2:1,server3:1 sst distributed_hsa_test.py

  # Single server (testing):
  mpirun -np 3 sst distributed_hsa_test.py
"""

import sst
import os

# Get MPI rank information
my_rank, num_ranks = sst.getMPIRankCount()

print(f"[Rank {my_rank}/{num_ranks}] Configuring distributed HSA simulation...")

# Configuration parameters
base_dir = os.path.dirname(os.path.abspath(__file__))
qemu_binary = os.path.join(base_dir, "../../../qemu-build/qemu/build/qemu-system-riscv64")
kernel_image = os.path.join(base_dir, "../tests/hsa_test")
bios_image = os.path.join(base_dir, "../../../qemu-build/opensbi/build/platform/generic/firmware/fw_jump.bin")

# HSA configuration
hsa_host_base = 0x10500000
hsa_compute1_base = 0x10600000
hsa_compute2_base = 0x10700000
device_size = 4096

#
# Rank 0: QEMU Component + HSA Host Agent (tightly coupled)
#
if my_rank == 0:
    print(f"[Rank {my_rank}] Creating QEMUBinary + HSA Host Agent")

    qemu = sst.Component("qemu0", "acalsim.QEMUBinary")
    qemu.addParams({
        "qemu_binary": qemu_binary,
        "qemu_args": [
            "-M", "virt",
            "-cpu", "rv64",
            "-m", "128M",
            "-nographic",
            "-bios", bios_image,
            "-kernel", kernel_image,
            "-device", f"acalsim-hsa-host,addr={hex(hsa_host_base)},size={device_size}",
            "-device", f"acalsim-hsa-compute,addr={hex(hsa_compute1_base)},size={device_size}",
            "-device", f"acalsim-hsa-compute,addr={hex(hsa_compute2_base)},size={device_size}"
        ],
        "socket_path": "/tmp/qemu_sst_hsa_distributed.sock",
        "verbose": "2"
    })
    qemu.setRank(0)

    # HSA Host Agent (on same rank as QEMU for efficient packet dispatch)
    hsa_host = sst.Component("hsa_host", "acalsim.HSAHost")
    hsa_host.addParams({
        "clock": "1GHz",
        "base_addr": str(hsa_host_base),
        "size": str(device_size),
        "verbose": "2",
        "max_queues": "8",
        "packet_processor_latency": "10"
    })
    hsa_host.setRank(0)

    print(f"[Rank {my_rank}] QEMU + HSA Host configured (co-located)")
else:
    # Other ranks declare for link connections
    qemu = sst.Component("qemu0", "acalsim.QEMUBinary")
    qemu.setRank(0)
    hsa_host = sst.Component("hsa_host", "acalsim.HSAHost")
    hsa_host.setRank(0)

#
# Rank 1: HSA Compute Agent 1 (distributed)
#
if my_rank == 1:
    print(f"[Rank {my_rank}] Creating HSA Compute Agent 1")

    hsa_compute1 = sst.Component("hsa_compute1", "acalsim.HSACompute")
    hsa_compute1.addParams({
        "clock": "2GHz",  # Faster compute clock
        "base_addr": str(hsa_compute1_base),
        "size": str(device_size),
        "verbose": "2",
        "compute_units": "16",
        "wavefront_size": "64",
        "kernel_latency": "100"
    })
    hsa_compute1.setRank(1)

    print(f"[Rank {my_rank}] HSA Compute Agent 1 configured (16 CUs)")
else:
    hsa_compute1 = sst.Component("hsa_compute1", "acalsim.HSACompute")
    hsa_compute1.setRank(1)

#
# Rank 2: HSA Compute Agent 2 (distributed)
#
if my_rank == 2:
    print(f"[Rank {my_rank}] Creating HSA Compute Agent 2")

    hsa_compute2 = sst.Component("hsa_compute2", "acalsim.HSACompute")
    hsa_compute2.addParams({
        "clock": "2GHz",  # Faster compute clock
        "base_addr": str(hsa_compute2_base),
        "size": str(device_size),
        "verbose": "2",
        "compute_units": "16",
        "wavefront_size": "64",
        "kernel_latency": "100"
    })
    hsa_compute2.setRank(2)

    print(f"[Rank {my_rank}] HSA Compute Agent 2 configured (16 CUs)")
else:
    hsa_compute2 = sst.Component("hsa_compute2", "acalsim.HSACompute")
    hsa_compute2.setRank(2)

#
# SST Links: QEMU <-> All HSA Components
#
print(f"[Rank {my_rank}] Configuring SST links for HSA components")

# HSA Host links (rank 0 <-> rank 0, but via SST links for consistency)
link_hsa_host = sst.Link("qemu_hsa_host_link")
link_hsa_host.connect(
    (qemu, "device_port_0", "1ns"),
    (hsa_host, "cpu_port", "1ns")
)

# HSA Compute 1 links (rank 0 <-> rank 1)
link_hsa_compute1 = sst.Link("qemu_hsa_compute1_link")
link_hsa_compute1.connect(
    (qemu, "device_port_1", "10ns"),  # Higher latency for cross-server
    (hsa_compute1, "cpu_port", "10ns")
)

# HSA Compute 2 links (rank 0 <-> rank 2)
link_hsa_compute2 = sst.Link("qemu_hsa_compute2_link")
link_hsa_compute2.connect(
    (qemu, "device_port_2", "10ns"),  # Higher latency for cross-server
    (hsa_compute2, "cpu_port", "10ns")
)

# HSA packet dispatch links (Host -> Compute Agents)
link_dispatch1 = sst.Link("hsa_dispatch1_link")
link_dispatch1.connect(
    (hsa_host, "dispatch_port_0", "10ns"),
    (hsa_compute1, "dispatch_port", "10ns")
)

link_dispatch2 = sst.Link("hsa_dispatch2_link")
link_dispatch2.connect(
    (hsa_host, "dispatch_port_1", "10ns"),
    (hsa_compute2, "dispatch_port", "10ns")
)

# HSA completion links (Compute Agents -> Host)
link_completion1 = sst.Link("hsa_completion1_link")
link_completion1.connect(
    (hsa_compute1, "completion_port", "10ns"),
    (hsa_host, "completion_port_0", "10ns")
)

link_completion2 = sst.Link("hsa_completion2_link")
link_completion2.connect(
    (hsa_compute2, "completion_port", "10ns"),
    (hsa_host, "completion_port_1", "10ns")
)

print(f"[Rank {my_rank}] All links configured")

#
# Simulation parameters
#
sst.setProgramOption("stop-at", "100ms")  # Longer time for compute workloads

print(f"[Rank {my_rank}] Configuration complete")
print(f"[Rank {my_rank}] ================================================")

# Print deployment summary
if my_rank == 0:
    print(f"[Rank 0] Components on this rank:")
    print(f"[Rank 0]   - QEMUBinary (QEMU process co-located)")
    print(f"[Rank 0]   - HSA Host Agent (packet dispatch)")
    print(f"[Rank 0]   - Links to 2 compute agents via MPI")
elif my_rank == 1:
    print(f"[Rank 1] Components on this rank:")
    print(f"[Rank 1]   - HSA Compute Agent 1 (16 CUs, 2GHz)")
    print(f"[Rank 1]   - Receives AQL packets from Host via MPI")
elif my_rank == 2:
    print(f"[Rank 2] Components on this rank:")
    print(f"[Rank 2]   - HSA Compute Agent 2 (16 CUs, 2GHz)")
    print(f"[Rank 2]   - Receives AQL packets from Host via MPI")

print(f"[Rank {my_rank}] ================================================")

# Print usage examples
if my_rank == 0:
    print(f"\n[Deployment Options]")
    print(f"1. Three servers (recommended):")
    print(f"   mpirun -np 3 -host server1:1,server2:1,server3:1 sst {__file__}")
    print(f"\n2. Single server testing:")
    print(f"   mpirun -np 3 sst {__file__}")
    print(f"")
