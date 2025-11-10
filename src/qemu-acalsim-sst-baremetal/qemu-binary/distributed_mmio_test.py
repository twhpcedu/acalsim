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
Distributed Simulation Example - QEMU with MMIO Device Across Servers

This configuration demonstrates running QEMU and ACALSim devices on different
physical servers using SST's MPI-based distributed simulation.

Architecture:
  Server 1 (Rank 0):  QEMUBinary Component
  Server 2 (Rank 1):  ACALSimMMIODevice Component

Communication:
  - QEMU Process <--Unix Socket--> QEMUBinary (must be co-located on Rank 0)
  - QEMUBinary <--SST Links (MPI)--> ACALSimMMIODevice (can be on different ranks/servers)

Usage:
  # Single server (for testing):
  mpirun -np 2 sst distributed_mmio_test.py

  # Two servers (production):
  mpirun -np 2 -host server1:1,server2:1 sst distributed_mmio_test.py

  # Explicit host assignment:
  mpirun -H server1 -np 1 sst distributed_mmio_test.py : \
         -H server2 -np 1 sst distributed_mmio_test.py
"""

import sst
import os

# Get MPI rank information
rank = sst.getMPIRankCount()[0]  # (my_rank, total_ranks)
my_rank = rank
num_ranks = sst.getMPIRankCount()[1]

print(f"[Rank {my_rank}/{num_ranks}] Configuring distributed simulation...")

# Configuration parameters
base_dir = os.path.dirname(os.path.abspath(__file__))
qemu_binary = os.path.join(base_dir, "../../../qemu-build/qemu/build/qemu-system-riscv64")
kernel_image = os.path.join(base_dir, "../tests/mmio_interrupt_test")
bios_image = os.path.join(base_dir, "../../../qemu-build/opensbi/build/platform/generic/firmware/fw_jump.bin")

# Device configuration
mmio_device_base = 0x10001000
mmio_device_size = 4096
mmio_irq_num = 1

#
# Rank 0: QEMU Component (with co-located QEMU process via Unix socket)
#
if my_rank == 0:
    print(f"[Rank {my_rank}] Creating QEMUBinary component on this rank")

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
            "-device", f"acalsim-mmio,addr={hex(mmio_device_base)},size={mmio_device_size},irq={mmio_irq_num}"
        ],
        "socket_path": "/tmp/qemu_sst_distributed.sock",
        "verbose": "2"
    })

    # Set component to rank 0 explicitly
    qemu.setRank(0)

    print(f"[Rank {my_rank}] QEMUBinary configured with socket /tmp/qemu_sst_distributed.sock")

else:
    # Other ranks still need to declare the component for link connections
    # but SST will only instantiate it on rank 0
    qemu = sst.Component("qemu0", "acalsim.QEMUBinary")
    qemu.setRank(0)  # Ensure it's only on rank 0

#
# Rank 1: MMIO Device Component (on separate server/rank)
#
if my_rank == 1:
    print(f"[Rank {my_rank}] Creating ACALSimMMIODevice on this rank")

    mmio_device = sst.Component("mmio_device0", "acalsim.MMIODevice")
    mmio_device.addParams({
        "clock": "1GHz",
        "base_addr": str(mmio_device_base),
        "size": str(mmio_device_size),
        "irq_num": str(mmio_irq_num),
        "verbose": "2",
        "default_latency": "100"  # 100 cycles for operations
    })

    # Set component to rank 1 explicitly
    mmio_device.setRank(1)

    print(f"[Rank {my_rank}] ACALSimMMIODevice configured at base 0x{mmio_device_base:x}")

else:
    # Other ranks need to declare for link connections
    mmio_device = sst.Component("mmio_device0", "acalsim.MMIODevice")
    mmio_device.setRank(1)  # Ensure it's only on rank 1

#
# SST Links: Cross-rank communication via MPI
#
print(f"[Rank {my_rank}] Configuring SST links (cpu_port, irq_port)")

# CPU port: QEMU (rank 0) <--> MMIO Device (rank 1) for MMIO transactions
cpu_link = sst.Link("qemu_mmio_cpu_link")
cpu_link.connect(
    (qemu, "device_port_0", "1ns"),           # QEMU's device port 0
    (mmio_device, "cpu_port", "1ns")          # Device's CPU port
)

# IRQ port: MMIO Device (rank 1) --> QEMU (rank 0) for interrupts
irq_link = sst.Link("qemu_mmio_irq_link")
irq_link.connect(
    (mmio_device, "irq_port", "1ns"),         # Device's IRQ port
    (qemu, "irq_port_0", "1ns")               # QEMU's IRQ port 0
)

print(f"[Rank {my_rank}] Links configured: cpu_link (MMIO) and irq_link (interrupts)")

#
# Simulation parameters
#
sst.setProgramOption("stop-at", "10ms")  # Run for 10ms simulated time

print(f"[Rank {my_rank}] Configuration complete")
print(f"[Rank {my_rank}] ================================================")

if my_rank == 0:
    print(f"[Rank 0] Components on this rank: QEMUBinary")
    print(f"[Rank 0] QEMU process will connect via Unix socket (co-located)")
    print(f"[Rank 0] SST links will use MPI for cross-rank communication")
elif my_rank == 1:
    print(f"[Rank 1] Components on this rank: ACALSimMMIODevice")
    print(f"[Rank 1] Device will receive MMIO transactions via MPI from rank 0")
    print(f"[Rank 1] Device will send interrupts via MPI to rank 0")

print(f"[Rank {my_rank}] ================================================")
