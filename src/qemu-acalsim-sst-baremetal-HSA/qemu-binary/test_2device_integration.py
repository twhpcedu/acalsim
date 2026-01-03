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

#!/usr/bin/env python3
"""
SST 2-Device Integration Test
Tests QEMU-SST integration with 2 devices using single socket
"""

import sst

# Enable SST statistics
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputConsole")

# QEMU Binary Component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "1GHz",
    "verbose": 2,
    "binary_path": "/home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal/riscv-programs/multi_device_test.elf",
    "qemu_path": "/home/user/qemu-build/qemu/build/qemu-system-riscv32",
    "socket_path": "/tmp/qemu-sst-mmio.sock",
    "num_devices": 2,
    "device0_base": "0x10200000",
    "device0_size": 4096,
    "device0_name": "echo_device",
    "device1_base": "0x10300000",
    "device1_size": 4096,
    "device1_name": "compute_device"
})

# Device 0: Echo Device
echo_device = sst.Component("echo_device", "acalsim.QEMUDevice")
echo_device.addParams({"clock": "1GHz", "verbose": 2, "echo_latency": 10})

# Device 1: Compute Device
compute_device = sst.Component("compute_device", "acalsim.ComputeDevice")
compute_device.addParams({"clock": "1GHz", "verbose": 2, "compute_latency": 100})

# Links between QEMU and devices
link0 = sst.Link("link_0")
link0.connect((qemu, "device_port_0", "1ns"), (echo_device, "cpu_port", "1ns"))

link1 = sst.Link("link_1")
link1.connect((qemu, "device_port_1", "1ns"), (compute_device, "cpu_port", "1ns"))

# Simulation parameters
sst.setProgramOption("stop-at", "10s")
