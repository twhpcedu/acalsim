#!/usr/bin/env python3
"""
Minimal QEMU test - just load the component
"""

import sst

# QEMU Binary Component only
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "1GHz",
    "verbose": 3,  # Max verbosity
    "binary_path": "/home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal/riscv-programs/multi_device_test.elf",
    "qemu_path": "qemu-system-riscv32",
    "num_devices": 1
})

# Short simulation
sst.setProgramOption("stop-at", "100ms")
