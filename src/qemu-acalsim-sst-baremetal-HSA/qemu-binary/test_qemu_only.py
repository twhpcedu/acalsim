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
