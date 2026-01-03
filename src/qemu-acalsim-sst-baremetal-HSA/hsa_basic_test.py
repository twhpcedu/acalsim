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
HSA Protocol Basic Test

Tests basic HSA protocol communication between host and compute agents.
Demonstrates AQL packet submission, kernel execution, and signal completion.
"""

import sst

# Enable SST statistics
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputConsole")

# HSA Host Component (CPU Agent - Job Submitter)
host = sst.Component("hsa_host", "acalsim.HSAHost")
host.addParams({
    "clock": "2GHz",
    "verbose": 2,
    "num_dispatches": 5,
    "workgroup_size_x": 256,
    "workgroup_size_y": 1,
    "workgroup_size_z": 1,
    "grid_size_x": 1024,
    "grid_size_y": 1,
    "grid_size_z": 1,
    "dispatch_interval": 10000  # 10000 cycles between dispatches
})

# HSA Compute Component (GPU/Accelerator Agent - Job Executor)
compute = sst.Component("hsa_compute", "acalsim.HSACompute")
compute.addParams({
    "clock": "1GHz",
    "verbose": 2,
    "queue_depth": 16,
    "cycles_per_workitem": 50,
    "kernel_launch_overhead": 500,
    "memory_latency": 100
})

# Create links for HSA communication
# AQL queue link (host → compute: job descriptors)
aql_link = sst.Link("aql_queue_link")
aql_link.connect((host, "aql_port", "10ns"), (compute, "aql_port", "10ns"))

# Signal link (compute → host: completion notifications)
signal_link = sst.Link("signal_link")
signal_link.connect((host, "signal_port", "10ns"), (compute, "signal_port", "10ns"))

# Optional doorbell link (host → compute: queue notifications)
doorbell_link = sst.Link("doorbell_link")
doorbell_link.connect((host, "doorbell_port", "5ns"), (compute, "doorbell_port", "5ns"))

# Simulation parameters
sst.setProgramOption("stop-at", "1ms")

print("=" * 60)
print("HSA Protocol Basic Test")
print("=" * 60)
print(f"Configuration:")
print(f"  Host: 2GHz, {host.params['num_dispatches']} dispatches")
print(f"  Compute: 1GHz, {compute.params['cycles_per_workitem']} cycles/workitem")
print(
    f"  Workgroup: {host.params['workgroup_size_x']}x{host.params['workgroup_size_y']}x{host.params['workgroup_size_z']}"
)
print(
    f"  Grid: {host.params['grid_size_x']}x{host.params['grid_size_y']}x{host.params['grid_size_z']}"
)
print(
    f"  Total workitems per kernel: {int(host.params['grid_size_x']) * int(host.params['grid_size_y']) * int(host.params['grid_size_z'])}"
)
print("=" * 60)
