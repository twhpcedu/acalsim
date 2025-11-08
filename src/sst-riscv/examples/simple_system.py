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
SST Configuration for ACALSim Simple System Example

This example demonstrates:
- Creating ACALSim components in SST
- Connecting multiple components via SST Links
- Configuring component parameters
- Running a simple processor-memory system simulation
"""

import sst

# Simulation parameters
CLOCK_FREQ = "2GHz"
MAX_INSTRUCTIONS = 1000
MEMORY_LATENCY = "50ns"

# ========== Component Creation ==========

# Create Simple Processor (ACALSim component)
processor = sst.Component("processor", "acalsim.SimpleProcessor")
processor.addParams({
    "clock": CLOCK_FREQ,
    "max_instructions": MAX_INSTRUCTIONS,
    "verbose": 2,
    "name": "simple_processor_0"
})

# Create Simple Memory Controller (ACALSim component)
# Note: This would be another ACALSim component you implement
memory = sst.Component("memory", "acalsim.SimpleMemory")
memory.addParams({
    "clock": CLOCK_FREQ,
    "latency": MEMORY_LATENCY,
    "size": "1GiB",
    "verbose": 2,
    "name": "simple_memory_0"
})

# ========== Link Connections ==========

# Create link between processor and memory
mem_link = sst.Link("processor_memory_link")
mem_link.connect((processor, "mem_port", MEMORY_LATENCY), (memory, "cpu_port", MEMORY_LATENCY))

# ========== Statistics Configuration ==========

# Enable statistics collection
sst.setStatisticLoadLevel(5)
sst.setStatisticOutput("sst.statOutputConsole")

# Enable statistics for processor
processor.enableAllStatistics({"type": "sst.AccumulatorStatistic", "rate": "1us"})

# Enable statistics for memory
memory.enableAllStatistics({"type": "sst.AccumulatorStatistic", "rate": "1us"})

# ========== Simulation Configuration ==========

# Set simulation end time (optional - components control end)
# sst.setProgramOption("stop-at", "1ms")

print("SST Configuration Complete:")
print(f"  - Processor clock: {CLOCK_FREQ}")
print(f"  - Max instructions: {MAX_INSTRUCTIONS}")
print(f"  - Memory latency: {MEMORY_LATENCY}")
print(f"  - Link: processor <-> memory")
