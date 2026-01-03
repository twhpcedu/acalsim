#!/usr/bin/env python3
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
"""
SST Configuration for ACALSim Multi-Core System with NoC

This example demonstrates:
- Multi-core processor system with ACALSim components
- Network-on-Chip (NoC) interconnect
- Hierarchical memory system (L1, L2 caches, shared memory)
- Cache coherence protocol integration
- Performance analysis and statistics collection

System Architecture:
    [CPU0]-[L1$]--+
    [CPU1]-[L1$]--+--[NoC]--[L2$]--[Memory]
    [CPU2]-[L1$]--+
    [CPU3]-[L1$]--+
"""

import sst

# ========== Configuration Parameters ==========

# System configuration
NUM_CORES = 4
CLOCK_FREQ = "2GHz"
MAX_INSTRUCTIONS = 10000

# Cache configuration
L1_SIZE = "32KiB"
L1_ASSOC = 4
L1_LATENCY = "1ns"

L2_SIZE = "1MiB"
L2_ASSOC = 8
L2_LATENCY = "10ns"

# Memory configuration
MEM_SIZE = "4GiB"
MEM_LATENCY = "100ns"

# NoC configuration
NOC_TOPOLOGY = "mesh"
NOC_ROUTING = "xy"
NOC_LINK_LATENCY = "5ns"

# ========== Helper Functions ==========


def create_processor(cpu_id):
	"""Create a processor core with L1 cache"""
	# Create processor
	cpu = sst.Component(f"cpu{cpu_id}", "acalsim.SimpleProcessor")
	cpu.addParams({
	    "clock": CLOCK_FREQ,
	    "max_instructions": MAX_INSTRUCTIONS,
	    "verbose": 1,
	    "name": f"cpu_{cpu_id}",
	    "cpu_id": cpu_id
	})

	# Create L1 cache
	l1_cache = sst.Component(f"l1_cache{cpu_id}", "acalsim.SimpleCache")
	l1_cache.addParams({
	    "clock": CLOCK_FREQ,
	    "cache_size": L1_SIZE,
	    "associativity": L1_ASSOC,
	    "cache_line_size": 64,
	    "latency": L1_LATENCY,
	    "verbose": 1,
	    "name": f"l1_cache_{cpu_id}"
	})

	# Connect CPU to L1 cache
	cpu_l1_link = sst.Link(f"cpu{cpu_id}_l1_link")
	cpu_l1_link.connect((cpu, "mem_port", L1_LATENCY), (l1_cache, "cpu_port", L1_LATENCY))

	return cpu, l1_cache


def create_noc_router(router_id, x, y):
	"""Create a NoC router"""
	router = sst.Component(f"router{router_id}", "acalsim.NoCRouter")
	router.addParams({
	    "clock": CLOCK_FREQ,
	    "router_id": router_id,
	    "x_pos": x,
	    "y_pos": y,
	    "routing_algorithm": NOC_ROUTING,
	    "buffer_depth": 4,
	    "verbose": 1,
	    "name": f"router_{router_id}"
	})
	return router


# ========== Component Creation ==========

print(f"Creating {NUM_CORES}-core system with NoC interconnect...")

# Create processors and L1 caches
cpus = []
l1_caches = []
for i in range(NUM_CORES):
	cpu, l1 = create_processor(i)
	cpus.append(cpu)
	l1_caches.append(l1)

# Create NoC routers (2x2 mesh for 4 cores)
routers = []
for i in range(NUM_CORES):
	x = i % 2
	y = i // 2
	router = create_noc_router(i, x, y)
	routers.append(router)

# Connect L1 caches to NoC routers
for i in range(NUM_CORES):
	l1_router_link = sst.Link(f"l1_{i}_router_{i}_link")
	l1_router_link.connect((l1_caches[i], "noc_port", NOC_LINK_LATENCY),
	                       (routers[i], "local_port", NOC_LINK_LATENCY))

# Connect NoC routers in mesh topology (2x2)
# Horizontal connections
for y in range(2):
	for x in range(1):
		router_id = y * 2 + x
		router_link = sst.Link(f"router_{router_id}_router_{router_id+1}_link")
		router_link.connect((routers[router_id], "east_port", NOC_LINK_LATENCY),
		                    (routers[router_id + 1], "west_port", NOC_LINK_LATENCY))

# Vertical connections
for x in range(2):
	for y in range(1):
		router_id = y * 2 + x
		router_link = sst.Link(f"router_{router_id}_router_{router_id+2}_link")
		router_link.connect((routers[router_id], "south_port", NOC_LINK_LATENCY),
		                    (routers[router_id + 2], "north_port", NOC_LINK_LATENCY))

# Create L2 cache
l2_cache = sst.Component("l2_cache", "acalsim.SimpleCache")
l2_cache.addParams({
    "clock": CLOCK_FREQ,
    "cache_size": L2_SIZE,
    "associativity": L2_ASSOC,
    "cache_line_size": 64,
    "latency": L2_LATENCY,
    "verbose": 1,
    "name": "l2_cache_shared"
})

# Connect L2 cache to NoC (via router 0)
l2_noc_link = sst.Link("l2_noc_link")
l2_noc_link.connect((routers[0], "l2_port", NOC_LINK_LATENCY),
                    (l2_cache, "noc_port", NOC_LINK_LATENCY))

# Create main memory
memory = sst.Component("memory", "acalsim.SimpleMemory")
memory.addParams({
    "clock": CLOCK_FREQ,
    "size": MEM_SIZE,
    "latency": MEM_LATENCY,
    "verbose": 1,
    "name": "main_memory"
})

# Connect L2 cache to memory
l2_mem_link = sst.Link("l2_memory_link")
l2_mem_link.connect((l2_cache, "mem_port", MEM_LATENCY), (memory, "cache_port", MEM_LATENCY))

# ========== Statistics Configuration ==========

print("Configuring statistics collection...")

# Enable statistics
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputCSV", {"filepath": "multicore_stats.csv", "separator": ","})

# CPU statistics
for i, cpu in enumerate(cpus):
	cpu.enableAllStatistics({"type": "sst.AccumulatorStatistic", "rate": "10us"})

# Cache statistics
for i, l1 in enumerate(l1_caches):
	l1.enableAllStatistics({"type": "sst.AccumulatorStatistic", "rate": "10us"})

l2_cache.enableAllStatistics({"type": "sst.AccumulatorStatistic", "rate": "10us"})

# NoC statistics
for i, router in enumerate(routers):
	router.enableAllStatistics({"type": "sst.AccumulatorStatistic", "rate": "10us"})

# Memory statistics
memory.enableAllStatistics({"type": "sst.AccumulatorStatistic", "rate": "10us"})

# ========== Simulation Configuration ==========

# Optional: Set simulation time limit
# sst.setProgramOption("stop-at", "10ms")

# Print configuration summary
print("\n" + "=" * 60)
print("Multi-Core System Configuration Summary")
print("=" * 60)
print(f"Number of cores:       {NUM_CORES}")
print(f"Clock frequency:       {CLOCK_FREQ}")
print(f"Max instructions:      {MAX_INSTRUCTIONS}")
print(f"\nCache Hierarchy:")
print(f"  L1 Cache:            {L1_SIZE}, {L1_ASSOC}-way, {L1_LATENCY} latency")
print(f"  L2 Cache (shared):   {L2_SIZE}, {L2_ASSOC}-way, {L2_LATENCY} latency")
print(f"\nMemory:")
print(f"  Size:                {MEM_SIZE}")
print(f"  Latency:             {MEM_LATENCY}")
print(f"\nNetwork-on-Chip:")
print(f"  Topology:            {NOC_TOPOLOGY} (2x2)")
print(f"  Routing:             {NOC_ROUTING}")
print(f"  Link latency:        {NOC_LINK_LATENCY}")
print(f"\nStatistics output:     multicore_stats.csv")
print("=" * 60)
