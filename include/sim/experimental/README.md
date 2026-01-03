<!--
Copyright 2023-2026 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Experimental ThreadManager Versions

This directory contains experimental ThreadManager implementations that were explored during research but **not validated for production use** in the ICCAD 2025 paper.

## ⚠️ Warning

**These versions are NOT recommended for production use.** They are research prototypes that may have:
- Performance limitations
- Incomplete testing
- Lack of documentation
- Potential bugs or edge cases

## Production Versions

For production use, please use the validated versions documented in the ICCAD 2025 paper:

| Version | Name | Best For | Location |
|---------|------|----------|----------|
| **V1** | PriorityQueue | Sparse activation patterns (e.g., DGXSim) | `include/sim/ThreadManagerV1/` |
| **V2** | Barrier | C++20 barrier-based synchronization | `include/sim/ThreadManagerV2/` |
| **V3** | PrebuiltTaskList | Memory-intensive workloads (e.g., GPUSim) | `include/sim/ThreadManagerV3/` |
| **V6** | LocalTaskQueue | Lock-optimized V1 | `include/sim/ThreadManagerV6/` |

## Experimental Versions

### ThreadManagerV4
- **Approach**: Dedicated thread per simulator (no thread pooling)
- **Status**: Explored but not validated
- **Limitation**: High context switching overhead when #simulators > #cores

### ThreadManagerV5
- **Approach**: Simplified V1 with active bit mask optimization removed
- **Status**: Explored but not validated
- **Limitation**: Slightly slower than V1

### ThreadManagerV7
- **Approach**: Hybrid - C++20 barriers + pre-built task lists
- **Status**: Explored but not validated
- **Characteristics**: Most concise implementation (109 lines)

### ThreadManagerV8
- **Approach**: Experimental refinement of V3
- **Status**: Research prototype
- **Characteristics**: Similar to V3 with experimental modifications

## How to Choose a ThreadManager

See the main documentation: `docs/for-developers/thread-manager-selection-guide.md`

1. **Profile your workload** using ACALSim's profiling tools
2. **Check simulation patterns**:
   - Sparse activation → V1 (PriorityQueue)
   - Memory-intensive → V3 (PrebuiltTaskList)
   - V1 with lock contention → V6 (LocalTaskQueue)
3. **Run experiments** with different versions

## References

- ICCAD 2025 Paper: "ACALSim: A Multi-Threaded Simulation Framework for Large-Scale Parallel System Design Space Exploration"
- Figure 5 & 7: Performance comparison of V1, V3, and V6
- Section IV: Thread Manager Specialization

## Git History

These experimental versions can be found in the git history if needed for research purposes. They were moved to this directory during the ThreadManager consolidation refactoring to clearly separate production-ready from experimental code.
