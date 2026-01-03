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

# ACALSim Example Programs {#examples}

This page lists all comprehensive example programs in the `src/` directory. Each example demonstrates specific features of the ACALSim discrete-event simulation framework.

## Quick Start Template

- @ref src/ProjectTemplate/ProjectTemplate.cc "ProjectTemplate" - Complete template for creating custom simulations
  - Shows three-step workflow: SimBase → SimTop → main()
  - Configuration system integration
  - CLI argument handling
  - Simulator registration and port connections

## Fundamental Communication Examples

### Port-Based Communication
- @ref src/testSimPort/main.cc "testSimPort" - CPU-Bus-Memory architecture
  - MasterPort/SlavePort communication
  - Backpressure handling
  - Multi-master arbitration
  - Outstanding request tracking

### Channel-Based Communication
- @ref src/testChannel/testChannel.cc "testChannel" - TrafficGenerator-NOC-Cache system
  - Lock-free dual-queue ping-pong buffers
  - Asynchronous message passing
  - Multi-tick latency modeling

- @ref src/testSimChannel/testSimChannel.cc "testSimChannel" - Bidirectional channel communication
  - Request-response protocol
  - Channel vs. Port comparison

### Mixed Communication
- @ref src/testCommunication/testCommunication.cc "testCommunication" - PE mesh with mixed ports and channels
  - Port and Channel integration
  - Visitor pattern for packet dispatch
  - PE mesh topology

## Advanced System Examples

### Accelerator Architectures
- @ref src/testAccelerator/testAccelerator.cc "testAccelerator" - Host-accelerator offloading
  - MCPU task generator
  - PE array (16 processing elements)
  - Central NOC fabric
  - Cache hierarchy

- @ref src/testBlackBear/testBlackBear.cc "testBlackBear" - BlackBear AI accelerator
  - 4×4 PE array with scratchpad memories
  - Dual NOC (Request + Data planes)
  - Multi-level cache clusters
  - PyTorch JIT integration

### Tile-Based Architectures
- @ref src/testPETile/testPETile.cc "testPETile" - PE tile with AXI bus
  - AXI interconnect
  - Local SRAM
  - Scalable mesh extension

## RISC-V ISA Simulation

### Full Implementation
- @ref src/riscv/main.cc "riscv" - RISC-V RV32I ISA simulator
  - Complete RV32I instruction set
  - Separate IF and EXE pipeline stages
  - Event-driven execution
  - Backpressure and retry mechanisms
  - Assembly parser and emulator

### Simplified Template
- @ref src/riscvSimTemplate/main.cc "riscvSimTemplate" - Educational RISC-V template
  - Simplified architecture (no separate EXE stage)
  - Good starting point for learning
  - Extensible to full pipeline

## Framework Feature Examples

### Configuration System
- @ref src/testConfig/main.cc "testConfig" - Configuration management
  - SimConfig parameter registration
  - JSON configuration parsing
  - CLI11 argument integration
  - Configuration priority (CLI > JSON > defaults)

### Memory Management
- @ref src/testResourceRecycling/testResourceRecycling.cc "testResourceRecycling" - RecycleContainer object pooling
  - `acquire<T>()` and `renew()` patterns
  - Automatic recycling
  - Performance benefits (5-10× speedup)
  - Zero manual new/delete

### SystemC Integration
- @ref src/testSTSim/testSTSim.cc "testSTSim" - Basic SystemC integration
- @ref src/testSTSystemC/testSTSystemC.cc "testSTSystemC" - Advanced SystemC example
  - STSimBase for SystemC modules
  - sc_in/sc_out port usage
  - Clock-driven simulation
  - MAC layer simulation

## Basic Test Examples

- @ref src/test/test.cc "test" - Basic event-driven simulation
  - TrafficGenerator → NOC → Cache
  - Event scheduling patterns
  - Inter-simulator communication

## Navigation

- **Files → File List**: Browse all source files with full documentation
- **Classes → Class List**: Browse all simulator components
- **Main Page**: Return to framework overview

Each example includes:
- ASCII architecture diagrams
- Communication flow charts
- Complete code with inline documentation
- Usage examples and CLI options
- Extension points for customization
- Performance characteristics

For detailed information, click on any example link above or navigate to **Files → File List** to browse all documented source files.
