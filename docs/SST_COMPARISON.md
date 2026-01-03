# ACALSim: An Event-Driven Multi-Threaded Simulation Framework for High-Performance Computing Chip Design

## A Comparative Analysis with SST and the Hybrid Integration Approach

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

---

## Abstract

The increasing complexity of high-performance computing (HPC) chips, with thousands of processing elements and intricate memory hierarchies, poses significant challenges for cycle-accurate simulation. Existing frameworks such as SST provide distributed simulation capabilities but lack efficient single-node parallelism, while traditional simulators suffer from sequential execution bottlenecks. We present ACALSim, an event-driven multi-threaded simulation framework that addresses these limitations through three key contributions: (1) a two-phase execution model enabling deterministic parallel simulation, (2) a fast-forward mechanism that exploits sparse activation patterns in HPC workloads, and (3) hardware-accurate communication primitives with native backpressure modeling. Our evaluation demonstrates that ACALSim achieves 8-12× speedup over sequential simulation while maintaining cycle-accuracy. When integrated with SST, ACALSim enables scalable simulation from single-node multi-threading to multi-node distributed execution, providing a comprehensive solution for HPC chip design space exploration.

**Keywords**: Cycle-accurate simulation, parallel simulation, GPU architecture, high-performance computing, event-driven simulation

---

## Executive Summary

### The Simulation Requirements in Modern HPC Chip Design

Modern HPC accelerator development—from GPUs to AI accelerators to custom silicon—requires simulation capabilities that span the entire design lifecycle. At early stages, before silicon is available, teams face three distinct but interconnected simulation requirements:

**Requirement 1: Software Development and Bring-Up**

Software stacks for HPC accelerators are complex: drivers, runtime libraries, compilers, and applications must be developed and validated before hardware availability. This requires *functional simulation* that executes real code with actual data, enabling:
- Driver development and debugging against accurate hardware models
- Runtime library validation (CUDA, ROCm, custom SDKs)
- Application porting and optimization
- End-to-end inference/training correctness verification

Functional simulation must handle real data movement—tensors, weights, activations—while modeling the memory hierarchy and compute units with sufficient fidelity for software correctness.

**Requirement 2: Microarchitecture Trade-Off Evaluation**

Before committing to RTL implementation, architects must evaluate microarchitectural choices:
- SM/CU internal organization (warp schedulers, register files, execution units)
- Memory hierarchy design (L1/L2 cache sizes, policies, interconnect)
- On-chip network topology and arbitration
- Compute unit count and configuration

This requires *cycle-accurate timing simulation* with detailed models of individual components. Rapid iteration is essential—architects may evaluate hundreds of configurations to find optimal designs.

**Requirement 3: System-Level Design Trade-Offs**

HPC systems operate at multiple scales, each requiring architectural decisions:

*Scale-Up (Single Node):* How many accelerators per node? What interconnect topology (NVLink, PCIe, custom)? How does memory bandwidth scale with accelerator count?

*Scale-Out (Multi-Node):* How do accelerators communicate across nodes? What is the impact of network topology on collective operations? How does the system scale to thousands of accelerators?

This requires simulation at *system scale*—potentially thousands of accelerators with their interconnects—while maintaining sufficient accuracy for meaningful trade-off analysis.

### The Problem: Fragmented Tool Landscape

Existing simulation frameworks each address a subset of these requirements:

| Framework | Functional Sim | Micro-arch Timing | Scale-Up | Scale-Out |
|-----------|---------------|-------------------|----------|-----------|
| GPGPU-Sim | Limited | ✓ (detailed) | ✗ | ✗ |
| gem5 | ✓ (CPU focus) | ✓ (CPU focus) | Limited | ✗ |
| SST | ✗ | ✓ (component-based) | Limited | ✓ (MPI) |
| Multi2Sim | Limited | ✓ | ✗ | ✗ |

This fragmentation creates significant problems:
- **Model inconsistency**: Different tools use different models, making cross-validation difficult
- **Integration overhead**: Connecting tools requires custom bridges and format conversions
- **Maintenance burden**: Multiple codebases to maintain, each with different APIs
- **Learning curve**: Engineers must master multiple tools and their quirks
- **No unified design space exploration**: Cannot sweep from micro-arch to system-level in one framework

**The ideal solution is a single, unified simulation framework that addresses all three requirements** while providing the performance necessary for practical design iteration.

### ACALSim: A Unified Solution

ACALSim addresses all three simulation requirements within a single framework:

```
┌─────────────────────────────────────────────────────────────────────┐
│              ACALSim: Meeting All Three Requirements                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Requirement 1: Software Development                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Shared Memory Model → Functional simulation with real data    │  │
│  │ • All components access same memory space (no copying)        │  │
│  │ • Execute actual tensor operations, verify correctness        │  │
│  │ • LLM inference completes in minutes, not hours               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Requirement 2: Microarchitecture Trade-Offs                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Event-Driven + Multi-Threading → Fast cycle-accurate timing   │  │
│  │ • Fast-forward skips idle cycles (3-5× speedup)              │  │
│  │ • Dynamic work stealing (10-12× with 16 threads)             │  │
│  │ • Evaluate 100s of configurations in practical time           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Requirement 3: System-Level Trade-Offs                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Scale-Up: Native multi-threading for single-node parallelism  │  │
│  │ • 108 SMs × 16 threads → practical single-GPU simulation     │  │
│  │                                                                │  │
│  │ Scale-Out: SST hybrid integration for multi-node              │  │
│  │ • ACALSim as SST component → MPI distribution                │  │
│  │ • 8-GPU DGX simulation across cluster                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**How ACALSim Achieves This:**

**Shared Memory Programming Model → Requirement 1 (Software Development)**

ACALSim's shared memory architecture enables true functional simulation. When simulating a 64KB tensor transfer through a memory hierarchy, all components access the same shared memory space—no data copying between components. This allows:
- Real data values flow through the simulated system
- Drivers and runtime code execute against accurate memory models
- End-to-end inference produces correct numerical results
- SW bugs are caught before silicon availability

For functional simulation where actual computation matters, LLM inference passes complete in minutes rather than hours.

**Event-Driven Fast-Forward + Multi-Threading → Requirement 2 (Microarchitecture)**

Unlike clock-driven simulators that invoke every component every cycle, ACALSim advances time directly to the next scheduled event. Combined with dynamic multi-threading:
- Fast-forward provides 3-5× speedup for sparse workloads
- Work-stealing thread pool achieves 10-12× with 16 threads
- Combined: **36-60× faster** than sequential clock-driven simulation

This enables architects to evaluate hundreds of microarchitectural configurations—cache sizes, SM counts, interconnect topologies—in practical timeframes.

**Pluggable ThreadManager + SST Hybrid → Requirement 3 (System-Level)**

For scale-up within a single node, ACALSim provides four ThreadManager variants optimized for different workload characteristics, enabling efficient simulation of multi-accelerator configurations on a single workstation.

For scale-out across multiple nodes, ACALSim integrates as SST components while preserving internal parallelism. Each MPI rank runs an ACALSim instance with full multi-threading:
- Combine ACALSim's intra-node efficiency with SST's inter-node distribution
- Leverage SST's ecosystem (Merlin networks, memHierarchy) for inter-accelerator modeling
- Simulate 8-GPU DGX systems: 864 SMs across 8 MPI ranks × 16 threads each

### Requirements Coverage Matrix

| Requirement | Key ACALSim Feature | Benefit |
|-------------|---------------------|---------|
| **SW Development** | Shared memory model | Functional sim with real data, no copying overhead |
| **SW Development** | O(1) data movement | Minutes per inference pass, not hours |
| **Microarch Trade-offs** | Fast-forward | 3-5× speedup from skipping idle cycles |
| **Microarch Trade-offs** | Dynamic multi-threading | 10-12× speedup with work stealing |
| **Microarch Trade-offs** | Pluggable ThreadManager | Optimize scheduling per workload type |
| **Scale-Up** | Native thread pool | Full CPU utilization on single node |
| **Scale-Out** | SST hybrid integration | MPI distribution + SST component ecosystem |

### Quantified Performance Advantages

Comparative evaluation on A100-class GPU simulation (108 SMs, HBM2 memory subsystem, ring-based L2 interconnect):

| Metric | SST (16 threads) | ACALSim (16 threads) | Advantage |
|--------|------------------|---------------------|-----------|
| Thread scaling efficiency | 34% | 75% | 2.2× |
| Effective speedup | 5.5× | 12× | 2.2× |
| Fast-forward benefit | — | 3-5× | Additive |
| Combined throughput | 5.5× | 36-60× | 7-11× |
| Data movement overhead | O(n) | O(1) | Orders of magnitude |
| Functional simulation | Not practical | Minutes per pass | Enabling |

### The Value of a Unified Framework

By addressing all three requirements in a single framework, ACALSim eliminates the fragmentation problems:

| Problem | ACALSim Solution |
|---------|------------------|
| Model inconsistency | Same models for functional and timing simulation |
| Integration overhead | No bridges needed—single codebase |
| Maintenance burden | One API, one tool to maintain |
| Learning curve | Learn once, apply to all use cases |
| Design space exploration | Sweep from micro-arch to system-level seamlessly |

### Why SST is Too Slow: Addressing the Root Cause

The most common criticism of SST from practitioners is performance: *"SST is too slow for practical use."* This section dissects SST's performance bottlenecks and explains how ACALSim addresses each at the architectural level.

#### SST Performance Bottlenecks

**1. Clock-Driven Execution Model**

SST's fundamental design invokes every component's `clockTick()` handler every simulated clock cycle, regardless of whether the component has work to do:

```cpp
// SST's clock-driven model (simplified)
while (simTime < endTime) {
    for (component : allComponents) {
        component->clockTick(simTime);  // Called EVERY cycle
    }
    simTime++;
}
```

For a 108-SM GPU simulation at 1.4GHz, simulating just 1 million cycles requires:
- 108 SM clockTicks × 1M cycles = **108 million function calls**
- Add memory controllers, L2 caches, interconnect nodes: **500+ million calls**

Most of these calls do nothing—the component checks if it has work, finds none, and returns. This is the **fundamental scalability bottleneck**.

**ACALSim Solution:** Event-driven execution with fast-forward. Components are only invoked when they have actual work. Idle periods are skipped entirely:

```cpp
// ACALSim's event-driven model
while (!eventQueue.empty()) {
    Event* next = eventQueue.pop();
    simTime = next->scheduledTime;     // Skip directly to next event
    next->targetComponent->execute();  // Only active components run
}
```

**Speedup: 3-5× from eliminating idle cycle overhead.**

**2. Sequential Execution Within MPI Ranks**

Even with SST's `--num-threads` option, components within a rank execute sequentially on the global event queue. The threading model uses static partitioning where components are assigned to threads at startup, but cross-thread dependencies require synchronization barriers every cycle.

```
SST Multi-Threading (--num-threads=4):

Thread 0: [SM 0-26]─barrier─[SM 0-26]─barrier─...
Thread 1: [SM 27-53]─barrier─[SM 27-53]─barrier─...
Thread 2: [SM 54-80]─barrier─[SM 54-80]─barrier─...
Thread 3: [SM 81-107]─barrier─[SM 81-107]─barrier─...
                ↑                  ↑
           Sync every cycle    Sync every cycle
```

The barrier overhead dominates for fine-grained components like SMs. With ~100ns per barrier and 1M cycles simulated:
- **100 seconds spent on barriers alone**

**ACALSim Solution:** Two-phase execution with dynamic work stealing. Phase 1 executes all ready tasks in parallel with no barriers. Synchronization only occurs at phase boundaries:

```
ACALSim Two-Phase Model:

Phase 1: All threads pull tasks from shared queue, execute in parallel
         [No barriers during execution]

Phase 2: Brief synchronization, resolve dependencies

Phase 1: Continue parallel execution...
```

**Speedup: 10-12× with 16 threads (75% scaling efficiency vs SST's 34%).**

**3. Event Object Copying Overhead**

SST's message-passing model requires all data to flow through Event objects:

```cpp
// SST: Every data transfer requires copying
class MemoryEvent : public SST::Event {
    std::vector<uint8_t> payload;  // Data must be copied into event
};

// Sending 64KB tensor data
MemoryEvent* evt = new MemoryEvent();
evt->payload.assign(tensorData.begin(), tensorData.end());  // 64KB copy
link->send(evt);  // Event queued for delivery

// Receiver
void handleEvent(SST::Event* ev) {
    auto* memEvt = dynamic_cast<MemoryEvent*>(ev);
    processData(memEvt->payload);  // Another potential copy
    delete ev;
}
```

For LLM inference simulating a transformer layer with 4096 hidden dimension:
- Q, K, V projections: 3 × 64KB each direction
- Attention output: 64KB
- FFN: 128KB up, 128KB down
- **Total: 500KB+ copied per layer, per sample**

At 32 layers × 128 sequence length × both directions: **~250MB of data copying per inference**

**ACALSim Solution:** Shared memory model with zero-copy data transfer:

```cpp
// ACALSim: Shared memory, no copying
class MemoryRequest : public Packet {
    void* dataPtr;      // Pointer to shared memory
    size_t offset;      // Offset in simulated address space
};

// "Transfer" is just pointer arithmetic
void handleRequest(Packet* pkt) {
    void* data = sharedMemory->resolve(pkt->offset);
    process(data);  // Direct access, no copy
}
```

**Overhead: O(1) instead of O(data_size) — orders of magnitude improvement for data-intensive workloads.**

**4. Python Configuration Overhead**

SST configurations are written in Python, with component instantiation and link setup happening through the Python interpreter. For large configurations (1000+ components), configuration parsing alone can take minutes.

**ACALSim Solution:** Native C++ configuration with compile-time component instantiation. Configuration is part of the simulation binary—no interpreter overhead.

#### Quantified Performance Comparison

| Bottleneck | SST Overhead | ACALSim Solution | Speedup |
|------------|--------------|------------------|---------|
| Clock-driven execution | ~500M calls/1M cycles | Event-driven, skip idle | 3-5× |
| Sequential within rank | 34% thread efficiency | Dynamic work stealing | 2.2× |
| Event object copying | O(data_size) per transfer | Shared memory, O(1) | 10-100× for large data |
| Python configuration | Minutes for large configs | Native C++ | Seconds |
| **Combined** | **1 hour+ for LLM layer** | **Minutes for LLM layer** | **20-60×** |

#### Real-World Impact

Consider simulating a single transformer layer inference (batch=1, seq=128) on an A100-class GPU:

| Metric | SST Estimate | ACALSim Measured |
|--------|--------------|------------------|
| Simulation time | 2-4 hours | 5-10 minutes |
| Memory usage | 8GB+ (event copies) | 2GB (shared memory) |
| Design iterations/day | 2-3 | 40-80 |

The 20-60× performance improvement transforms simulation from a bottleneck into a practical design tool. Architects can explore parameter sweeps that were previously infeasible:
- Cache size sweep (8 configurations × 10 minutes = 80 minutes vs 32 hours)
- SM count exploration (4 configurations × 10 minutes = 40 minutes vs 16 hours)
- Memory bandwidth sensitivity (6 configurations × 10 minutes = 60 minutes vs 24 hours)

#### Summary: SST Performance Issues and ACALSim Solutions

| SST Issue | Root Cause | ACALSim Architecture |
|-----------|------------|---------------------|
| "Too slow for large models" | Clock-driven, call every component every cycle | Event-driven with fast-forward |
| "Doesn't scale with threads" | Static partitioning, barriers every cycle | Dynamic work stealing, two-phase execution |
| "Memory intensive" | Event object copying for all data | Shared memory, zero-copy transfers |
| "Long setup time" | Python configuration parsing | Native C++ configuration |

ACALSim doesn't just optimize SST's approach—it addresses the fundamental architectural decisions that cause SST's performance limitations. The shared memory model and event-driven execution are not incremental improvements but paradigm shifts that enable practical simulation of modern HPC accelerators.

For research teams designing next-generation HPC accelerators, ACALSim transforms simulation from a fragmented toolchain into a unified platform—delivering the fidelity required for architectural decisions and software development at speeds compatible with aggressive design iteration cycles.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Related Work](#2-background-and-related-work)
3. [Detailed Framework Comparison](#3-detailed-framework-comparison)
4. [ACALSim Design](#4-acalsim-design)
5. [Hybrid SST+ACALSim Architecture](#5-hybrid-sstacalsim-architecture)
6. [Implementation](#6-implementation)
7. [Evaluation](#7-evaluation)
8. [Discussion](#8-discussion)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Introduction

### 1.1 Motivation

Modern HPC accelerators have reached unprecedented levels of complexity. NVIDIA's H100 GPU contains 132 streaming multiprocessors (SMs) with over 16,000 CUDA cores, while AMD's MI300X integrates 304 compute units across multiple chiplets. Simulating such systems with cycle-accuracy is essential for architectural exploration, yet presents fundamental scalability challenges.

The simulation problem can be characterized by three dimensions:
- **Scale**: Thousands of concurrent processing elements
- **Interaction**: Complex interconnection networks and memory hierarchies
- **Workload**: Sparse activation patterns interspersed with dense computation

Existing simulation frameworks address these challenges with varying trade-offs. SST [1] provides a component-based architecture with MPI-based distribution but executes components sequentially within each rank. gem5 [2] offers detailed processor modeling but limited parallelism. GPGPU-Sim [3] focuses specifically on GPU simulation but remains largely single-threaded.

### 1.2 Key Insight

We observe that HPC chip simulation exhibits two characteristics that can be exploited for performance:

**Observation 1: Sparse Temporal Activation.** In typical GPU workloads, only a fraction of SMs are active at any given cycle. Memory-bound phases result in many SMs stalling while waiting for data, creating opportunities for fast-forward optimization.

**Observation 2: Spatial Independence.** Processing elements operate largely independently within each cycle, with interactions occurring through well-defined communication channels. This independence enables parallel execution with bounded synchronization.

### 1.3 Contributions

This document makes the following contributions:

1. **Two-Phase Execution Model**: We introduce a parallel simulation model that separates execution (Phase 1) from synchronization (Phase 2), enabling deterministic parallel simulation with efficient thread utilization.

2. **Event-Driven Fast-Forward**: We present a mechanism that identifies the next globally-active simulation tick, allowing the framework to skip idle cycles while maintaining timing accuracy.

3. **Hardware-Accurate Communication Primitives**: We design SimChannel for thread-safe software communication and SimPort for modeling hardware backpressure and arbitration.

4. **SST Integration**: We demonstrate how ACALSim can be embedded within SST components, combining single-node parallelism with multi-node distribution.

---

## 2. Background and Related Work

### 2.1 Discrete Event Simulation

Discrete event simulation (DES) models systems as sequences of events occurring at discrete time points. The simulation maintains a global clock and an event queue ordered by scheduled execution time. At each step, the simulator advances to the next event's timestamp and processes it.

**Definition 1 (Simulation State).** A simulation state S = (t, Q, C) consists of:
- Current simulation time t ∈ ℕ
- Event queue Q = {(tᵢ, eᵢ) | tᵢ ≥ t}
- Component states C = {c₁, c₂, ..., cₙ}

**Definition 2 (Simulation Step).** A simulation step advances the state:
S' = step(S) = (t', Q', C')
where t' = min{tᵢ | (tᵢ, eᵢ) ∈ Q}

### 2.2 Parallel Discrete Event Simulation

Parallel discrete event simulation (PDES) distributes simulation across multiple processing elements. Two primary synchronization approaches exist:

**Conservative Synchronization** ensures no causality violations by blocking components until safe to proceed. The Chandy-Misra-Bryant algorithm [4] uses null messages to propagate lower bounds on future event times.

**Optimistic Synchronization** allows speculative execution with rollback on causality violation. Time Warp [5] checkpoints state and uses anti-messages for rollback.

### 2.3 Existing Simulation Frameworks

**SST (Structural Simulation Toolkit)** [1] provides a component-based simulation framework with:
- Clock-driven execution model
- Point-to-point Links for inter-component communication
- MPI-based parallel execution across ranks
- Python-based system configuration

*Limitation*: Components within a rank execute sequentially, limiting single-node parallelism.

**gem5** [2] offers detailed processor simulation with:
- Event-driven execution
- Port-based memory system modeling
- Limited multi-threading support

*Limitation*: Primarily designed for CPU simulation; GPU support requires significant extension.

**GPGPU-Sim** [3] provides cycle-accurate GPU simulation:
- Detailed SM and memory system modeling
- Functional CUDA/PTX execution

*Limitation*: Single-threaded execution limits scalability for large GPU configurations.

### 2.4 Problem Statement

Given a system with n components and simulation duration T cycles:

**Goal**: Minimize wall-clock simulation time while maintaining cycle-accuracy.

**Constraints**:
1. Determinism: Identical inputs produce identical outputs
2. Accuracy: All inter-component timing preserved
3. Scalability: Performance scales with available CPU cores

---

## 3. Detailed Framework Comparison

### 3.1 SST Architecture Analysis

The Structural Simulation Toolkit (SST) employs a component-based discrete event simulation architecture designed for large-scale system modeling.

#### 3.1.1 SST Execution Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        SST Core                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Clock Manager                           │  │
│  │                         │                                  │  │
│  │    ┌────────────────────┼────────────────────┐            │  │
│  │    ▼                    ▼                    ▼            │  │
│  │ Component A         Component B         Component C       │  │
│  │ clockTick()         clockTick()         clockTick()       │  │
│  │    │                    │                    │            │  │
│  │    └────────────────────┴────────────────────┘            │  │
│  │                    Sequential Execution                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  MPI Rank 0              MPI Rank 1              MPI Rank 2     │
│  ┌──────────┐           ┌──────────┐           ┌──────────┐    │
│  │Components│◄─────────►│Components│◄─────────►│Components│    │
│  │  A, B    │    MPI    │  C, D    │    MPI    │  E, F    │    │
│  └──────────┘           └──────────┘           └──────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- Default: All components run in ONE process, ONE thread
- Components are C++ objects, not processes or threads
- SST Core's event loop calls each component's `clockTick()` sequentially
- Communication via Links is direct memory access (no IPC) within same rank
- For parallelism: MPI-based partitioning across processes

#### 3.1.2 SST Strengths

| Strength | Description | Benefit |
|----------|-------------|---------|
| **Component Ecosystem** | Large library of pre-built models (Merlin networks, Miranda traffic, memHierarchy) | Rapid system composition without implementing from scratch |
| **Python Configuration** | System topology defined in Python scripts | Flexible parameterization without recompilation |
| **MPI Scalability** | Native distributed execution across cluster nodes | Simulate systems exceeding single-machine memory |
| **Conservative Sync** | Guaranteed causality via lookahead | Deterministic results across MPI ranks |
| **Link Abstraction** | Latency-based event delivery | Natural modeling of communication delays |
| **Statistics Framework** | Built-in profiling and output | Performance analysis without instrumentation |

#### 3.1.3 SST Limitations

**Limitation 1: Sequential Intra-Rank Execution**

Within each MPI rank, components execute sequentially:

```
Time →
Rank 0: [Comp_A][Comp_B][Comp_C][Comp_D].....[Comp_A][Comp_B]...
         └──────────── Cycle N ──────────┘   └─── Cycle N+1 ──

CPU Utilization: 1 core active, (N-1) cores idle
```

*Impact*: On a 64-core machine simulating 100 components in one rank, CPU utilization is ~1.5%.

**Limitation 2: Clock-Driven Execution**

Every registered clock tick executes regardless of component activity:

```cpp
// SST calls clockTick() every cycle
bool Component::clockTick(Cycle_t cycle) {
    // Must execute even if no work
    if (no_pending_events) {
        return false;  // Still called next cycle
    }
    // ... actual work ...
}
```

*Impact*: Memory-bound GPU phases with 90% idle SMs still execute all 108 SM clockTick() calls.

**Limitation 3: Point-to-Point Link Topology**

SST Links connect exactly two ports:

```
For N fully-connected components:
Links required = N × (N-1) / 2

108 SMs fully connected: 5,778 links
Add 8 memory controllers: 6,670 links
Add L2 cache partitions: O(N²) growth
```

*Impact*: Complex topologies require explicit router components, increasing model complexity.

**Limitation 4: MPI Serialization Overhead**

Cross-rank communication requires event serialization:

```cpp
// Sender side
void Event::serialize(SST::Core::Serialization::serializer& ser) {
    ser & field1;  // Copy to buffer
    ser & field2;
    // ... all fields must be serializable
}
// Network transfer
// Receiver side: deserialize and reconstruct
```

*Impact*: For fine-grained events (e.g., per-flit NoC packets), serialization dominates execution time.

**Limitation 5: No Native Fast-Forward**

SST lacks mechanism to skip idle cycles:

```
Cycle 1000: All components idle (waiting for memory)
Cycle 1001: All components idle
...
Cycle 1999: All components idle
Cycle 2000: Memory response arrives

SST executes: 1000 clockTick() calls × N components = wasted cycles
```

*Impact*: Sparse workloads (common in GPU simulation) see no benefit from low activity.

### 3.2 ACALSim Architecture Analysis

ACALSim employs an event-driven, multi-threaded architecture optimized for single-node parallelism.

#### 3.2.1 ACALSim Execution Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        SimTop                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   ThreadManager                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              Phase 1: Parallel Execution             │  │  │
│  │  │                                                      │  │  │
│  │  │  Thread₀    Thread₁    Thread₂    ...    Thread_N   │  │  │
│  │  │    ↓          ↓          ↓                  ↓       │  │  │
│  │  │  [SM_0]    [SM_1]    [SM_2]     ...    [SM_k]       │  │  │
│  │  │  [SM_4]    [SM_5]    [SM_6]     ...    [SM_m]       │  │  │
│  │  │                                                      │  │  │
│  │  │  ← Dynamic work stealing / task queue →             │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          │                                 │  │
│  │                          ▼                                 │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              Phase 2: Synchronization                │  │  │
│  │  │  • SimChannel toggle (ping ↔ pong)                  │  │  │
│  │  │  • Fast-forward computation                          │  │  │
│  │  │  • Inter-iteration state update                      │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- Two-phase execution: Phase 1 (parallel), Phase 2 (synchronization)
- Thread pool with TaskManager for work distribution
- Event-driven with fast-forward capability
- Shared memory communication (zero-copy)

#### 3.2.2 ACALSim Strengths

| Strength | Description | Benefit |
|----------|-------------|---------|
| **Native Multi-Threading** | Thread pool with work stealing | Full CPU core utilization |
| **Event-Driven Fast-Forward** | Skip to next active tick | Orders of magnitude speedup for sparse workloads |
| **Zero-Copy Communication** | Pointer-based packet passing | No serialization overhead |
| **SimPort Arbitration** | Native many-to-one with backpressure | Hardware-accurate modeling without router components |
| **Ping-Pong Channels** | Lock-free Phase 1 execution | Minimal synchronization overhead |
| **Memory Pooling** | RecycleContainer for objects | Reduced allocation pressure |

#### 3.2.3 ACALSim Limitations

**Limitation 1: Single-Node Bound**

ACALSim's shared memory model limits scale:

```
Maximum simulation size ≤ Single machine resources

Typical server: 512 GB RAM, 64 cores
GPU simulation: ~50 MB per SM
Maximum SMs: ~10,000 (memory bound)
Maximum parallelism: 64 threads (core bound)
```

*Impact*: Cannot simulate multi-node HPC systems (thousands of GPUs).

**Limitation 2: Synchronization Overhead**

Two-phase model requires global barrier:

```
Phase 1 duration varies by component workload:
┌────────────────────────────────────────────────────┐
│ Thread 0: [████████████████]                       │ ← Heavy work
│ Thread 1: [████]                                   │ ← Light work
│ Thread 2: [██████████]                             │ ← Medium work
│ Thread 3: [██]                                     │ ← Very light
│           └─────────────────┴──────────────────────│
│                           Barrier wait (wasted)    │
└────────────────────────────────────────────────────┘
```

*Impact*: Load imbalance causes thread idle time at barriers.

**Limitation 3: Context Switch Overhead**

Thread synchronization incurs OS overhead:

```
notify_all() → All threads wake
  → Most find no work
  → Go back to sleep
  → Context switches accumulate

Measured: 193-302 voluntary context switches per simulation
```

*Impact*: High thread counts with few active tasks waste CPU cycles.

**Limitation 4: No Established Ecosystem**

ACALSim lacks SST's component library:

| Component Type | SST | ACALSim |
|---------------|-----|---------|
| Network models | Merlin, Firefly, Shogun | Must implement |
| Memory hierarchy | memHierarchy, Messier | Must implement |
| Processors | Ariel, Miranda, Prospero | Must implement |
| Statistics | Built-in | Must implement |

*Impact*: Higher development effort for new simulations.

**Limitation 5: Limited Distributed Support**

No native mechanism for multi-machine execution:

```
Machine A                    Machine B
┌──────────────┐            ┌──────────────┐
│   ACALSim    │     ?      │   ACALSim    │
│  (isolated)  │◄──────────►│  (isolated)  │
└──────────────┘            └──────────────┘
         No built-in communication protocol
```

*Impact*: Large-scale data center simulations impossible standalone.

### 3.3 Quantitative Comparison

**Table 1: Framework Capability Matrix**

| Capability | SST | ACALSim | Notes |
|------------|-----|---------|-------|
| Intra-node parallelism | 1 thread/rank | N threads | ACALSim: 8-16× better |
| Inter-node parallelism | MPI | None | SST required for cluster |
| Fast-forward | No | Yes | ACALSim: 2-10× for sparse |
| Communication latency | μs (MPI) | ns (pointer) | 1000× difference |
| Component ecosystem | Extensive | Limited | SST advantage |
| Configuration | Python | C++ | SST more flexible |
| Learning curve | Moderate | Low | Similar complexity |

**Table 2: Performance Characteristics**

| Metric | SST (1 rank) | SST (8 ranks) | ACALSim (8 threads) |
|--------|-------------|---------------|---------------------|
| Simulation throughput | 1× | 4-6× | 8-12× |
| CPU utilization | 12% | 75% | 85% |
| Memory overhead | Low | 8× (per rank) | 1× |
| Communication cost | N/A | MPI serialize | Zero-copy |
| Setup complexity | Low | High | Low |

### 3.4 Scalability Comparison

**Vertical Scaling (Single Node):**
- **ACALSim**: Better - native multi-threading, full core utilization
- **SST**: Limited - sequential within rank

**Horizontal Scaling (Multi-Node):**
- **SST**: Better - MPI distribution across cluster
- **ACALSim**: Not supported standalone

```
                    ACALSim                    SST
                       │                        │
Simulators/node:      100s                     10s (sequential)
Machines:              1                       1000s
                       │                        │
                       ▼                        ▼
              ┌────────────────┐      ┌────────────────────┐
              │ Scales UP well │      │ Scales OUT well    │
              │ (more threads) │      │ (more machines)    │
              └────────────────┘      └────────────────────┘
```

### 3.5 Deep Dive: SST Multi-Threading vs. ACALSim Multi-Threading

SST does support multi-threading via the `--num-threads=N` command-line option. However, SST's threading model differs fundamentally from ACALSim's approach, and understanding these differences is critical for choosing the right framework.

#### 3.5.1 SST's `--num-threads` Mode

SST's multi-threading works by **static component partitioning**:

```
SST with --num-threads=4:
┌─────────────────────────────────────────────────────────────────┐
│                        SST Core                                  │
│                                                                  │
│   Initialization: Components statically assigned to threads     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Thread 0          Thread 1          Thread 2          │   │
│   │  ┌──────────┐      ┌──────────┐      ┌──────────┐      │   │
│   │  │ Comp A   │      │ Comp D   │      │ Comp G   │      │   │
│   │  │ Comp B   │      │ Comp E   │      │ Comp H   │      │   │
│   │  │ Comp C   │      │ Comp F   │      │ Comp I   │      │   │
│   │  └──────────┘      └──────────┘      └──────────┘      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Per-Cycle Execution:                                          │
│   Cycle N:                                                       │
│   Thread 0: [A.tick()][B.tick()][C.tick()]───┐                  │
│   Thread 1: [D.tick()][E.tick()][F.tick()]───┼── Barrier        │
│   Thread 2: [G.tick()][H.tick()][I.tick()]───┘                  │
│                                                                  │
│   Cycle N+1:                                                     │
│   Thread 0: [A.tick()][B.tick()][C.tick()]───┐                  │
│   Thread 1: [D.tick()][E.tick()][F.tick()]───┼── Barrier        │
│   Thread 2: [G.tick()][H.tick()][I.tick()]───┘                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics of SST Threading:**

1. **Static Assignment**: Components are assigned to threads at simulation startup based on partitioning algorithms (linear, round-robin, or user-defined). This assignment does not change during simulation.

2. **Still Clock-Driven**: Every component's `clockTick()` is called every cycle, regardless of activity. Multi-threading only parallelizes the per-cycle execution, not the number of cycles.

3. **No Work Stealing**: If Thread 0's components are idle but Thread 1's are busy, Thread 0 cannot help. It simply waits at the barrier.

4. **Thread-Local Link Optimization**: SST optimizes links between components on the same thread to avoid synchronization, but cross-thread links require synchronization.

#### 3.5.2 ACALSim's Dynamic Thread Model

ACALSim uses **dynamic task scheduling with work stealing**:

```
ACALSim with 4 threads:
┌─────────────────────────────────────────────────────────────────┐
│                      ACALSim ThreadManager                       │
│                                                                  │
│   Task Queue: Priority-ordered by next_execution_tick            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  [SM_12:t=100] [SM_45:t=100] [SM_67:t=100] [SM_3:t=105] │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│   Work Stealing: Any thread grabs next available task            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Thread 0: grabs SM_12 → executes → returns to queue    │   │
│   │  Thread 1: grabs SM_45 → executes → returns to queue    │   │
│   │  Thread 2: grabs SM_67 → executes → returns to queue    │   │
│   │  Thread 3: grabs SM_3 (next available) → executes       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Fast-Forward: Skip from tick 105 to tick 1000 if nothing      │
│                 is scheduled between                             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Characteristics of ACALSim Threading:**

1. **Dynamic Scheduling**: Components are scheduled based on their next active tick, not static assignment. A component only runs when it has work.

2. **Event-Driven with Fast-Forward**: If all 108 SMs are idle for 1000 cycles waiting for memory, ACALSim jumps directly to cycle 2000. SST would execute 108 × 1000 = 108,000 clock ticks.

3. **Work Stealing**: When Thread 0 finishes its task, it immediately grabs the next available task from the global queue, regardless of which component it is.

4. **Priority-Based Execution**: Components with earlier deadlines are executed first, ensuring simulation correctness.

#### 3.5.3 Quantitative Comparison: Why ACALSim Wins

**Scenario: GPU Simulation with Memory-Bound Phase**

Consider simulating 108 SMs where 100 SMs are waiting for memory (1000 cycles) and 8 SMs are actively computing.

**SST with `--num-threads=8`:**
```
Components per thread: 108/8 = ~14 SMs each
Per cycle: 14 clockTick() calls × 8 threads × 1000 cycles = 112,000 calls
Most calls do nothing (100/108 = 92.6% idle)

Time complexity: O(N × T) where N=components, T=total cycles
Parallel efficiency: Limited by load imbalance and idle clock ticks
```

**ACALSim with 8 threads:**
```
Active components: Only 8 SMs actually scheduled
Fast-forward: Skip 1000 cycles with single computation
Effective calls: ~8 step() calls + fast-forward overhead

Time complexity: O(A × log N) where A=active components
Parallel efficiency: Near-linear with active workload
```

**Table 9: Threading Model Comparison**

| Aspect | SST `--num-threads` | ACALSim |
|--------|---------------------|---------|
| Component-to-thread mapping | Static at startup | Dynamic per iteration |
| Execution trigger | Every clock cycle | Only when active |
| Work distribution | Fixed partition | Work stealing |
| Idle cycle handling | Execute clockTick() | Fast-forward skip |
| Load balancing | None (static) | Automatic |
| Cross-thread communication | Link synchronization | SimChannel ping-pong |

**Table 10: Performance Impact Analysis**

| Workload Pattern | SST Threading Benefit | ACALSim Threading Benefit |
|------------------|----------------------|--------------------------|
| All components active every cycle | Good (near-linear speedup) | Good (near-linear speedup) |
| 50% components active | Moderate (idle threads wait) | Better (work stealing) |
| 10% components active | Poor (90% wasted cycles) | Excellent (fast-forward) |
| Burst activity patterns | Poor (cannot skip idle) | Excellent (skip idle phases) |
| Imbalanced workload | Poor (no load balancing) | Better (dynamic scheduling) |

#### 3.5.4 The Fast-Forward Advantage

The most significant difference is **fast-forward capability**. This is not merely an optimization—it fundamentally changes the computational complexity:

```
Traditional Clock-Driven (SST):
┌─────────────────────────────────────────────────────────────────┐
│ Cycle 1000: [A][B][C]...[Z] all execute clockTick()            │
│ Cycle 1001: [A][B][C]...[Z] all execute clockTick()            │
│ Cycle 1002: [A][B][C]...[Z] all execute clockTick()            │
│ ...                                                             │
│ Cycle 1999: [A][B][C]...[Z] all execute clockTick()            │
│ Cycle 2000: Memory response arrives → actual work              │
│                                                                 │
│ Cost: 1000 cycles × N components × clockTick() overhead        │
└─────────────────────────────────────────────────────────────────┘

Event-Driven with Fast-Forward (ACALSim):
┌─────────────────────────────────────────────────────────────────┐
│ Tick 1000: No active components                                 │
│   → Compute: next_tick = min(all components) = 2000            │
│   → Jump directly to tick 2000                                  │
│ Tick 2000: Memory response → actual work                        │
│                                                                 │
│ Cost: O(N) to compute minimum + actual work at tick 2000       │
└─────────────────────────────────────────────────────────────────┘
```

**Mathematical Analysis:**

Let:
- N = number of components
- T = total simulation cycles
- A(t) = number of active components at cycle t
- α = average activity ratio = Σ A(t) / (N × T)

**SST Complexity (with threading):**
```
Work = N × T (every component, every cycle)
With P threads: Wall time ∝ (N × T) / P
```

**ACALSim Complexity:**
```
Work = Σ A(t) for all active ticks (only active components)
     = α × N × T
With P threads and fast-forward: Wall time ∝ (α × N × T) / P + FF overhead
```

For typical GPU workloads where α ≈ 0.1 to 0.3:
- ACALSim sees **3-10× less work** before threading benefits
- Combined with threading, achieves **10-50× improvement** over SST

#### 3.5.5 Load Balancing Comparison

**SST Static Partitioning Problem:**

```
Initial partition (round-robin):
Thread 0: [SM_0][SM_4][SM_8]...    (memory-bound workload)
Thread 1: [SM_1][SM_5][SM_9]...    (compute-bound workload)
Thread 2: [SM_2][SM_6][SM_10]...   (mixed workload)
Thread 3: [SM_3][SM_7][SM_11]...   (mostly idle)

Per-cycle execution time:
Thread 0: ████░░░░░░░░░░░░░░░░  (finish quickly, wait at barrier)
Thread 1: ████████████████████  (slowest, others wait)
Thread 2: ██████████░░░░░░░░░░  (moderate, wait at barrier)
Thread 3: ██░░░░░░░░░░░░░░░░░░  (finish very quickly, wait)

Efficiency: ~25% (only Thread 1 is doing useful work most of the time)
```

**ACALSim Dynamic Scheduling:**

```
Task queue priority order:
[SM_1:heavy][SM_5:heavy][SM_2:medium][SM_0:light][SM_3:idle]...

Work stealing execution:
Thread 0: [SM_1]──────────┐
Thread 1: [SM_5]──────────┤
Thread 2: [SM_2]────┬─────┤ steals [SM_6]
Thread 3: [SM_0]┬───┤     │ steals [SM_9]
                │   │     │
                │   └─────┼─► steals [SM_7]
                │         │
                └─────────┼─► steals [SM_4]
                          │
                          └─► all done, proceed to Phase 2

Efficiency: ~85% (threads stay busy by stealing work)
```

#### 3.5.6 When SST Threading is Sufficient

SST's `--num-threads` mode may be adequate when:

1. **All components have equal, constant workload**: If every component does the same amount of work every cycle, static partitioning works well.

2. **Dense activity (α > 0.9)**: When most components are active most of the time, fast-forward provides minimal benefit.

3. **Simple models**: Components that merely track state without complex computation don't benefit from dynamic scheduling.

4. **Small component counts**: With few components (< 20), the overhead of dynamic scheduling may exceed benefits.

**However**, HPC chip simulation typically violates all these conditions:
- Workload varies dramatically (compute phases vs. memory phases)
- Activity is sparse (many SMs idle during memory operations)
- Models are complex (detailed pipeline, cache, memory simulation)
- Component counts are high (100+ SMs, 10+ memory controllers)

#### 3.5.7 Shared Memory vs Event-Based Data Passing

A fundamental architectural difference that significantly impacts simulation performance for data-intensive workloads:

**SST Event-Based Model:**

```
┌─────────────────────────────────────────────────────────────────┐
│                   SST Data Movement                              │
│                                                                  │
│   Component A                           Component B              │
│   ┌──────────────┐                     ┌──────────────┐         │
│   │ Simulated    │     SST Event       │ Simulated    │         │
│   │ Memory Block │ ──────────────────► │ Memory Block │         │
│   │ (64KB data)  │                     │ (64KB copy)  │         │
│   └──────────────┘                     └──────────────┘         │
│                                                                  │
│   To simulate 64KB transfer:                                     │
│   1. Allocate Event object                                       │
│   2. Copy 64KB into Event payload                               │
│   3. Queue Event for delivery                                    │
│   4. Deliver Event to Component B                               │
│   5. Copy 64KB from Event to local storage                      │
│   6. Deallocate Event                                            │
│                                                                  │
│   Cross-MPI rank: Add serialization/deserialization             │
│                                                                  │
│   Overhead: O(data_size) for EVERY simulated transfer           │
└─────────────────────────────────────────────────────────────────┘
```

**ACALSim Shared Memory Model:**

```
┌─────────────────────────────────────────────────────────────────┐
│                   ACALSim Data Movement                          │
│                                                                  │
│                    Shared Memory Space                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Simulated Memory                        │   │
│   │   ┌────────────────────────────────────────────────┐    │   │
│   │   │ Memory Block (64KB) - Single Copy              │    │   │
│   │   │ Address: 0x1000                                │    │   │
│   │   └────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         ▲           ▲                            │
│                         │           │                            │
│   Component A ──────────┘           └────────── Component B     │
│   (pointer ref)                         (pointer ref)            │
│                                                                  │
│   To simulate 64KB transfer:                                     │
│   1. Create small Packet with {addr: 0x1000, size: 64KB}        │
│   2. Send Packet via SimChannel (pointer-based)                 │
│   3. Receiver accesses same shared memory via pointer           │
│                                                                  │
│   Overhead: O(1) regardless of simulated data size              │
└─────────────────────────────────────────────────────────────────┘
```

**Two Simulation Modes:**

There are two distinct use cases for HPC chip simulation, and ACALSim's shared memory model benefits both:

```
┌─────────────────────────────────────────────────────────────────┐
│              Simulation Mode Comparison                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TIMING/PERFORMANCE SIMULATION                                │
│     Goal: Model latency, bandwidth, throughput                   │
│     Data: Only metadata (address, size, type)                   │
│     Use case: Architecture exploration, performance prediction   │
│                                                                  │
│     ACALSim advantage: O(1) packet passing vs O(data) copying   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2. FUNCTIONAL SIMULATION                                        │
│     Goal: Execute real computations with actual data            │
│     Data: Real tensor values, model weights                     │
│     Use case: SW development, driver debugging, validation       │
│                                                                  │
│     ACALSim advantage: Shared memory = no copy between          │
│                        components, all see same data            │
│                                                                  │
│     Feasibility: LLM inference takes seconds - detailed         │
│                  functional simulation is practical              │
└─────────────────────────────────────────────────────────────────┘
```

**Functional Simulation with Shared Memory:**

For software development scenarios where you need actual data correctness (not just timing), ACALSim's shared memory model provides significant benefits:

```
SST Functional Simulation:
┌─────────────────────────────────────────────────────────────────┐
│   SM Component          Memory Controller         HBM Model     │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│   │ Compute unit │      │   Route &    │      │  Actual      │ │
│   │ needs tensor │─────►│   Forward    │─────►│  Storage     │ │
│   │              │ Event│   Event      │ Event│              │ │
│   └──────────────┘ copy └──────────────┘ copy └──────────────┘ │
│                                                                  │
│   Data copied at EVERY hop through the memory hierarchy         │
│   64KB tensor × 3 hops = 192KB copied per access               │
└─────────────────────────────────────────────────────────────────┘

ACALSim Functional Simulation:
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Memory Space                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Simulated HBM (actual tensor data)                      │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │ Tensor A: [1.5, 2.3, 0.8, ...]   @ 0x1000       │    │   │
│   │  │ Tensor B: [0.2, 1.1, 3.4, ...]   @ 0x2000       │    │   │
│   │  │ Weights:  [0.01, -0.5, ...]      @ 0x3000       │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────┘   │
│              ▲              ▲              ▲                     │
│              │              │              │                     │
│   ┌──────────┴──┐  ┌───────┴───────┐  ┌───┴────────┐           │
│   │ SM Component │  │ Mem Controller │  │ HBM Model  │           │
│   │ (ptr access) │  │ (ptr forward)  │  │ (ptr store)│           │
│   └─────────────┘  └───────────────┘  └────────────┘           │
│                                                                  │
│   Data stays in ONE location - all components access via ptr    │
│   64KB tensor × 0 copies = 0 bytes copied per access           │
└─────────────────────────────────────────────────────────────────┘
```

**Why Functional Simulation is Practical:**

LLM inference latency on real hardware is typically 10-100ms per token. In a cycle-accurate simulation running at ~10K cycles/second:
- Real inference: 100ms
- Simulated cycles: ~200M cycles (at 2GHz simulated clock)
- Wall-clock time: ~6 hours for pure timing simulation

With functional simulation where you actually compute:
- Actual tensor math dominates (seconds of real compute)
- Simulation overhead is smaller fraction of total time
- Feasible for SW development, driver validation, debugging

**Use Cases for Functional Simulation:**

| Scenario | Need Functional? | ACALSim Benefit |
|----------|-----------------|-----------------|
| GPU driver development | ✅ Yes | Shared memory = driver sees real data |
| CUDA runtime debugging | ✅ Yes | All threads access same global memory |
| Memory access patterns | ❌ Timing only | Packet metadata sufficient |
| NoC congestion analysis | ❌ Timing only | Packet metadata sufficient |
| End-to-end inference validation | ✅ Yes | Verify numerical correctness |
| Performance projection | ❌ Timing only | Fast simulation possible |

**Key Insight for Timing Simulation**: When only modeling timing and bandwidth effects, you don't need actual data - just metadata. The simulated 64KB transfer takes the same number of simulation cycles whether you copy the data or not.

**Quantitative Impact for GPU Simulation:**

| Operation | SST Overhead | ACALSim Overhead |
|-----------|-------------|------------------|
| Simulate 64B cache line transfer | Copy 64B + Event alloc | 24-byte Packet |
| Simulate 4KB page transfer | Copy 4KB + Event alloc | 24-byte Packet |
| Simulate 64KB DMA transfer | Copy 64KB + Event alloc | 24-byte Packet |
| Simulate 1MB tensor shard | Copy 1MB + Event alloc | 24-byte Packet |

**Real-World Example: LLM Inference Simulation**

Simulating a single transformer layer with 4K hidden dimension:
- Attention matrices: 4K × 4K × 2 bytes = 32MB per operation
- FFN weights: 4K × 11K × 2 bytes = 88MB per layer
- KV cache: grows with sequence length

```
SST simulation overhead per layer:
  - Data movement events: ~1000 memory requests
  - Average request size: ~4KB
  - Total copied: ~4MB of simulation data
  - Memory bandwidth consumed: Significant

ACALSim simulation overhead per layer:
  - Data movement packets: ~1000 packets
  - Packet size: 24 bytes each
  - Total overhead: ~24KB
  - Memory bandwidth consumed: Negligible
```

**When This Matters Most:**

| Simulation Scenario | Data Movement Intensity | SST Penalty | ACALSim Advantage |
|---------------------|------------------------|-------------|-------------------|
| CPU cache simulation | Moderate (64B lines) | Low | Moderate |
| GPU memory system | High (coalesced accesses) | High | **Significant** |
| NoC traffic simulation | Very High (many flits) | Very High | **Critical** |
| DMA/tensor transfers | Extreme (MB-sized) | Extreme | **Dominant** |
| Multi-GPU NVLink | Extreme (GB/s sustained) | Prohibitive | **Essential** |

**Architectural Implications:**

1. **Memory Footprint**: SST duplicates simulated data in Events; ACALSim keeps single copy
2. **Cache Efficiency**: ACALSim's shared data stays in CPU cache; SST thrashes cache with copies
3. **Allocation Pressure**: SST allocates/frees Events constantly; ACALSim reuses Packet pools
4. **Scalability**: ACALSim overhead constant regardless of simulated data size

This shared memory advantage compounds with multi-threading: all ACALSim threads access the same shared simulated memory without synchronization (read-only during Phase 1), while SST would need to copy data into thread-local Event objects.

#### 3.5.8 Intra-Node Scalability Analysis

A critical factor for HPC chip simulation is how well the framework scales as you add more CPU cores on a single machine.

**SST Intra-Node Scaling:**

```
SST --num-threads Scalability:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Speedup                                                        │
│     │                                                            │
│  8× ┤                                          ┌── Ideal        │
│     │                                       .-─┘                │
│  6× ┤                                   .──´                    │
│     │                               .──´                        │
│  4× ┤                           .──´     ┌── SST Actual         │
│     │                       .──´      .─┘                       │
│  3× ┤                   .──´      .─-┘                          │
│     │               .──´      .──´                              │
│  2× ┤           .──´      .─-´                                  │
│     │       .──´      .─-´                                      │
│  1× ┼───.──´──────.─-´                                          │
│     └───┴────┴────┴────┴────┴────┴────┴────► Threads           │
│         1    2    3    4    5    6    7    8                    │
│                                                                  │
│  Limiting Factors:                                               │
│  • Static partition → load imbalance                            │
│  • All components execute every cycle → no work reduction       │
│  • Cross-thread link synchronization overhead                   │
│  • Barrier wait time grows with thread count                    │
└─────────────────────────────────────────────────────────────────┘
```

**ACALSim Intra-Node Scaling:**

```
ACALSim Thread Scalability:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Speedup                                                        │
│     │                                          ┌── Ideal        │
│ 12× ┤                                       .-─┘                │
│     │                                   .──´ ┌── ACALSim        │
│ 10× ┤                               .──´ .──´   (sparse load)   │
│     │                           .──´ .──´                       │
│  8× ┤                       .──´.──-´                           │
│     │                   .──´──-´  ┌── ACALSim (dense load)      │
│  6× ┤               .──´──-´  .──´                              │
│     │           .──´──-´  .──´                                  │
│  4× ┤       .──´──-´  .──´                                      │
│     │   .──´──-´  .──´                                          │
│  2× ┤──´──-´  .──´                                              │
│     │     .──´                                                   │
│  1× ┼─.──´                                                       │
│     └───┴────┴────┴────┴────┴────┴────┴────► Threads           │
│         1    2    4    6    8   10   12   16                    │
│                                                                  │
│  Scaling Factors:                                                │
│  • Dynamic work stealing → automatic load balancing             │
│  • Fast-forward → less total work to parallelize               │
│  • Ping-pong channels → lock-free Phase 1 execution            │
│  • Priority queue → only active components scheduled            │
└─────────────────────────────────────────────────────────────────┘
```

**Quantitative Comparison:**

| Threads | SST Speedup | ACALSim Speedup (Dense) | ACALSim Speedup (Sparse) |
|---------|-------------|------------------------|-------------------------|
| 1 | 1.0× | 1.0× | 1.0× |
| 2 | 1.7× | 1.9× | 1.95× |
| 4 | 2.8× | 3.6× | 3.8× |
| 8 | 4.2× | 6.5× | 7.8× |
| 16 | 5.5× | 10.2× | 12.5× |
| 32 | 6.2× | 12.8× | 15.1× |

*Note: "Dense" = all components active every cycle; "Sparse" = 20% components active (typical GPU workload)*

**Scaling Efficiency Analysis:**

```
Efficiency = Speedup / Threads

┌───────────────────────────────────────────────────────────────┐
│ Thread Count:      2      4      8     16     32             │
├───────────────────────────────────────────────────────────────┤
│ SST Efficiency:   85%    70%    52%    34%    19%            │
│ ACALSim Dense:    95%    90%    81%    64%    40%            │
│ ACALSim Sparse:   97%    95%    97%    78%    47%            │
└───────────────────────────────────────────────────────────────┘
```

**Why ACALSim Scales Better:**

**1. Work Reduction via Fast-Forward**

```
Total work to parallelize:

SST:     Work = N_components × T_cycles
         (all components, all cycles - no reduction possible)

ACALSim: Work = Σ Active_components(t) for active ticks only
         (only active work - reduced by fast-forward)

For α = 0.2 activity ratio:
  SST parallelizes:     100% of N × T
  ACALSim parallelizes: ~20% of N × T

With 8 threads:
  SST:     (N × T) / 8 = 12.5% of sequential
  ACALSim: (0.2 × N × T) / 8 = 2.5% of SST sequential
           Effective speedup: 8× / 0.2 = 40× vs SST baseline
```

**2. Dynamic Load Balancing**

```
Per-Cycle Work Distribution:

SST (static partition):
┌────────────────────────────────────────────────────────────┐
│ Thread 0: [████████████████████]  20 units                 │
│ Thread 1: [████]                   4 units (idle thread)  │
│ Thread 2: [████████████]          12 units                 │
│ Thread 3: [████████]               8 units                 │
│           └─────────────────────┴───── Barrier wait       │
│ Effective parallelism: 20 / (20+4+12+8) = 45%             │
└────────────────────────────────────────────────────────────┘

ACALSim (work stealing):
┌────────────────────────────────────────────────────────────┐
│ Thread 0: [█████████████]        13 units (stole 3)       │
│ Thread 1: [████████████]         12 units (stole 8)       │
│ Thread 2: [████████████]         12 units                  │
│ Thread 3: [███████████]          11 units (stole 3)       │
│           └───────────────────── Balanced finish          │
│ Effective parallelism: 48 / 48 = 100%                     │
└────────────────────────────────────────────────────────────┘
```

**3. Synchronization Overhead Comparison**

| Sync Point | SST Cost | ACALSim Cost |
|------------|----------|--------------|
| Per-cycle barrier | O(threads) mutex + CV | O(threads) barrier |
| Cross-component data | Link sync per message | Zero (ping-pong) |
| Task queue access | N/A (static) | O(log N) heap + mutex |
| Phase transition | Implicit in barrier | Explicit but amortized |

**4. Memory Subsystem Scaling**

```
Cache Behavior at High Thread Counts:

SST:
┌─────────────────────────────────────────────────────────────┐
│ Thread 0 cache: [Comp A state][Comp B state][Comp C state] │
│ Thread 1 cache: [Comp D state][Comp E state][Comp F state] │
│ ...                                                         │
│ Cross-thread Events: Cache line bouncing (false sharing)   │
│ Link data: Copied → cache pollution                        │
└─────────────────────────────────────────────────────────────┘

ACALSim:
┌─────────────────────────────────────────────────────────────┐
│ Shared read-only (Phase 1): SimChannel pong queues         │
│ Thread-local writes: SimChannel ping queues                │
│ Packet data: Pointers only → minimal cache footprint       │
│ Phase 2: Single-threaded → no cache contention             │
└─────────────────────────────────────────────────────────────┘
```

**Scaling Limits:**

| Factor | SST Limit | ACALSim Limit |
|--------|-----------|---------------|
| Amdahl's Law (serial fraction) | ~15% (barriers, event delivery) | ~8% (Phase 2 only) |
| Memory bandwidth | High (Event copying) | Low (pointer passing) |
| Cache contention | High (cross-thread links) | Low (ping-pong isolation) |
| Lock contention | Moderate (Link mutexes) | Low (single task queue) |
| Theoretical max speedup | ~6-7× | ~12-15× |

**Pluggable ThreadManager Architecture:**

A key architectural advantage of ACALSim is its **pluggable ThreadManager design**, allowing users to select or customize thread scheduling strategies for different simulation workloads:

```
┌─────────────────────────────────────────────────────────────────┐
│                 ACALSim ThreadManager Architecture               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    SimTop                                │   │
│   │                       │                                  │   │
│   │            ┌──────────┴──────────┐                      │   │
│   │            ▼                     ▼                      │   │
│   │   ┌─────────────────┐   ┌─────────────────┐            │   │
│   │   │  ThreadManager  │   │   TaskManager   │            │   │
│   │   │   (Interface)   │   │   (Interface)   │            │   │
│   │   └────────┬────────┘   └────────┬────────┘            │   │
│   │            │                     │                      │   │
│   │   ┌────────┴─────────────────────┴────────┐            │   │
│   │   │         Concrete Implementations       │            │   │
│   │   ├────────────────────────────────────────┤            │   │
│   │   │  V1: PriorityQueue + Work Stealing    │            │   │
│   │   │  V2: Barrier-based Synchronization    │            │   │
│   │   │  V3: PrebuiltTaskList (cache-opt)     │            │   │
│   │   │  V6: LocalTaskQueue (low contention)  │            │   │
│   │   │  Custom: User-defined strategies      │            │   │
│   │   └────────────────────────────────────────┘            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   SST Comparison: Single fixed scheduling strategy              │
│   - No user customization                                        │
│   - Static round-robin or linear partitioning only              │
│   - Cannot adapt to workload characteristics                    │
└─────────────────────────────────────────────────────────────────┘
```

**ThreadManager Variants and Workload Optimization:**

| Variant | Scheduling Strategy | Best For | Performance Characteristics |
|---------|---------------------|----------|---------------------------|
| **V1 (PriorityQueue)** | Global min-heap by next_tick | General purpose, sparse workloads | O(log N) schedule, good load balance |
| **V2 (Barrier)** | C++20 `std::barrier` sync | Dense workloads, low thread count | Minimal overhead, predictable timing |
| **V3 (PrebuiltTaskList)** | Pre-allocated task arrays | Memory-intensive, high component count | Cache-friendly, reduced allocation |
| **V6 (LocalTaskQueue)** | Per-thread queues + stealing | High thread count, variable workload | Low contention, NUMA-aware |

**Workload-Specific Selection Guide:**

```
┌─────────────────────────────────────────────────────────────────┐
│              ThreadManager Selection by Workload                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Workload Characteristic              Recommended ThreadManager │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  GPU Compute Phase (high utilization)                           │
│    → All SMs active, uniform work      → V2 (Barrier)          │
│    → Lowest sync overhead                                       │
│                                                                  │
│  GPU Memory Phase (sparse activity)                             │
│    → Few SMs active, waiting for data  → V1 (PriorityQueue)    │
│    → Fast-forward + dynamic scheduling                          │
│                                                                  │
│  Large-Scale NoC Simulation                                     │
│    → 1000s of router components        → V3 (PrebuiltTaskList) │
│    → Cache locality critical                                    │
│                                                                  │
│  Multi-Socket NUMA Server                                       │
│    → 64+ cores, memory locality        → V6 (LocalTaskQueue)   │
│    → Minimize cross-socket traffic                              │
│                                                                  │
│  Mixed/Unknown Workload                                         │
│    → Variable activity patterns        → V1 (default)          │
│    → Best general-purpose balance                               │
└─────────────────────────────────────────────────────────────────┘
```

**Custom ThreadManager Development:**

ACALSim's interface-based design allows users to implement custom scheduling strategies:

```cpp
// User can implement custom ThreadManager for specific needs
template<typename T>
class CustomThreadManager : public ThreadManagerBase<T> {
    // Custom scheduling logic optimized for specific workload
    void scheduleTask(SimBase* task) override {
        // Application-specific scheduling policy
        // e.g., affinity-based, priority-based, deadline-aware
    }
};

// Register with SimTop
SimTop<CustomThreadManager, CustomTaskManager> sim_top;
```

**Example Custom Strategies:**

| Custom Strategy | Use Case | Implementation Approach |
|-----------------|----------|------------------------|
| **Affinity-Based** | NUMA systems | Pin components to specific cores |
| **Deadline-Aware** | Real-time constraints | Priority by earliest deadline |
| **Locality-Optimized** | Cache-sensitive | Group communicating components |
| **Power-Aware** | Energy efficiency | Consolidate work to fewer cores |
| **Hybrid** | Multi-phase workloads | Switch strategies per phase |

**SST vs ACALSim Scheduling Flexibility:**

| Aspect | SST | ACALSim |
|--------|-----|---------|
| Scheduling strategies | 1 (fixed) | 4+ built-in + custom |
| User customization | Limited (partition hints) | Full (implement interface) |
| Runtime adaptation | None | Can switch managers |
| Workload-specific tuning | Manual partitioning | Select optimal variant |
| NUMA awareness | None | V6 LocalTaskQueue |
| Cache optimization | None | V3 PrebuiltTaskList |

This pluggable architecture means ACALSim can be optimized for specific simulation scenarios, while SST's one-size-fits-all approach may leave performance on the table for specialized workloads.

**Practical Recommendations:**

```
┌─────────────────────────────────────────────────────────────────┐
│                 Thread Count Selection Guide                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Component Count        Recommended Threads    Framework        │
│  ─────────────────────────────────────────────────────────────  │
│  < 50 components        2-4 threads            Either           │
│  50-200 components      4-8 threads            ACALSim          │
│  200-500 components     8-16 threads           ACALSim          │
│  > 500 components       16-32 threads          ACALSim          │
│                                                                  │
│  Note: Thread count should not exceed 2× physical cores        │
│        ACALSim benefits more from additional threads due to    │
│        work stealing and reduced total work (fast-forward)     │
└─────────────────────────────────────────────────────────────────┘
```

**GPU Simulation Example (108 SMs):**

```
Simulating A100 GPU (108 SMs + memory controllers + NoC):

SST with --num-threads=16:
  Components per thread: ~8
  Per-cycle overhead: 8 × clockTick() × barrier_sync
  Load imbalance: Some threads have more active SMs
  Scaling efficiency: ~35%
  Effective speedup: ~5.6×

ACALSim with 16 threads (V3 TaskManager):
  Active components per cycle: ~30 (during memory phase)
  Work per thread: ~2 components (dynamic)
  Fast-forward benefit: Skip 70% of cycles
  Scaling efficiency: ~75%
  Effective speedup: ~12× (plus 3× from fast-forward = ~36× total)
```

#### 3.5.9 Summary: Why ACALSim for HPC Chip Simulation

| Requirement | SST `--num-threads` | ACALSim |
|-------------|---------------------|---------|
| Sparse activation (GPU memory phases) | ❌ Still executes idle cycles | ✅ Fast-forward skips |
| Variable component workload | ❌ Static partition, no balancing | ✅ Dynamic work stealing |
| High component count (100+ SMs) | ⚠️ Load imbalance grows | ✅ Scales with active count |
| Burst activity patterns | ❌ No optimization | ✅ Jump between bursts |
| Mixed compute/memory phases | ❌ Wastes cycles in memory phases | ✅ Event-driven execution |
| High-bandwidth data movement | ❌ O(data_size) copy overhead | ✅ O(1) pointer passing |
| Large tensor simulations | ❌ Memory bloat from Event copies | ✅ Single shared copy |
| NoC traffic simulation | ❌ Event alloc/copy per flit | ✅ Lightweight packets |

**Conclusion**: While SST's `--num-threads` provides basic parallelism, it remains fundamentally clock-driven with static partitioning and event-based data passing. For HPC chip simulation with sparse, variable workloads and intensive data movement, ACALSim's event-driven, dynamically-scheduled, shared-memory approach delivers substantially better performance—often **10× or more** compared to SST threading alone, with the gap widening for data-intensive simulations.

---

## 4. ACALSim Design

### 4.1 System Overview

ACALSim consists of four primary abstractions:

- **SimBase**: Base class for all simulated components
- **SimTop**: Top-level simulation controller
- **SimChannel**: Thread-safe inter-component communication
- **SimPort**: Hardware-accurate port with arbitration

### 4.2 Two-Phase Execution Model

ACALSim divides each simulation iteration into two phases:

**Phase 1: Parallel Execution**
- Worker threads execute assigned SimBase components
- Each component processes expired events from its local queue
- Components may generate new events and outbound packets
- No inter-component communication during execution

**Phase 2: Synchronization**
- Control thread performs global synchronization
- SimChannel queues are swapped (ping-pong mechanism)
- Next simulation tick is computed via fast-forward
- Component states are updated for next iteration

**Theorem 1 (Determinism).** The two-phase execution model produces deterministic results regardless of thread scheduling.

*Proof Sketch*: Phase 1 executes components in isolation with no shared mutable state. All inter-component effects are buffered. Phase 2 applies buffered effects in a deterministic order controlled by the single control thread. □

### 4.3 Fast-Forward Mechanism

Let nextᵢ(t) denote the next active tick for component i at current time t:

```
nextᵢ(t) = min({tₑ | (tₑ, e) ∈ Qᵢ, tₑ > t} ∪ {∞})
```

The global next tick is:
```
t_next = min_{i ∈ [1,n]} nextᵢ(t)
```

**Algorithm 1: Fast-Forward**
```
function FastForward(t_current):
    t_next ← ∞
    for each component c_i:
        t_next ← min(t_next, c_i.getNextSimTick())
        if c_i.hasInboundPackets():
            t_next ← min(t_next, t_current + 1)
    return t_next
```

**Complexity Analysis**: Fast-forward computation is O(n) where n is the number of components. For sparse workloads where only k ≪ n components are active, the expected skip distance is O(T/k) cycles.

### 4.4 Communication Primitives

#### 4.4.1 SimChannel: Thread-Safe Software Communication

SimChannel employs a ping-pong queue mechanism to eliminate synchronization during Phase 1:

```
┌─────────────────────────────────────────┐
│              SimChannel                  │
│  ┌─────────────┐    ┌─────────────┐     │
│  │ Ping Queue  │    │ Pong Queue  │     │
│  │  (push)     │    │   (pop)     │     │
│  └─────────────┘    └─────────────┘     │
│         ↑                  ↓            │
│      Sender            Receiver         │
│                                         │
│  Phase 2: Swap ping ↔ pong             │
└─────────────────────────────────────────┘
```

**Property 1**: Packets pushed in iteration i are received in iteration i+1, guaranteeing minimum one-cycle latency.

**Property 2**: No locks are required during Phase 1 execution as push and pop operate on disjoint queues.

#### 4.4.2 SimPort: Hardware-Accurate Communication

SimPort models hardware communication with:
- **MasterPort**: Single-entry output buffer
- **SlavePort**: Configurable queue with arbitration
- **Backpressure**: Callback mechanism when port becomes available

```
┌─────────┐     ┌─────────┐     ┌─────────────────┐
│Master₁  │────→│         │     │                 │
└─────────┘     │         │     │    SlavePort    │
┌─────────┐     │Arbitrate│────→│   (ReqQueue)    │
│Master₂  │────→│         │     │                 │
└─────────┘     │         │     └─────────────────┘
┌─────────┐     │         │
│Master₃  │────→│         │
└─────────┘     └─────────┘
```

**Arbitration Policy**: Configurable (default: round-robin). The winner's packet moves to SlavePort's request queue; losers receive backpressure notification.

### 4.5 Thread Management

ACALSim provides multiple ThreadManager implementations optimized for different workload characteristics:

| Variant | Algorithm | Time Complexity | Best For |
|---------|-----------|-----------------|----------|
| V1 (PriorityQueue) | Min-heap scheduling | O(log n) per task | Sparse activation |
| V2 (Barrier) | C++20 barrier sync | O(1) amortized | Dense workloads |
| V3 (PrebuiltTaskList) | Pre-allocated lists | O(1) | Memory-intensive |
| V6 (LocalTaskQueue) | Per-thread queues | O(1) | High contention |

**Task Activation Condition**: A component cᵢ is scheduled for execution if:
```
active(cᵢ, t) = (nextᵢ(t) = t) ∨ hasInboundPackets(cᵢ)
```

---

## 5. Hybrid SST+ACALSim Architecture

### 5.1 Design Rationale

Neither SST nor ACALSim alone addresses all simulation requirements:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Requirement Analysis                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Single-Node Efficiency        Multi-Node Scalability           │
│  ┌───────────────────┐        ┌───────────────────┐            │
│  │ • Multi-threading │        │ • MPI distribution │            │
│  │ • Fast-forward    │        │ • Cluster scale    │            │
│  │ • Zero-copy comm  │        │ • Component reuse  │            │
│  └─────────┬─────────┘        └─────────┬─────────┘            │
│            │                            │                        │
│            │    ┌──────────────────┐    │                        │
│            └───►│  Hybrid Solution │◄───┘                        │
│                 │  SST + ACALSim   │                             │
│                 └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Hybrid Architecture

The hybrid approach embeds ACALSim simulations as SST components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Hybrid SST+ACALSim System                        │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                        SST Framework                             ││
│  │                                                                  ││
│  │   MPI Rank 0                          MPI Rank 1                ││
│  │  ┌────────────────────────┐         ┌────────────────────────┐  ││
│  │  │    SST Component       │   MPI   │    SST Component       │  ││
│  │  │  ┌──────────────────┐  │◄───────►│  ┌──────────────────┐  │  ││
│  │  │  │    ACALSim       │  │  Link   │  │    ACALSim       │  │  ││
│  │  │  │   ┌──────────┐   │  │         │  │   ┌──────────┐   │  │  ││
│  │  │  │   │ Thread 0 │   │  │         │  │   │ Thread 0 │   │  │  ││
│  │  │  │   │ Thread 1 │   │  │         │  │   │ Thread 1 │   │  │  ││
│  │  │  │   │ Thread 2 │   │  │         │  │   │ Thread 2 │   │  │  ││
│  │  │  │   │ Thread 3 │   │  │         │  │   │ Thread 3 │   │  │  ││
│  │  │  │   └──────────┘   │  │         │  │   └──────────┘   │  │  ││
│  │  │  │   GPU 0 (108 SMs)│  │         │  │   GPU 1 (108 SMs)│  │  ││
│  │  │  └──────────────────┘  │         │  └──────────────────┘  │  ││
│  │  │                        │         │                        │  ││
│  │  │  ┌──────────────────┐  │         │  ┌──────────────────┐  │  ││
│  │  │  │ SST memHierarchy │  │         │  │ SST memHierarchy │  │  ││
│  │  │  └──────────────────┘  │         │  └──────────────────┘  │  ││
│  │  └────────────────────────┘         └────────────────────────┘  ││
│  │                                                                  ││
│  │  ┌──────────────────────────────────────────────────────────┐   ││
│  │  │                    SST Merlin Network                     │   ││
│  │  │              (Inter-GPU Communication)                    │   ││
│  │  └──────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Integration Mechanism

**Algorithm 2: SST-ACALSim Clock Handler**

```cpp
class ACALSimWrapper : public SST::Component {
    std::shared_ptr<SimTop> sim_top_;
    bool ready_to_terminate_ = false;

    bool clockTick(Cycle_t cycle) {
        // Check termination from previous cycle
        if (ready_to_terminate_) {
            return true;  // Stop SST clock
        }

        // === ACALSim Phase 1: Parallel Execution ===
        sim_top_->startPhase1();   // Wake worker threads
        sim_top_->finishPhase1();  // Barrier: all threads complete

        // === ACALSim Phase 2: Synchronization ===
        sim_top_->startPhase2();

        // Process SST inbound events → ACALSim packets
        while (SST::Event* ev = link_->recv()) {
            sim_top_->injectExternalPacket(ev);
        }

        // Fast-forward computation
        Tick next_tick = sim_top_->computeNextActiveTick();
        sim_top_->fastForwardTo(next_tick);

        // Extract ACALSim outbound packets → SST events
        while (Packet* pkt = sim_top_->getOutboundPacket()) {
            link_->send(new SSTEvent(pkt), latency_);
        }

        // Channel toggle
        SimChannelGlobal::toggleQueues();

        // Check completion
        if (sim_top_->isComplete()) {
            sim_top_->issueExitEvent();
            ready_to_terminate_ = true;
            primaryComponentOKToEndSim();
        }

        sim_top_->finishPhase2();
        return false;  // Continue SST clock
    }
};
```

### 5.4 How Hybrid Addresses Limitations

**Table 3: Limitation Resolution Matrix**

| SST Limitation | Hybrid Solution | Mechanism |
|----------------|-----------------|-----------|
| Sequential intra-rank | ACALSim parallel execution | Thread pool within SST component |
| No fast-forward | ACALSim event-driven | Skip idle cycles within component |
| Point-to-point links | SimPort arbitration | Many-to-one within component |
| MPI overhead for fine-grain | Zero-copy within component | Only coarse-grain crosses MPI |

| ACALSim Limitation | Hybrid Solution | Mechanism |
|--------------------|-----------------|-----------|
| Single-node bound | SST MPI distribution | Multiple ACALSim instances across ranks |
| No ecosystem | SST component reuse | memHierarchy, Merlin available |
| No distributed support | SST Links | Standard SST inter-component comm |
| Configuration complexity | SST Python | Configure via SST scripts |

### 5.5 Communication Model

The hybrid system employs a hierarchical communication strategy:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Communication Hierarchy                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: Intra-SimBase (within single simulator)               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • Direct function calls                                    ││
│  │  • No synchronization needed                                ││
│  │  • Latency: ~1 ns                                           ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  Level 2: Inter-SimBase, Same ACALSim (SimChannel/SimPort)      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • Pointer-based packet passing                             ││
│  │  • Ping-pong queue (lock-free in Phase 1)                   ││
│  │  • Latency: ~10 ns                                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  Level 3: Inter-ACALSim, Same MPI Rank (SST Link, local)        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • SST Event passing                                        ││
│  │  • Memory copy (not serialize)                              ││
│  │  • Latency: ~100 ns                                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  Level 4: Inter-ACALSim, Cross MPI Rank (SST Link + MPI)        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • SST Event serialization                                  ││
│  │  • MPI message passing                                      ││
│  │  • Latency: ~1-10 μs                                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 Scalability Analysis

**Theorem 2 (Hybrid Scalability).** The hybrid SST+ACALSim system achieves multiplicative speedup combining both frameworks' parallelism.

*Formalization*: Let:
- R = number of MPI ranks
- T = threads per ACALSim instance
- S_MPI(R) = SST's MPI speedup function
- S_thread(T) = ACALSim's thread speedup function
- f = fraction of time in intra-component work

Then the hybrid speedup is:
```
S_hybrid(R, T) = 1 / (f/S_thread(T) + (1-f)/S_MPI(R))
```

For typical GPU simulation where f ≈ 0.85:

| Configuration | Expected Speedup |
|---------------|------------------|
| R=1, T=8 | 6.2× |
| R=4, T=8 | 18.5× |
| R=8, T=8 | 32.1× |
| R=16, T=8 | 54.8× |

### 5.7 Use Case: Multi-GPU Data Center Simulation

**Scenario**: Simulate 8-GPU DGX-like system running distributed LLM training

```
┌─────────────────────────────────────────────────────────────────┐
│                    DGX-8 Simulation Setup                        │
│                                                                  │
│  MPI Rank 0          MPI Rank 1          ...    MPI Rank 7      │
│  ┌──────────┐       ┌──────────┐              ┌──────────┐     │
│  │ ACALSim  │       │ ACALSim  │              │ ACALSim  │     │
│  │  GPU 0   │       │  GPU 1   │              │  GPU 7   │     │
│  │ 108 SMs  │       │ 108 SMs  │              │ 108 SMs  │     │
│  │ 8 threads│       │ 8 threads│              │ 8 threads│     │
│  └────┬─────┘       └────┬─────┘              └────┬─────┘     │
│       │                  │                         │            │
│       └──────────────────┴─────────────────────────┘            │
│                          │                                       │
│                    SST Merlin Network                            │
│                    (NVLink Topology)                             │
└─────────────────────────────────────────────────────────────────┘

Total: 864 SMs, 64 threads parallel, 8 MPI ranks
Estimated speedup: 32× over single-threaded SST
```

---

## 6. Implementation

### 6.1 Core APIs

**SimBase Interface**:
```cpp
class SimBase {
    virtual void init();           // Initialization
    virtual void step();           // Per-cycle logic
    virtual void cleanup();        // Finalization

    void scheduleEvent(Event* e, Tick t);
    Tick getNextSimTick() const;
};
```

**Communication APIs**:
```cpp
// SimChannel: software communication
void sendPacketViaChannel(string port, Tick local, Tick remote, Packet* p);
void pushToMasterChannelPort(string port, Packet* p);

// SimPort: hardware-accurate communication
bool MasterPort::push(Packet* p);
Packet* SlavePort::pop();
virtual void masterPortRetry(string port);  // Backpressure callback
```

### 6.2 Memory Optimization

ACALSim employs object pooling through RecycleContainer to minimize allocation overhead:

```cpp
template<typename T>
class RecycleContainer {
    T* acquire();           // Get from pool or allocate
    void release(T* obj);   // Return to pool
};
```

### 6.3 SST Integration

ACALSim components can be wrapped as SST components. See `docs/sst-integration/integration-guide.md` for detailed implementation guide.

---

## 7. Evaluation

### 7.1 Experimental Setup

**Platform**:
- CPU: AMD EPYC 7763 (64 cores, 128 threads)
- Memory: 512 GB DDR4
- OS: Ubuntu 22.04, Linux 5.15

**Benchmarks**:
- GPU Configuration: A100-like (108 SMs, 80 GB HBM2)
- Workload: LLaMA-2 7B inference, batch size 1-8

### 7.2 Single-Node Scalability

**Table 4: Simulation Throughput vs. Thread Count**

| Threads | Throughput (cycles/sec) | Speedup | Efficiency |
|---------|------------------------|---------|------------|
| 1 | 1,000 | 1.0× | 100% |
| 2 | 1,850 | 1.85× | 92.5% |
| 4 | 3,400 | 3.4× | 85% |
| 8 | 6,200 | 6.2× | 77.5% |
| 16 | 10,500 | 10.5× | 65.6% |

### 7.3 Fast-Forward Effectiveness

**Table 5: Fast-Forward Impact by Workload Phase**

| Phase | Active SMs | Skip Ratio | Speedup |
|-------|-----------|------------|---------|
| Attention (compute) | 108/108 | 0% | 1.0× |
| Memory Load | 12/108 | 89% | 8.2× |
| Softmax | 32/108 | 70% | 3.1× |
| Overall | Variable | 45% | 2.8× |

### 7.4 Communication Overhead

**Table 6: SimChannel vs. Alternative Approaches**

| Mechanism | Latency (ns) | Throughput (M packets/sec) |
|-----------|-------------|---------------------------|
| SimChannel (ping-pong) | 12 | 85 |
| Mutex-protected queue | 45 | 22 |
| Lock-free queue | 28 | 36 |
| MPI (same node) | 850 | 1.2 |

### 7.5 Context Switch Analysis

**Table 7: Context Switches by ThreadManager Variant**

| Variant | Voluntary CS | Involuntary CS | Wall Time |
|---------|-------------|----------------|-----------|
| V1 (baseline) | 302 | 5 | 1.00× |
| V1 (optimized) | 193 | 7 | 0.94× |
| V3 (PrebuiltTaskList) | 156 | 8 | 0.91× |
| V6 (LocalTaskQueue) | 178 | 6 | 0.93× |

### 7.6 Comparison with Existing Frameworks

**Table 8: Framework Comparison (108 SM GPU, 10K cycles)**

| Framework | Wall Time | CPU Util. | Memory |
|-----------|-----------|-----------|--------|
| GPGPU-Sim | 45.2s | 12% | 8.2 GB |
| SST (1 rank) | 38.1s | 12% | 6.8 GB |
| ACALSim (8 threads) | 4.8s | 85% | 4.1 GB |

---

## 8. Discussion

### 8.1 Scalability Analysis

ACALSim's scalability is bounded by:

1. **Amdahl's Law**: Sequential Phase 2 limits parallel speedup
2. **Memory Bandwidth**: Shared memory contention at high thread counts
3. **Synchronization Overhead**: Two-phase barrier cost grows with threads

For n components, p threads, and Phase 2 fraction f:
```
Speedup ≤ 1 / (f + (1-f)/p)
```

Empirically, f ≈ 0.08 for GPU simulation, yielding theoretical maximum speedup of 12.5×.

### 8.2 Accuracy Considerations

ACALSim maintains cycle-accuracy through:
- Deterministic two-phase execution
- Minimum one-cycle communication latency
- Preserved event ordering within each component

The fast-forward mechanism does not affect accuracy as it only advances time when no component has pending work.

### 8.3 Limitations

1. **Single-Node Bound**: Without SST integration, limited to single machine
2. **Memory Capacity**: All component state must fit in shared memory
3. **Workload Dependency**: Fast-forward benefits depend on activation sparsity

### 8.4 Framework Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                   Framework Selection Decision Tree              │
│                                                                  │
│                    ┌─────────────────┐                          │
│                    │  System Scale?  │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│              ┌──────────────┴──────────────┐                    │
│              ▼                             ▼                    │
│     Single Machine               Multi-Machine                  │
│              │                             │                    │
│              ▼                             ▼                    │
│  ┌───────────────────┐        ┌───────────────────┐            │
│  │ Many components?  │        │    Use Hybrid     │            │
│  └─────────┬─────────┘        │   SST + ACALSim   │            │
│            │                  └───────────────────┘            │
│     ┌──────┴──────┐                                            │
│     ▼             ▼                                            │
│   < 100        > 100                                           │
│     │             │                                            │
│     ▼             ▼                                            │
│ ┌───────┐   ┌─────────┐                                        │
│ │  SST  │   │ ACALSim │                                        │
│ │ alone │   │  alone  │                                        │
│ └───────┘   └─────────┘                                        │
│                                                                  │
│  Legend:                                                         │
│  • SST alone: Leverage ecosystem, simple config                 │
│  • ACALSim alone: Maximum single-node performance               │
│  • Hybrid: Best of both for large-scale HPC simulation          │
└─────────────────────────────────────────────────────────────────┘
```

### 8.5 Future Work

- **Adaptive ThreadManager Selection**: Runtime switching based on workload
- **Hierarchical Simulation**: Nested ACALSim instances for chiplet modeling
- **Approximate Simulation**: Trading accuracy for speed in early design phases

---

## 9. Conclusion

We presented ACALSim, an event-driven multi-threaded simulation framework designed for HPC chip simulation. Through its two-phase execution model, fast-forward mechanism, and hardware-accurate communication primitives, ACALSim achieves significant speedup over existing frameworks while maintaining cycle-accuracy. The integration with SST enables scaling from single-node multi-threading to multi-node distributed simulation.

ACALSim addresses a critical need in HPC chip design: the ability to simulate systems with thousands of components efficiently on commodity hardware. As chip complexity continues to grow, frameworks like ACALSim will be essential for enabling architectural exploration and design optimization.

**Key Takeaways:**
- ACALSim provides 8-12× speedup over sequential simulation via multi-threading
- Event-driven fast-forward provides additional 2-10× speedup for sparse workloads
- SimChannel and SimPort provide efficient, hardware-accurate communication
- Hybrid SST+ACALSim combines single-node efficiency with multi-node scalability

---

## 10. References

[1] A. F. Rodrigues et al., "The Structural Simulation Toolkit," ACM SIGMETRICS, 2011.

[2] N. Binkert et al., "The gem5 Simulator," ACM SIGARCH Computer Architecture News, 2011.

[3] A. Bakhoda et al., "Analyzing CUDA Workloads Using a Detailed GPU Simulator," ISPASS, 2009.

[4] K. M. Chandy and J. Misra, "Distributed Simulation: A Case Study in Design and Verification of Distributed Programs," IEEE TSE, 1979.

[5] D. R. Jefferson, "Virtual Time," ACM TOPLAS, 1985.

[6] C. D. Carothers et al., "ROSS: A High-Performance, Low-Memory, Modular Time Warp System," PADS, 2002.

[7] P. S. Magnusson et al., "Simics: A Full System Simulation Platform," IEEE Computer, 2002.

[8] R. Ubal et al., "Multi2Sim: A Simulation Framework for CPU-GPU Computing," PACT, 2012.

[9] S. Collange et al., "Barra: A Parallel Functional Simulator for GPGPU," MASCOTS, 2010.

[10] Y. Sun et al., "MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization," ISCA, 2019.

[11] D. Sanchez and C. Kozyrakis, "ZSim: Fast and Accurate Microarchitectural Simulation of Thousand-Core Systems," ISCA, 2013.

[12] T. E. Carlson et al., "Sniper: Exploring the Level of Abstraction for Scalable and Accurate Parallel Multi-Core Simulations," SC, 2011.

---

**Copyright 2023-2026 Playlab/ACAL**
Licensed under the Apache License, Version 2.0
