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

# HSA Protocol Implementation for QEMU-ACALSim-SST

Implementation of the HSA (Heterogeneous System Architecture) protocol for job launch modeling between QEMU runtime and SST device components.

## Overview

This project extends the qemu-acalsim-sst-baremetal framework with HSA protocol support, enabling realistic modeling of heterogeneous compute workloads using industry-standard HSA specifications.

### Key Features

- **HSA AQL Packets**: Architected Queuing Language for kernel dispatch
- **User Mode Queues**: Shared memory queue abstraction
- **Doorbells**: Notification mechanism for new work
- **Signals**: Completion notification and synchronization
- **Cycle-Accurate Modeling**: Realistic kernel execution latency
- **Multi-Agent Support**: Host (CPU) and Compute (GPU/Accelerator) agents

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     SST Simulation                           │
│                                                               │
│  ┌────────────────┐                  ┌───────────────────┐   │
│  │  HSA Host      │    AQL Packets   │  HSA Compute      │   │
│  │  Component     │─────────────────►│  Component        │   │
│  │  (CPU Agent)   │                  │  (GPU/Accel)      │   │
│  │                │◄─────────────────│                   │   │
│  └────────────────┘  Signals         └───────────────────┘   │
│         │                                      │              │
│         │ Doorbell                             │              │
│         └──────────────────────────────────────┘              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### HSA Protocol Components

1. **AQL Packets (Kernel Dispatch)**:
   - Header: packet type, fence scope
   - Work dimensions: grid size, workgroup size
   - Kernel info: code address, argument address
   - Memory requirements: private/group segment sizes
   - Completion signal: for notification

2. **Signals**:
   - Unique handle for each kernel dispatch
   - Value-based completion (0 = success)
   - Atomic operations support (STORE, ADD, SUB, etc.)

3. **Doorbells**:
   - Queue notifications
   - Ring to indicate new work available
   - Optional optimization for queue management

4. **User Mode Queues**:
   - Modeled as std::queue in compute agent
   - Configurable depth
   - Back-pressure handling

## File Organization

### Shared HSA Framework Components

HSA components are part of the ACALSim framework and shared by all APPs via framework-level directories:

```
# Framework-level shared sources (at project root: ../../..)
libs/HSA/
├── HSAHostComponent.cc          # Host agent implementation
├── HSAComputeComponent.cc       # Compute agent implementation
└── Makefile                      # (For reference only)

include/HSA/
├── HSAEvents.hh                  # HSA event type definitions
├── HSAHostComponent.hh           # Host component header
└── HSAComputeComponent.hh        # Compute component header
```

**Build Integration**: Each APP's `acalsim-device/Makefile` references these shared framework sources and compiles them into `libacalsim.so` alongside APP-specific device components. This ensures:
- HSA code is maintained in one shared location
- All APPs can use the same HSA implementation
- No separate HSA library needed - everything in libacalsim.so

### APP-Specific Files (src/qemu-acalsim-sst-baremetal-HSA/)

```
qemu-acalsim-sst-baremetal-HSA/
├── README_HSA.md                 # This file
├── hsa_basic_test.py             # Basic HSA protocol test
├── hsa_scalability_test.py       # Multi-agent scalability test
└── (other files from baremetal)  # Inherited from base project
```

## HSA Event Types

### 1. HSAAQLPacketEvent

Kernel dispatch packet (HSA 1.2 spec section 2.8):

```cpp
class HSAAQLPacketEvent : public SST::Event {
    uint16_t header;               // Packet type + fence scope
    uint16_t setup;                // Dimensions
    uint32_t workgroup_size_x/y/z; // Work-group dimensions
    uint32_t grid_size_x/y/z;      // Grid dimensions
    uint64_t kernel_object;        // Kernel code address
    uint64_t kernarg_address;      // Kernel arguments address
    uint32_t private_segment_size; // Private memory per work-item
    uint32_t group_segment_size;   // Group memory per work-group
    uint64_t completion_signal;    // Signal handle
    uint32_t dispatch_id;          // Dispatch identifier
};
```

### 2. HSASignalEvent

Completion and synchronization signals:

```cpp
class HSASignalEvent : public SST::Event {
    uint64_t signal_handle;        // Signal identifier
    int64_t signal_value;          // Signal value (0 = complete)
    Operation operation;           // WAIT, STORE, ADD, SUB, etc.
    uint32_t dispatch_id;          // Associated dispatch
};
```

### 3. HSADoorbellEvent

Queue notification events:

```cpp
class HSADoorbellEvent : public SST::Event {
    uint32_t queue_id;             // Queue identifier
    uint64_t doorbell_value;       // Write index value
};
```

### 4. HSAMemoryEvent

Memory operations (optional):

```cpp
class HSAMemoryEvent : public SST::Event {
    Type type;                     // READ, WRITE, COPY
    uint64_t address;              // Memory address
    uint64_t size;                 // Size in bytes
    bool is_complete;              // Completion status
};
```

## Components

### HSAHostComponent (Host/CPU Agent)

Submits kernels to compute agents using AQL packets.

**Parameters**:
- `clock`: Clock frequency (default: "1GHz")
- `verbose`: Verbosity level 0-3 (default: 0)
- `num_dispatches`: Number of kernels to submit (default: 10)
- `workgroup_size_x/y/z`: Work-group dimensions (default: 256/1/1)
- `grid_size_x/y/z`: Grid dimensions (default: 1024/1/1)
- `dispatch_interval`: Cycles between dispatches (default: 1000)

**Ports**:
- `aql_port`: Send AQL packets to compute agent
- `signal_port`: Receive completion signals
- `doorbell_port`: Send doorbell notifications (optional)

**Statistics**:
- `dispatches_submitted`: Total kernels submitted
- `dispatches_completed`: Total kernels completed
- `avg_latency`: Average kernel latency (ns)

### HSAComputeComponent (GPU/Accelerator Agent)

Executes kernels received via AQL packets.

**Parameters**:
- `clock`: Clock frequency (default: "1GHz")
- `verbose`: Verbosity level 0-3 (default: 0)
- `queue_depth`: Maximum AQL queue depth (default: 256)
- `cycles_per_workitem`: Simulated cycles per work-item (default: 100)
- `kernel_launch_overhead`: Launch overhead cycles (default: 1000)
- `memory_latency`: Memory access latency cycles (default: 100)

**Ports**:
- `aql_port`: Receive AQL packets from host
- `signal_port`: Send completion signals
- `doorbell_port`: Receive doorbell notifications (optional)

**Statistics**:
- `kernels_executed`: Total kernels executed
- `total_workitems`: Total work-items processed
- `avg_kernel_latency`: Average execution latency (cycles)
- `queue_occupancy`: Average queue depth

## Build Instructions

### 1. Build libacalsim.so with HSA Components

HSA components are automatically included when building the device library:

```bash
cd acalsim-device
make clean && make && make install
```

This builds `libacalsim.so` containing all device components (QEMUDevice, ComputeDevice, HSAHost, HSACompute) using shared framework sources from `../../libs/HSA/`.

### 2. Verify Installation

```bash
sst-info acalsim
```

Expected output showing all components including HSA:
```
Component: QEMUDevice
Component: ComputeDevice
Component: HSAHost
Component: HSACompute
```

## Running Tests

### Basic HSA Protocol Test

Tests basic host-compute communication:

```bash
cd src/qemu-acalsim-sst-baremetal-HSA
sst hsa_basic_test.py
```

**Expected Output**:
```
HSA Protocol Basic Test
============================================================
Configuration:
  Host: 2GHz, 5 dispatches
  Compute: 1GHz, 50 cycles/workitem
  Workgroup: 256x1x1
  Grid: 1024x1x1
  Total workitems per kernel: 1024
============================================================
HSAHost: Submitted dispatch 0: signal=0x1000, workitems=1024
HSACompute: Received AQL packet: dispatch=0
HSACompute: Starting kernel: dispatch=0, estimated_latency=52600 cycles
HSACompute: Kernel completed: dispatch=0, latency=52600 cycles
HSAHost: Dispatch 0 completed: latency=52600 ns
...
HSAHost: All 5 dispatches completed
```

### Scalability Test

Test with multiple compute agents:

```bash
sst hsa_scalability_test.py
```

## Performance Modeling

### Kernel Execution Latency

The compute component models kernel execution latency as:

```
total_latency = launch_overhead + compute_latency + memory_latency

where:
  compute_latency = workitems × cycles_per_workitem
  memory_latency = workgroups × 2 × memory_latency_cycles
```

**Example**:
- Grid: 1024 workitems (4 workgroups of 256)
- cycles_per_workitem: 100
- launch_overhead: 1000 cycles
- memory_latency: 100 cycles

```
compute_latency = 1024 × 100 = 102,400 cycles
memory_latency = 4 × 2 × 100 = 800 cycles
total_latency = 1000 + 102,400 + 800 = 104,200 cycles
```

## Integration with QEMU

The HSA protocol can be integrated with QEMU-based simulations:

### Architecture with QEMU

```
┌──────────────────────────────────────────────────────────────┐
│                     SST Simulation                           │
│                                                               │
│  ┌────────────────┐     AQL      ┌───────────────────┐       │
│  │  QEMU Binary   │─────────────►│  HSA Compute      │       │
│  │  Component     │              │  Agent            │       │
│  │                │◄─────────────│                   │       │
│  └────────┬───────┘   Signals    └───────────────────┘       │
│           │                                                    │
│           │ MMIO (HSA Queue Mgmt)                             │
│           ▼                                                    │
│  ┌──────────────────┐                                         │
│  │ RISC-V CPU       │                                         │
│  │ (in QEMU)        │                                         │
│  └──────────────────┘                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### RISC-V Program Example

```c
// HSA queue management via MMIO
#define HSA_QUEUE_BASE   0x10400000
#define HSA_WRITE_INDEX  (*(volatile uint64_t *)(HSA_QUEUE_BASE + 0x00))
#define HSA_READ_INDEX   (*(volatile uint64_t *)(HSA_QUEUE_BASE + 0x08))
#define HSA_DOORBELL     (*(volatile uint32_t *)(HSA_QUEUE_BASE + 0x10))

// Submit kernel via AQL packet
void submit_kernel() {
    struct AQLPacket {
        uint16_t header;
        uint16_t setup;
        uint32_t workgroup_size[3];
        uint32_t grid_size[3];
        uint64_t kernel_object;
        uint64_t kernarg_address;
        uint64_t completion_signal;
    } packet;

    // Fill packet fields
    packet.header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << 8);
    packet.workgroup_size[0] = 256;
    packet.grid_size[0] = 1024;
    packet.kernel_object = 0x100000;
    packet.completion_signal = 0x2000;

    // Write packet to queue (via MMIO)
    write_aql_packet(&packet);

    // Ring doorbell
    HSA_DOORBELL = 1;

    // Wait for completion signal
    while (read_signal(0x2000) != 0);
}
```

## Reference

- **HSA Specification**: [HSA System Architecture 1.2](https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf)
- **AQL Packets**: Section 2.8 (Kernel Dispatch Packet)
- **Signals**: Section 2.11 (Signals)
- **User Mode Queues**: Section 2.7 (Queues)

## Related Projects

- **qemu-acalsim-sst-baremetal**: Base bare-metal MMIO project
- **qemu-acalsim-sst-linux**: Linux-based variant

## License

Copyright 2023-2026 Playlab/ACAL

Licensed under the Apache License, Version 2.0.

---

**Status**: Complete HSA protocol implementation
**Last Updated**: 2025-11-10
