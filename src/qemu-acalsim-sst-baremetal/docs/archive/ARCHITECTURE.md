<!--
Copyright 2023-2025 Playlab/ACAL

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

# QEMU-ACALSim Distributed SST Architecture

## System Overview

This document describes the architecture of a distributed simulation system that combines QEMU (full-system emulation) with ACALSim (cycle-accurate modeling) using SST's distributed simulation framework.

## Key Design Principles

1. **Process Isolation**: QEMU and ACALSim run in separate OS processes
2. **IPC Communication**: Processes communicate via SST's MPI-based messaging
3. **Loose Coupling**: Components interact only through well-defined SST interfaces
4. **Scalability**: Design supports multi-node distributed execution

## Component Architecture

### 1. QEMU SST Component

The QEMU component wraps a QEMU RISC-V system emulator as an SST component.

```
┌─────────────────────────────────────────┐
│        QEMU SST Component               │
│  ┌───────────────────────────────────┐  │
│  │   QEMU RISC-V System Emulator     │  │
│  │   ┌────────────┐  ┌─────────────┐ │  │
│  │   │   CPU      │  │  RAM/ROM    │ │  │
│  │   └────────────┘  └─────────────┘ │  │
│  │   ┌────────────┐  ┌─────────────┐ │  │
│  │   │  uboot     │  │  Devices    │ │  │
│  │   └────────────┘  └─────────────┘ │  │
│  └───────────────────────────────────┘  │
│           │                              │
│  ┌────────▼──────────┐                   │
│  │  Memory-Mapped    │                   │
│  │  Device Region    │                   │
│  │  0x10000000 -     │                   │
│  │  0x10000FFF       │                   │
│  └────────┬──────────┘                   │
│           │                              │
│  ┌────────▼──────────┐                   │
│  │  SST Link Handler │                   │
│  └────────┬──────────┘                   │
└───────────┼──────────────────────────────┘
            │ SST Event (IPC)
            ▼
```

#### Key Features:
- Intercepts memory accesses to device region (0x10000000-0x10000FFF)
- Converts QEMU memory transactions to SST events
- Handles clock synchronization between QEMU and SST
- Manages QEMU execution in SST clock cycles

### 2. ACALSim Device Component

The ACALSim component models a cycle-accurate hardware device.

```
┌─────────────────────────────────────────┐
│    ACALSim Device SST Component         │
│  ┌───────────────────────────────────┐  │
│  │  SST Link Handler (Receive)       │  │
│  └────────┬──────────────────────────┘  │
│           │                              │
│  ┌────────▼──────────┐                   │
│  │  Request Queue    │                   │
│  └────────┬──────────┘                   │
│           │                              │
│  ┌────────▼──────────────────────────┐   │
│  │   ACALSim Device Model            │   │
│  │   ┌──────────┐  ┌──────────────┐  │   │
│  │   │ Registers│  │ State Machine│  │   │
│  │   └──────────┘  └──────────────┘  │   │
│  │   ┌──────────┐  ┌──────────────┐  │   │
│  │   │  Memory  │  │   Logic      │  │   │
│  │   └──────────┘  └──────────────┘  │   │
│  └────────┬──────────────────────────┘   │
│           │                              │
│  ┌────────▼──────────┐                   │
│  │  Response Queue   │                   │
│  └────────┬──────────┘                   │
│           │                              │
│  ┌────────▼──────────────────────────┐   │
│  │  SST Link Handler (Send)          │   │
│  └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### Key Features:
- Receives load/store requests via SST Link
- Models device behavior using ACALSim event-driven simulation
- Cycle-accurate timing for device operations
- Sends responses back to QEMU via SST Link

## Communication Protocol

### SST Event Types

```cpp
// Memory transaction event
class MemoryTransactionEvent : public SST::Event {
public:
    enum Type { LOAD, STORE };

    Type        type;       // Load or Store
    uint64_t    address;    // Memory address
    uint32_t    data;       // Data (for stores)
    uint32_t    size;       // Transaction size (1, 2, 4 bytes)
    uint64_t    req_id;     // Request ID for matching
};

// Memory response event
class MemoryResponseEvent : public SST::Event {
public:
    uint64_t    req_id;     // Request ID
    uint32_t    data;       // Response data (for loads)
    bool        success;    // Transaction success
};
```

### Transaction Flow

```
QEMU                    SST Link                ACALSim Device
  │                         │                         │
  │  Store 0xDEADBEEF       │                         │
  │  to 0x10000000          │                         │
  ├─────────────────────────┤                         │
  │  MemoryTransaction      │                         │
  │  (STORE, addr, data)    ├────────────────────────►│
  │                         │                         │ Process Store
  │                         │                         │ Update Register
  │                         │                         │
  │                         │◄────────────────────────┤
  │                         │  MemoryResponse         │
  │◄────────────────────────┤  (success=true)         │
  │                         │                         │
  │  Load from              │                         │
  │  0x10000004             │                         │
  ├─────────────────────────┤                         │
  │  MemoryTransaction      │                         │
  │  (LOAD, addr)           ├────────────────────────►│
  │                         │                         │ Process Load
  │                         │                         │ Read Register
  │                         │                         │
  │                         │◄────────────────────────┤
  │                         │  MemoryResponse         │
  │◄────────────────────────┤  (data=0xDEADBEEF)      │
  │                         │                         │
```

## Device Register Map (Simple Echo Device)

```
Address      Register      Access   Description
0x10000000   DATA_IN       W        Write data to device
0x10000004   DATA_OUT      R        Read echoed data
0x10000008   STATUS        R        Device status
0x1000000C   CONTROL       R/W      Control register
```

### Register Behavior

**DATA_IN (0x10000000, Write-Only)**
- Writing to this register stores data in device
- Triggers echo operation (copies to DATA_OUT)
- Takes 10 cycles to complete

**DATA_OUT (0x10000004, Read-Only)**
- Returns last written DATA_IN value
- Read returns 0 if no data written yet

**STATUS (0x10000008, Read-Only)**
- Bit 0: BUSY (1 when processing, 0 when idle)
- Bit 1: DATA_READY (1 when DATA_OUT is valid)

**CONTROL (0x1000000C, Read/Write)**
- Bit 0: RESET (write 1 to reset device)
- Other bits reserved

## SST Configuration

### Partition Setup

The system uses SST's MPI-based partitioning:

```python
# SST Configuration for distributed simulation
import sst

# MPI rank determines which component runs where
rank = sst.getMPIRank()

if rank == 0:
    # Partition 0: QEMU component
    qemu = sst.Component("qemu0", "qemu.QEMUComponent")
    qemu.addParams({
        "clock": "1GHz",
        "machine": "virt",
        "cpu": "rv32",
        "memory": "128M",
        "device_base": "0x10000000",
        "device_size": "4K"
    })

elif rank == 1:
    # Partition 1: ACALSim device
    device = sst.Component("device0", "acalsim.DeviceComponent")
    device.addParams({
        "clock": "1GHz",
        "base_addr": "0x10000000",
        "size": "4K"
    })

# Create link between components
link = sst.Link("qemu_device_link")
if rank == 0:
    link.connect((qemu, "device_port", "1ns"), ...)
if rank == 1:
    link.connect((device, "cpu_port", "1ns"), ...)
```

## Timing and Synchronization

### Clock Domains

- **QEMU Clock**: Runs at configured frequency (e.g., 1GHz)
- **ACALSim Device Clock**: Independent clock, also 1GHz
- **SST Synchronization**: Handles clock domain crossing

### Latency Model

```
Operation              Latency (cycles)
-----------------      ----------------
QEMU → SST Event       1 (configurable)
SST → ACALSim          1 (link latency)
Device Processing      10 (echo operation)
Response → QEMU        2 (link latency)
-----------------      ----------------
Total Round-trip       ~14 cycles
```

## Implementation Phases

### Phase 1: Simple Echo Device (Current)
- Basic load/store support
- Simple register map
- Single-threaded operation
- Synchronous responses

### Phase 2: Realistic Device
- Pipelined operations
- DMA support
- Interrupt generation
- Asynchronous responses

### Phase 3: Performance Optimization
- Zero-copy data transfer
- Batch processing
- Predictive prefetching
- Adaptive synchronization

## Testing Strategy

### Unit Tests
1. ACALSim device standalone tests
2. QEMU component memory mapping tests
3. SST Link communication tests

### Integration Tests
1. Single-process mode (debugging)
2. Multi-process same-node
3. Multi-process different nodes

### Performance Tests
1. Transaction throughput
2. Latency measurements
3. Scaling studies

## References

- [SST-Core Distributed Simulation](http://sst-simulator.org/SSTPages/SSTDeveloperMPITutorial/)
- [QEMU Memory API](https://qemu.readthedocs.io/en/latest/devel/memory.html)
- [ACALSim Event-Driven Simulation](../../docs/for-users/event-driven-simulation.md)
