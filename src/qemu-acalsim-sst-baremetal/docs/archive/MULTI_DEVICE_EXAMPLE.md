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

# Multi-Device Communication Example

This document describes the advanced multi-device example that demonstrates QEMU communicating with multiple SST devices and inter-device communication.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Building](#building)
5. [Running](#running)
6. [Implementation Notes](#implementation-notes)
7. [Future Enhancements](#future-enhancements)

---

## Overview

The multi-device example extends the single-device QEMU-SST integration to demonstrate:

1. **Multiple Device Communication**: QEMU communicates with two separate SST devices
2. **Device Specialization**: Different devices serve different purposes (echo vs. compute)
3. **Inter-Device Communication**: Devices can exchange data via SST peer links
4. **Address-Based Routing**: Transactions are routed based on memory-mapped address ranges

### Devices

**Device 1: Echo Device**
- Base address: `0x10200000`
- Functionality: Echoes data written to it
- Use case: Simple data exchange and testing

**Device 2: Compute Device**
- Base address: `0x10300000`
- Functionality: Performs arithmetic operations (ADD, SUB, MUL, DIV)
- Use case: Computational accelerator modeling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SST Simulation                          │
│                                                               │
│  ┌───────────────┐                                           │
│  │               │  device_port     ┌────────────────┐      │
│  │  QEMUBinary   ├─────────────────►│ Echo Device    │      │
│  │  Component    │                  │ (0x10200000)   │      │
│  │               │                  └───────┬────────┘      │
│  │               │                          │                │
│  │               │                          │ peer_port      │
│  │               │                          │                │
│  │               │                          ▼                │
│  │               │                  ┌────────────────┐      │
│  │               │                  │ Compute Device │      │
│  │               │                  │ (0x10300000)   │      │
│  └───────────────┘                  └────────────────┘      │
│         │                                   ▲                │
│         │                                   │                │
│         │       Unix Socket (MMIO)          │                │
└─────────┼───────────────────────────────────┼────────────────┘
          │                                   │
          ▼                                   │
┌─────────────────────────────────────────────┼────────────────┐
│         QEMU Process                        │                │
│                                             │                │
│  ┌──────────────────┐           ┌──────────┴────────┐       │
│  │ SST Device       │           │ SST Device        │       │
│  │ @ 0x10200000     │           │ @ 0x10300000      │       │
│  │ (sst-device.c)   │           │ (sst-device.c)    │       │
│  └──────────────────┘           └───────────────────┘       │
│           ▲                              ▲                   │
│           │                              │                   │
│           │         MMIO Access          │                   │
│           │                              │                   │
│  ┌────────┴──────────────────────────────┴────────┐         │
│  │          RISC-V CPU                            │         │
│  │      (multi_device_test.elf)                   │         │
│  └────────────────────────────────────────────────┘         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. RISC-V Test Program (multi_device_test.c)

Located in `riscv-programs/multi_device_test.c`.

**Tests performed**:
1. **Echo Device Test**: Writes data to echo device and reads it back
2. **Compute Device Tests**:
   - Addition: 42 + 58 = 100
   - Multiplication: 12 * 5 = 60
   - Division: 100 / 4 = 25
3. **Inter-Device Communication Test**: Demonstrates devices exchanging data

**Memory Map**:
```c
// Echo Device (Device 1) - 0x10200000
#define ECHO_DATA_IN      0x10200000  // Write data
#define ECHO_DATA_OUT     0x10200004  // Read echoed data
#define ECHO_STATUS       0x10200008  // Status register
#define ECHO_CONTROL      0x1020000C  // Control register

// Compute Device (Device 2) - 0x10300000
#define COMPUTE_OPERAND_A    0x10300000  // First operand
#define COMPUTE_OPERAND_B    0x10300004  // Second operand
#define COMPUTE_OPERATION    0x10300008  // Operation code
#define COMPUTE_RESULT       0x1030000C  // Result
#define COMPUTE_STATUS       0x10300010  // Status register
#define COMPUTE_CONTROL      0x10300014  // Control register
#define COMPUTE_PEER_OUT     0x10300018  // Send to peer
#define COMPUTE_PEER_IN      0x1030001C  // Receive from peer
```

### 2. Echo Device Component (ACALSimDeviceComponent)

Located in `acalsim-device/ACALSimDeviceComponent.{hh,cc}`.

**Features**:
- Echoes data back to CPU with configurable latency
- Supports CPU port for QEMU communication
- Optional peer port for inter-device communication

**Register Map**:
- `0x00`: DATA_IN (Write)
- `0x04`: DATA_OUT (Read)
- `0x08`: STATUS (Read)
- `0x0C`: CONTROL (Read/Write)

### 3. Compute Device Component (ACALSimComputeDeviceComponent)

Located in `acalsim-device/ACALSimComputeDeviceComponent.{hh,cc}`.

**Features**:
- Performs arithmetic operations (ADD, SUB, MUL, DIV)
- Configurable computation latency
- Supports peer communication with other devices
- Error detection (e.g., division by zero)

**Register Map**:
- `0x00`: OPERAND_A (Write)
- `0x04`: OPERAND_B (Write)
- `0x08`: OPERATION (Write) - 0=ADD, 1=SUB, 2=MUL, 3=DIV
- `0x0C`: RESULT (Read)
- `0x10`: STATUS (Read)
- `0x14`: CONTROL (Read/Write)
- `0x18`: PEER_DATA_OUT (Write) - Send to peer device
- `0x1C`: PEER_DATA_IN (Read) - Receive from peer device

**Operations**:
```c
enum Operation {
    OP_ADD = 0,  // Addition
    OP_SUB = 1,  // Subtraction
    OP_MUL = 2,  // Multiplication
    OP_DIV = 3   // Division
};
```

### 4. SST Configuration (qemu_multi_device_test.py)

Located in `qemu_multi_device_test.py`.

**Configuration**:
- Creates QEMU component
- Creates echo device at 0x10200000
- Creates compute device at 0x10300000
- Links QEMU to devices
- Links devices together for peer communication

---

## Building

### 1. Build RISC-V Test Program

```bash
cd riscv-programs
make multi_device_test.elf
```

**Output files**:
- `multi_device_test.elf` - RISC-V executable
- `multi_device_test.bin` - Raw binary
- `multi_device_test.dump` - Disassembly

### 2. Build SST Device Components

```bash
cd acalsim-device
make clean
make
make install
```

**Verify installation**:
```bash
sst-info acalsim

# Should show:
# Component: QEMUDevice (Echo device)
# Component: ComputeDevice (Compute device)
```

### 3. Verify QEMU Device Integration

The custom QEMU must have SST devices at both addresses:

```bash
$QEMU_PATH -M virt,help 2>&1 | grep sst
# Should show sst-device
```

See `PHASE2C_INTEGRATION.md` for details on QEMU integration.

---

## Running

### Option 1: SST Simulation (Recommended)

This runs the full simulation with both SST devices:

```bash
# Set environment
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH
export QEMU_PATH=/home/user/qemu-build/install/bin/qemu-system-riscv32

# Run SST simulation
sst qemu_multi_device_test.py
```

**Expected output**:
```
QEMU-SST Multi-Device Test
===========================================

Device 1: Echo device at     0x10200000
Device 2: Compute device at  0x10300000

[TEST 1] Echo Device Test
  [PASS] Echo device: 0xCAFEBABE

[TEST 2] Compute Device Test
  [PASS] ADD: 42 + 58 = 0x00000064
  [PASS] MUL: 12 * 5 = 0x0000003C
  [PASS] DIV: 100 / 4 = 0x00000019

[TEST 3] Inter-Device Communication Test
  Requesting peer data from compute device...
  Received peer data: 0xCAFEBABE
  Echo device result: 0xCAFEBABE
  [PASS] Inter-device communication successful

===========================================
  All tests PASSED!
===========================================
```

### Option 2: Standalone QEMU (Limited)

You can run in standalone QEMU, but without SST devices the test will fail:

```bash
cd riscv-programs
make test-multi
```

This will launch QEMU but the MMIO accesses will timeout since there are no SST devices.

---

## Implementation Notes

### Current Limitations

**1. Single Device Port in QEMUBinaryComponent**

The current `QEMUBinaryComponent` has a single `device_port`. For true multi-device support, it needs:

```cpp
// Desired implementation
SST::Link* device1_port;  // Echo device
SST::Link* device2_port;  // Compute device

// Route based on address
if (address >= 0x10200000 && address < 0x10201000) {
    device1_port->send(event);
} else if (address >= 0x10300000 && address < 0x10301000) {
    device2_port->send(event);
}
```

**Current workaround**: All transactions route through echo device. For compute device access, additional routing logic would be needed.

**2. QEMU Device Integration**

Currently, there's one `sst-device.c` in QEMU. For multiple devices, you need either:
- Multiple SST device instances at different addresses
- Single device with internal routing

### Inter-Device Communication

Devices communicate via SST `peer_port`:

**Echo Device** (`ACALSimDeviceComponent`):
```cpp
// Optionally add peer_port to echo device
SST::Link* peer_link_;

// Handle peer messages
void handlePeerMessage(SST::Event *ev) {
    // Respond with current echo data
}
```

**Compute Device** (`ACALSimComputeDeviceComponent`):
```cpp
SST::Link* peer_link_;

// Send request to peer
void sendToPeer(uint32_t data) {
    peer_link_->send(new DeviceMessageEvent(DATA_REQUEST, data, 0));
}

// Handle peer response
void handlePeerMessage(SST::Event *ev) {
    DeviceMessageEvent *msg = dynamic_cast<DeviceMessageEvent*>(ev);
    peer_data_in_ = msg->getData();
    status_ |= STATUS_PEER_READY;
}
```

**Message Types**:
```cpp
enum MessageType {
    DATA_REQUEST,      // Request data from peer
    DATA_RESPONSE,     // Respond with data
    COMPUTE_REQUEST,   // Request computation
    COMPUTE_RESPONSE   // Respond with result
};
```

### Performance Characteristics

| Operation | Latency (cycles) | Notes |
|-----------|-----------------|-------|
| Echo | 10 | Configurable via `echo_latency` |
| Computation | 100 | Configurable via `compute_latency` |
| Peer Message | ~10-20 | Link latency + processing |

---

## Future Enhancements

### 1. Full Multi-Device Routing

Extend `QEMUBinaryComponent` to support multiple device ports:

```cpp
class QEMUBinaryComponent : public SST::Component {
    SST_ELI_DOCUMENT_PORTS(
        {"device1_port", "Port for device 1", {}},
        {"device2_port", "Port for device 2", {}},
        {"device3_port", "Port for device 3", {}}
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"device1_base", "Device 1 base address", "0x10200000"},
        {"device1_size", "Device 1 memory size", "4096"},
        {"device2_base", "Device 2 base address", "0x10300000"},
        {"device2_size", "Device 2 memory size", "4096"}
    )

    void handleMMIORequest() {
        // Route based on address ranges
        for (auto& dev : devices) {
            if (addr >= dev.base && addr < dev.base + dev.size) {
                dev.port->send(event);
                return;
            }
        }
    }
};
```

### 2. Device Router Component

Create a standalone router component:

```cpp
class DeviceRouterComponent : public SST::Component {
    SST_ELI_DOCUMENT_PORTS(
        {"cpu_port", "Port from CPU", {}},
        {"device_port_%d", "Port to device", {}}
    )

    void handleRequest(SST::Event *ev) {
        auto *trans = dynamic_cast<MemoryTransactionEvent*>(ev);
        uint64_t addr = trans->getAddress();

        // Find matching device
        for (size_t i = 0; i < device_ports.size(); i++) {
            if (addr >= device_bases[i] &&
                addr < device_bases[i] + device_sizes[i]) {
                device_ports[i]->send(trans);
                return;
            }
        }

        // Address not mapped - send error response
        sendErrorResponse(trans);
    }
};
```

### 3. Advanced Inter-Device Protocols

Implement more complex inter-device protocols:

**DMA (Direct Memory Access)**:
```cpp
// Device 1 initiates DMA from Device 2
peer_link->send(new DMARequestEvent(src_addr, dst_addr, size));
```

**Coherency Protocol**:
```cpp
// Cache coherency messages between devices
peer_link->send(new CoherencyEvent(INVALIDATE, cache_line));
```

**Network-on-Chip (NoC)**:
```cpp
// Route through NoC for multi-device communication
noc_link->send(new NoCPacket(src_id, dst_id, payload));
```

### 4. Additional Device Types

**Memory Controller**:
```cpp
class MemoryControllerComponent : public SST::Component {
    // Models DRAM with timing
    uint64_t read_latency_cycles;
    uint64_t write_latency_cycles;
};
```

**Interrupt Controller**:
```cpp
class InterruptControllerComponent : public SST::Component {
    // Routes interrupts to CPU
    void raiseInterrupt(uint32_t irq_num);
};
```

**DMA Controller**:
```cpp
class DMAControllerComponent : public SST::Component {
    // Handles bulk data transfers
    void transferData(uint64_t src, uint64_t dst, size_t size);
};
```

### 5. Multiple QEMU Instances

Support multiple QEMU instances for multi-core simulation:

```python
# SST Configuration
qemu0 = sst.Component("qemu0", "qemubinary.QEMUBinary")
qemu1 = sst.Component("qemu1", "qemubinary.QEMUBinary")

# Shared memory device
shared_mem = sst.Component("shared_mem", "acalsim.SharedMemory")

# Links
link0 = sst.Link("qemu0_mem")
link0.connect((qemu0, "mem_port", "1ns"),
              (shared_mem, "cpu0_port", "1ns"))

link1 = sst.Link("qemu1_mem")
link1.connect((qemu1, "mem_port", "1ns"),
              (shared_mem, "cpu1_port", "1ns"))
```

---

## Summary

The multi-device example demonstrates:

✅ **Multiple SST Devices**: Echo and Compute devices with different functionalities
✅ **Device Specialization**: Different register maps and operations
✅ **Inter-Device Communication**: Peer links for device-to-device data exchange
✅ **Extensible Architecture**: Foundation for more complex device networks

**Key Takeaways**:
1. SST makes it easy to model multiple devices with different characteristics
2. Peer links enable device-to-device communication modeling
3. Address-based routing allows QEMU to access multiple memory-mapped devices
4. The architecture scales to more devices and complex protocols

For more examples and detailed documentation, see:
- `USER_GUIDE.md` - Creating custom devices and tests
- `DEVELOPER_GUIDE.md` - Detailed development guide
- `BUILD_AND_TEST.md` - Build and test instructions

---

**Last Updated**: 2025-11-10
**Status**: Demonstration/Prototype
**Next Steps**: Implement full multi-device routing in QEMUBinaryComponent
