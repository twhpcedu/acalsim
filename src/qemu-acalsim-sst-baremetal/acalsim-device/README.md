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

# ACALSim Device SST Component

This directory contains the ACALSim device component for the QEMU-ACALSim distributed SST simulation.

## Overview

This directory contains multiple ACALSim device components demonstrating different communication patterns with QEMU:

1. **ACALSimDeviceComponent** - Simple echo device
2. **ACALSimComputeDeviceComponent** - Compute accelerator
3. **ACALSimMMIODevice** - MMIO device with interrupt support (NEW!)

## Available Device Components

### 1. ACALSimDeviceComponent (Echo Device)

Simple memory-mapped echo device for testing basic MMIO functionality.

**Features**:
- Memory-Mapped Interface (4KB region)
- Cycle-Accurate Timing (10 cycles default)
- SST Event-Driven Communication
- Statistics Tracking

### 2. ACALSimComputeDeviceComponent

Compute accelerator modeling realistic computation workloads.

**Features**:
- Configurable compute latency
- Multiple operation modes
- Resource modeling

### 3. ACALSimMMIODevice (NEW!)

Advanced MMIO device demonstrating interrupt-driven I/O.

**Features**:
- **MMIO Interface**: Load/store for register access
- **Interrupt Support**: InterruptEvent for async notification
- **DMA-like Operations**: Configurable source/dest/length
- **Cycle-Accurate Timing**: Realistic operation latencies
- **Full Register Set**: CTRL, STATUS, INT_STATUS, INT_ENABLE, etc.
- **Statistics**: MMIO ops, interrupts, latencies

**See [README_MMIO_DEVICE.md](README_MMIO_DEVICE.md) for complete documentation including:**
- Communication patterns (MMIO + interrupts)
- Register map and access patterns
- Implementation examples
- Driver code (bare-metal C)
- SST configuration
- Best practices

## Device Register Map

```
Offset    Register      Access   Description
0x0000    DATA_IN       W        Write data to device (triggers echo)
0x0004    DATA_OUT      R        Read echoed data
0x0008    STATUS        R        Device status (bit 0=busy, bit 1=ready)
0x000C    CONTROL       R/W      Control register (bit 0=reset)
```

## Building

### Prerequisites

- SST-Core installed with `sst-config` in PATH
- C++17 compatible compiler
- Make build system

### Build Commands

```bash
# Build the component
make

# Install to SST element library
make install

# Clean build artifacts
make clean

# Remove from SST
make uninstall
```

### Verification

After installation, verify the component is available:

```bash
sst-info acalsim
```

You should see output showing the `QEMUDevice` component.

## Usage in SST Configuration

```python
import sst

# Create device component
device = sst.Component("device0", "acalsim.QEMUDevice")
device.addParams({
    "clock": "1GHz",              # Device clock frequency
    "base_addr": "0x10000000",    # Base address in memory map
    "size": "4096",               # Device size (4KB)
    "verbose": "1",               # Verbosity level (0-3)
    "echo_latency": "10"          # Echo operation latency in cycles
})

# Connect to QEMU component
link = sst.Link("qemu_device_link")
link.connect((qemu, "device_port", "1ns"),
             (device, "cpu_port", "1ns"))
```

## SST Event Interface

### Input Events (from QEMU)

**MemoryTransactionEvent**:
- `type`: LOAD or STORE
- `address`: Memory address
- `data`: Data value (for STORE)
- `size`: Transaction size (1, 2, or 4 bytes)
- `req_id`: Unique request ID

### Output Events (to QEMU)

**MemoryResponseEvent**:
- `req_id`: Request ID (matches input)
- `data`: Response data (for LOAD)
- `success`: Transaction success status

## Implementation Details

### Echo Operation

1. QEMU writes data to DATA_IN register (0x10000000)
2. Device sets BUSY status bit
3. After 10 cycles, data is copied to DATA_OUT register
4. Device clears BUSY and sets DATA_READY status bit
5. QEMU reads DATA_OUT register (0x10000004)

### Timing Model

```
Cycle 0:  Write to DATA_IN
Cycle 1:  STATUS = BUSY
Cycle 10: Echo complete, DATA_OUT = DATA_IN
Cycle 10: STATUS = DATA_READY
Cycle 11: Read from DATA_OUT returns echoed value
```

## Files

- **ACALSimDeviceComponent.hh**: Component header with SST event definitions
- **ACALSimDeviceComponent.cc**: Component implementation
- **Makefile**: Build system
- **README.md**: This file

## Testing

See `../tests/echo_test.c` for a simple RISC-V test program that exercises this device.

## References

- [SST-Core Documentation](http://sst-simulator.org/)
- [ACALSim Framework](../../../README.md)
- [QEMU-ACALSim Architecture](../ARCHITECTURE.md)
