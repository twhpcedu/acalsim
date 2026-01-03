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

# QEMU-ACALSim Quick Start Guide

## Project Overview

This project demonstrates a **distributed simulation** architecture combining:
- **QEMU**: Full-system RISC-V emulator running uboot and applications
- **ACALSim**: Cycle-accurate device modeling framework
- **SST**: Discrete-event simulation framework with MPI-based IPC

### Key Concept: Two Separate Processes

```
┌──────────────────┐            ┌──────────────────┐
│  Process 1       │            │  Process 2       │
│  (MPI Rank 0)    │    IPC     │  (MPI Rank 1)    │
│                  │◄──────────►│                  │
│  QEMU RISC-V     │            │  ACALSim Device  │
│  + uboot         │  SST Link  │  + Sim Engine    │
│  + Apps          │            │  + Event Queue   │
└──────────────────┘            └──────────────────┘
```

## Simple Example: Echo Device

The first example implements a memory-mapped echo device:

1. **QEMU writes** data to device register (0x10000000)
2. **SST sends** transaction event to ACALSim process via IPC
3. **ACALSim processes** the write in cycle-accurate model
4. **QEMU reads** back echoed data from device (0x10000004)

## Project Structure

```
src/qemu-sst/
├── README.md              # Project overview
├── ARCHITECTURE.md        # Detailed design document
├── QUICKSTART.md          # This file
│
├── acalsim-device/        # ACALSim device component (Process 2)
│   ├── DeviceComponent.hh # Device SST component header
│   ├── DeviceComponent.cc # Device implementation
│   └── Makefile
│
├── qemu-component/        # QEMU wrapper component (Process 1)
│   ├── QEMUComponent.hh   # QEMU SST component header
│   ├── QEMUComponent.cc   # QEMU integration
│   └── Makefile
│
├── config/                # SST configuration
│   └── echo_device.py     # Multi-process setup script
│
└── tests/                 # Test programs
    └── echo_test.c        # Simple device test
```

## Next Steps

### 1. Implement ACALSim Device Component

Create a simple SST component that:
- Receives load/store events via SST Link
- Models 4 device registers
- Echoes data from input to output
- Returns responses via SST Link

**Files to create:**
- `acalsim-device/DeviceComponent.hh`
- `acalsim-device/DeviceComponent.cc`
- `acalsim-device/Makefile`

### 2. Implement QEMU Component Wrapper

Create SST wrapper for QEMU that:
- Launches QEMU RISC-V system emulator
- Intercepts memory accesses to device region
- Converts to SST events
- Handles responses

**Files to create:**
- `qemu-component/QEMUComponent.hh`
- `qemu-component/QEMUComponent.cc`
- `qemu-component/Makefile`

### 3. Create SST Configuration

Python script to configure multi-process simulation:
- Define two components (QEMU and Device)
- Assign to different MPI ranks
- Connect via SST Link

**File to create:**
- `config/echo_device.py`

### 4. Write Test Program

Simple RISC-V C program:
- Writes test pattern to device
- Reads back echoed value
- Verifies correctness

**File to create:**
- `tests/echo_test.c`

## Development Workflow

```bash
# 1. Build components
cd acalsim-device && make && make install
cd ../qemu-component && make && make install

# 2. Build test program
cd ../tests && make

# 3. Run simulation (2 processes, same node)
cd ../config
mpirun -n 2 sst echo_device.py

# 4. Check results
# Process 0 (QEMU) and Process 1 (ACALSim) logs
```

## Device Register Map

```
Offset    Register    Access   Description
0x0000    DATA_IN     W        Write data to device
0x0004    DATA_OUT    R        Read echoed data
0x0008    STATUS      R        0=idle, 1=busy
0x000C    CONTROL     R/W      bit 0 = reset
```

## Example Test Program

```c
#define DEV_BASE 0x10000000
#define DATA_IN  (DEV_BASE + 0x00)
#define DATA_OUT (DEV_BASE + 0x04)

int main() {
    volatile uint32_t *din  = (uint32_t*)DATA_IN;
    volatile uint32_t *dout = (uint32_t*)DATA_OUT;

    *din = 0xDEADBEEF;  // Write to device
    while(*dout != 0xDEADBEEF);  // Wait for echo
    return 0;
}
```

## Expected Behavior

```
[Process 0 - QEMU]
Info: QEMU started
Info: Memory transaction: STORE addr=0x10000000 data=0xDEADBEEF
Info: Sending SST event to device

[Process 1 - ACALSim Device]
Info: Received STORE request: addr=0x10000000 data=0xDEADBEEF
Info: Writing to DATA_IN register
Info: Echoing to DATA_OUT (10 cycles)
Info: Sending response

[Process 0 - QEMU]
Info: Memory transaction: LOAD addr=0x10000004
Info: Sending SST event to device

[Process 1 - ACALSim Device]
Info: Received LOAD request: addr=0x10000004
Info: Reading DATA_OUT register: 0xDEADBEEF
Info: Sending response

[Process 0 - QEMU]
Info: Received data: 0xDEADBEEF
Info: Test PASSED
```

## Current Status

✅ Project structure created
✅ Architecture documented
⏳ ACALSim device component (next task)
⏳ QEMU wrapper component
⏳ SST configuration
⏳ Test program

## Resources

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Detailed design
- [README.md](./README.md) - Project overview
- [SST Documentation](http://sst-simulator.org/)
- [QEMU Documentation](https://qemu.readthedocs.io/)
- [ACALSim Framework](../../README.md)
