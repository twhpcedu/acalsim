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

# QEMU-ACALSim Distributed SST Integration

This project demonstrates distributed simulation using SST with QEMU and ACALSim running as separate processes communicating via IPC.

## Architecture Overview

```
┌─────────────────────┐         SST IPC          ┌──────────────────────┐
│   QEMU SST Process  │◄────────────────────────►│ ACALSim SST Process  │
│                     │      (Partition 0)        │    (Partition 1)     │
│  ┌──────────────┐   │                           │  ┌────────────────┐  │
│  │ QEMU RISC-V  │   │                           │  │ Memory-Mapped  │  │
│  │   + uboot    │   │    Load/Store Requests    │  │    Device      │  │
│  │   + Apps     │───┼──────────────────────────►│  │  (ACALSim)     │  │
│  └──────────────┘   │                           │  └────────────────┘  │
└─────────────────────┘                           └──────────────────────┘
```

## Components

### 1. QEMU SST Component (`QEMUComponent`)
- Runs QEMU RISC-V emulator with uboot
- Exposes memory-mapped region for device communication
- Sends load/store transactions to ACALSim device via SST Link

### 2. ACALSim Device Component (`ACALSimDeviceComponent`)
- Models a simple memory-mapped device using ACALSim
- Receives load/store requests from QEMU
- Processes requests and returns responses

### 3. SST Multi-Process Configuration
- Configures SST to run in distributed mode (MPI)
- Partition 0: QEMU component
- Partition 1: ACALSim component
- Links connect components across partitions

## Simple Example: Echo Device

The initial example implements a simple echo device:
- Device has a 4KB memory-mapped region (0x10000000 - 0x10000FFF)
- QEMU writes data to device registers
- ACALSim device echoes data back
- QEMU reads back the echoed data

### Device Register Map
```
Offset  | Register      | Description
--------|---------------|----------------------------------
0x0000  | DATA_IN       | Write data to device
0x0004  | DATA_OUT      | Read echoed data from device
0x0008  | STATUS        | Device status (0=idle, 1=busy)
0x000C  | CONTROL       | Control register (bit 0=reset)
```

## Quick Start

### Automated Build and Run

```bash
# One-command build, install, and run
./build.sh

# Or step by step
./build.sh prereq    # Check prerequisites
./build.sh build     # Build components
./build.sh install   # Install to SST
./build.sh run       # Run simulation
```

### Manual Build Steps

```bash
# Using Makefile
make verify-sst      # Verify SST installation
make build           # Build all components
make install         # Install to SST
make run             # Run simulation

# Or manually
cd qemu-component && make && make install
cd ../acalsim-device && make && make install
cd ../config && mpirun -n 2 sst echo_device.py
```

## Building

### Prerequisites
- SST-Core with MPI support (installed and in PATH)
- C++17 compatible compiler
- MPI implementation (OpenMPI or MPICH)
- Make build system

### Detailed Build Steps
See [QUICKSTART.md](QUICKSTART.md) for detailed build instructions

## Running

### Single-Node Multi-Process
```bash
# Run with 2 MPI ranks (processes)
mpirun -n 2 sst config/echo_device.py
```

### Multi-Node Distributed
```bash
# Run QEMU on node0, ACALSim on node1
mpirun -n 1 -host node0 sst config/echo_device.py : \
       -n 1 -host node1 sst config/echo_device.py
```

## Testing

A simple RISC-V test program demonstrates the device communication:

```c
// test.c - Simple device test
#define DEVICE_BASE 0x10000000
#define DATA_IN     (DEVICE_BASE + 0x0)
#define DATA_OUT    (DEVICE_BASE + 0x4)
#define STATUS      (DEVICE_BASE + 0x8)

int main() {
    volatile uint32_t *data_in  = (uint32_t*)DATA_IN;
    volatile uint32_t *data_out = (uint32_t*)DATA_OUT;
    volatile uint32_t *status   = (uint32_t*)STATUS;

    // Write test pattern
    *data_in = 0xDEADBEEF;

    // Wait for device
    while(*status != 0);

    // Read back
    uint32_t result = *data_out;

    return (result == 0xDEADBEEF) ? 0 : 1;
}
```

## Project Structure

```
qemu-sst/
├── README.md                    # This file
├── ARCHITECTURE.md              # Detailed architecture documentation
├── QUICKSTART.md                # Quick start guide
├── Makefile                     # Top-level build system
├── build.sh                     # Automated build script
│
├── qemu-component/              # QEMU SST wrapper component
│   ├── QEMUComponent.hh         # Component header
│   ├── QEMUComponent.cc         # Component implementation
│   ├── Makefile                 # Build system
│   └── README.md                # Component documentation
│
├── acalsim-device/              # ACALSim device component
│   ├── ACALSimDeviceComponent.hh   # Component header
│   ├── ACALSimDeviceComponent.cc   # Component implementation
│   ├── Makefile                    # Build system
│   └── README.md                   # Component documentation
│
├── config/                      # SST configuration files
│   ├── echo_device.py           # Multi-process simulation config
│   └── README.md                # Configuration guide
│
└── tests/                       # Test programs
    ├── echo_test.c              # Reference RISC-V test program
    ├── Makefile                 # Test build system
    └── README.md                # Testing documentation
```

## Development Roadmap

### Phase 1: Foundation ✓ COMPLETE
- [x] Project structure setup
- [x] Simple echo device component (ACALSim)
- [x] Basic QEMU wrapper component
- [x] Single-node multi-process testing
- [x] Build system and automation
- [x] Comprehensive documentation

**Status**: Phase 1 is complete. The system demonstrates:
- Distributed SST simulation with 2 processes
- Memory-mapped device communication
- Request/response transaction flow
- Cycle-accurate echo device model
- Test program simulation

### Phase 2: Full QEMU Integration (Future)
- [ ] Integrate with real QEMU RISC-V emulator
- [ ] QEMU memory API integration
- [ ] Load RISC-V binaries into QEMU
- [ ] Trap device memory accesses
- [ ] Interrupt support

### Phase 3: Advanced Device Features (Future)
- [ ] DMA support
- [ ] Pipelined operations
- [ ] Asynchronous responses
- [ ] Performance monitoring
- [ ] Multiple device instances

### Phase 4: Distributed Execution (Future)
- [ ] Multi-node distributed execution
- [ ] Checkpoint/restart support
- [ ] Trace/debug infrastructure
- [ ] Performance optimization
- [ ] Scalability studies

## References

- [SST-Core Documentation](http://sst-simulator.org/)
- [SST Distributed Simulation](http://sst-simulator.org/SSTPages/SSTDeveloperMPITutorial/)
- [QEMU RISC-V](https://www.qemu.org/docs/master/system/target-riscv.html)
- [ACALSim Framework](../../README.md)
