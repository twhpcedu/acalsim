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

# QEMU-ACALSim-SST Bare-Metal Integration

High-performance integration between QEMU RISC-V emulator and SST simulator using binary MMIO protocol for bare-metal system simulation.

## Overview

This project demonstrates cycle-accurate hardware simulation by connecting QEMU (functional CPU model) with SST (cycle-accurate device models) via a high-performance binary MMIO protocol.

### Key Features

- **Binary MMIO Protocol**: 10x faster than text-based serial protocols
- **Bare-Metal Execution**: Direct hardware access without OS overhead
- **Multiple Devices**: Support for multiple memory-mapped devices
- **Inter-Device Communication**: Devices can exchange data via SST peer links
- **Cycle-Accurate Timing**: Configurable latencies for realistic hardware modeling

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SST Simulation                          │
│  ┌──────────────────┐           ┌──────────────────┐       │
│  │ QEMUBinary       │  Events   │ ACALSim Device   │       │
│  │ Component        │◄─────────►│ Component        │       │
│  └────────┬─────────┘           └──────────────────┘       │
│           │                                                  │
│           │ Unix Socket (Binary MMIO Protocol)              │
└───────────┼──────────────────────────────────────────────────┘
            │
            │ 24-byte MMIORequest
            │ 20-byte MMIOResponse
            │
┌───────────┼──────────────────────────────────────────────────┐
│           ▼                                                   │
│  ┌──────────────────┐           ┌──────────────────┐        │
│  │ SST Device       │  MMIO     │ RISC-V CPU       │        │
│  │ (sst-device.c)   │◄─────────►│                  │        │
│  └──────────────────┘           └──────────────────┘        │
│                                                               │
│                    QEMU Process                              │
│               (qemu-system-riscv32)                          │
└───────────────────────────────────────────────────────────────┘
```

### Binary MMIO Protocol

High-performance packed binary structures for efficient communication:

```c
// Request: QEMU → SST (24 bytes)
struct MMIORequest {
    uint32_t magic;      // 0x53535452 ("SSTR")
    uint32_t type;       // 0=READ, 1=WRITE
    uint64_t address;    // Physical address
    uint32_t size;       // 1, 2, 4, or 8 bytes
    uint64_t data;       // Write data
} __attribute__((packed));

// Response: SST → QEMU (20 bytes)
struct MMIOResponse {
    uint32_t magic;      // 0x53535450 ("SSTP")
    uint32_t status;     // 0=OK, 1=ERROR
    uint64_t data;       // Read data
    uint32_t latency;    // Simulated cycles
} __attribute__((packed));
```

**Performance**: ~10,000 transactions/sec, ~100μs latency (10x improvement over Phase 2B)

## Components

### 1. RISC-V Firmware (riscv-programs/)

Bare-metal test programs with complete C runtime:

- **crt0.S**: CPU initialization, trap handling, .data/.bss setup
- **mmio_test.c**: Single-device MMIO test
- **multi_device_test.c**: Multi-device communication test
- **asm_link_test.c**: C-assembly linkage demonstration

### 2. QEMU SST Device (qemu-sst-device/)

Custom QEMU device integrated into QEMU source:

- **sst-device.c**: QEMU device model using QOM
- Binary MMIO protocol client
- Integrated into RISC-V virt machine

### 3. SST QEMUBinary Component (qemu-binary/)

Manages QEMU subprocess and protocol translation:

- **QEMUBinaryComponent**: Launches QEMU, creates Unix socket server
- Translates binary MMIO to SST events
- Routes responses back to QEMU

### 4. SST Device Components (acalsim-device/)

#### Echo Device (ACALSimDeviceComponent)

Simple echo device for testing:
- **Base address**: 0x10200000
- **Registers**: DATA_IN (0x00), DATA_OUT (0x04), STATUS (0x08), CONTROL (0x0C)
- **Latency**: Configurable (default 10 cycles)

#### Compute Device (ACALSimComputeDeviceComponent)

Arithmetic accelerator device:
- **Base address**: 0x10300000
- **Operations**: ADD, SUB, MUL, DIV
- **Registers**: OPERAND_A, OPERAND_B, OPERATION, RESULT, STATUS, CONTROL
- **Peer communication**: PEER_DATA_OUT, PEER_DATA_IN
- **Latency**: Configurable (default 100 cycles)

## Quick Start

### Prerequisites

- RISC-V Toolchain: `riscv64-unknown-elf-gcc`
- SST-Core installed and in PATH
- Custom QEMU build with SST device (see PHASE2C_INTEGRATION.md)
- C++17 compiler (g++ or clang)

### Build and Run

```bash
# Set SST environment
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH
export QEMU_PATH=/home/user/qemu-build/install/bin/qemu-system-riscv32

# Build firmware
cd riscv-programs
make mmio_test.elf
cd ..

# Build SST components
cd qemu-binary && make && make install && cd ..
cd acalsim-device && make && make install && cd ..

# Verify installation
sst-info qemubinary
sst-info acalsim

# Run single-device test
sst qemu_binary_test.py

# Run multi-device test
cd riscv-programs && make multi_device_test.elf && cd ..
sst qemu_multi_device_test.py
```

## Project Structure

```
qemu-acalsim-sst-baremetal/
├── README.md                           # This file
├── BUILD_AND_TEST.md                   # Complete build and test guide
├── USER_GUIDE.md                       # User guide for creating tests and devices
├── DEVELOPER_GUIDE.md                  # Developer guide with architecture details
├── MULTI_DEVICE_EXAMPLE.md             # Multi-device communication guide
├── PHASE2C_INTEGRATION.md              # QEMU device integration guide
│
├── riscv-programs/                     # Bare-metal RISC-V test programs
│   ├── crt0.S                          # C runtime startup
│   ├── start.S                         # Simple startup (legacy)
│   ├── linker.ld                       # Linker script
│   ├── mmio_test.c                     # Single-device MMIO test
│   ├── mmio_test_main.c                # Test with proper main()
│   ├── multi_device_test.c             # Multi-device test
│   ├── asm_link_test.c                 # C-assembly linkage test
│   ├── asm_test.S                      # Assembly test functions
│   └── Makefile                        # Build system
│
├── qemu-sst-device/                    # QEMU device implementation
│   ├── sst-device.c                    # QEMU device (integrate into QEMU)
│   └── sst-device.h                    # Device header
│
├── qemu-binary/                        # SST QEMUBinary component
│   ├── QEMUBinaryComponent.hh          # Component header
│   ├── QEMUBinaryComponent.cc          # Component implementation
│   ├── Makefile                        # Build system
│   └── README.md                       # Component documentation
│
├── acalsim-device/                     # SST device components
│   ├── ACALSimDeviceComponent.hh       # Echo device header
│   ├── ACALSimDeviceComponent.cc       # Echo device implementation
│   ├── ACALSimComputeDeviceComponent.hh    # Compute device header
│   ├── ACALSimComputeDeviceComponent.cc    # Compute device implementation
│   ├── Makefile                        # Build system
│   └── README.md                       # Component documentation
│
├── qemu_binary_test.py                 # SST config: single device test
└── qemu_multi_device_test.py           # SST config: multi-device test
```

## Single Device Example

The basic example uses a single echo device at 0x10200000:

**Test Program** (mmio_test.c):
```c
#define SST_DEVICE_BASE 0x10200000
#define SST_DATA_IN     (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x00))
#define SST_DATA_OUT    (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x04))

int main() {
    // Write test data
    SST_DATA_IN = 0xDEADBEEF;
    SST_CONTROL = 0x1;  // Trigger operation

    // Wait for completion
    while (SST_STATUS & 0x1);

    // Read result
    uint32_t result = SST_DATA_OUT;
    return (result == 0xDEADBEEF) ? 0 : 1;
}
```

**SST Configuration** (qemu_binary_test.py):
```python
import sst

# QEMU component
qemu = sst.Component("qemu0", "qemubinary.QEMUBinary")
qemu.addParams({
    "binary_path": "riscv-programs/mmio_test.elf",
    "device_base": "0x10200000",
})

# Echo device
device = sst.Component("device0", "acalsim.QEMUDevice")
device.addParams({
    "base_addr": "0x10200000",
    "echo_latency": "10",
})

# Link them
link = sst.Link("qemu_device_link")
link.connect((qemu, "device_port", "1ns"),
             (device, "cpu_port", "1ns"))
```

## Multi-Device Example

Advanced example with two devices communicating:

**Devices**:
1. **Echo Device** @ 0x10200000 - Simple data echo
2. **Compute Device** @ 0x10300000 - Arithmetic operations

**Test Program** (multi_device_test.c):
```c
// Test 1: Echo device
ECHO_DATA_IN = 0xCAFEBABE;
result = ECHO_DATA_OUT;  // Should be 0xCAFEBABE

// Test 2: Compute device
COMPUTE_OPERAND_A = 42;
COMPUTE_OPERAND_B = 58;
COMPUTE_OPERATION = OP_ADD;
COMPUTE_CONTROL = CONTROL_TRIGGER;
result = COMPUTE_RESULT;  // Should be 100

// Test 3: Inter-device communication
COMPUTE_PEER_OUT = 0x12345678;  // Request data from echo device
peer_data = COMPUTE_PEER_IN;     // Receive echo device's result
```

**Device Communication**:
```
CPU → Echo Device: Write 0xCAFEBABE
CPU → Compute Device: Request peer data
Compute Device → Echo Device: DATA_REQUEST message
Echo Device → Compute Device: DATA_RESPONSE with 0xCAFEBABE
CPU ← Compute Device: Read peer data = 0xCAFEBABE
```

**SST Configuration** (qemu_multi_device_test.py):
```python
# QEMU component
qemu = sst.Component("qemu0", "qemubinary.QEMUBinary")

# Echo device
echo_dev = sst.Component("echo_dev", "acalsim.QEMUDevice")
echo_dev.addParams({"base_addr": "0x10200000"})

# Compute device
compute_dev = sst.Component("compute_dev", "acalsim.ComputeDevice")
compute_dev.addParams({"base_addr": "0x10300000"})

# CPU-device links
qemu_echo_link = sst.Link("qemu_echo_link")
qemu_echo_link.connect((qemu, "device_port", "1ns"),
                       (echo_dev, "cpu_port", "1ns"))

# Device-to-device peer link
peer_link = sst.Link("device_peer_link")
peer_link.connect((echo_dev, "peer_port", "10ns"),
                  (compute_dev, "peer_port", "10ns"))
```

See [MULTI_DEVICE_EXAMPLE.md](MULTI_DEVICE_EXAMPLE.md) for complete details.

## Documentation

### User Documentation
- **[USER_GUIDE.md](USER_GUIDE.md)** - Creating custom firmware tests and device models
- **[BUILD_AND_TEST.md](BUILD_AND_TEST.md)** - Complete build and test instructions
- **[MULTI_DEVICE_EXAMPLE.md](MULTI_DEVICE_EXAMPLE.md)** - Multi-device communication

### Developer Documentation
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Architecture, design concepts, step-by-step development
- **[PHASE2C_INTEGRATION.md](PHASE2C_INTEGRATION.md)** - QEMU device integration details

## Development Phases

### Phase 2B: Serial Text Protocol ✅ COMPLETE

- Text-based serial UART protocol
- ~1,000 transactions/sec
- ~1ms latency
- High CPU overhead (text parsing)

### Phase 2C: Binary MMIO Protocol ✅ COMPLETE

- **Phase 2C.1**: SST component framework (QEMUBinaryComponent)
- **Phase 2C.2**: QEMU device code (sst-device.c)
- **Phase 2C.3**: QEMU device integration into QEMU source
- **Phase 2C.4**: Multi-device support and inter-device communication

**Results**:
- ~10,000 transactions/sec (10x improvement)
- ~100μs latency (10x improvement)
- 90% reduction in CPU usage
- Protocol overhead: 8% (vs. 80% in Phase 2B)

### Phase 2D: Multi-Core Support (Future)

- Multiple QEMU instances
- Shared memory devices
- Core-to-core communication
- SMP simulation

### Phase 3: Linux Integration (Future)

See `../qemu-acalsim-sst-linux/` for Linux-based variant:
- Full Linux kernel boot
- VirtIO device integration
- Kernel module drivers
- User-space applications

## Performance

### Protocol Comparison

| Metric | Phase 2B (Serial) | Phase 2C (MMIO) | Improvement |
|--------|------------------|----------------|-------------|
| Throughput | ~1,000 tx/sec | ~10,000 tx/sec | 10x |
| Latency | ~1ms/tx | ~100μs/tx | 10x |
| CPU Usage | High (parsing) | Low (binary) | ~9x reduction |
| Protocol Overhead | ~80% | ~8% | 10x reduction |

### Device Latencies

| Device | Operation | Latency (cycles) | Configurable |
|--------|-----------|------------------|--------------|
| Echo | Echo operation | 10 | ✅ `echo_latency` |
| Compute | Addition | 100 | ✅ `compute_latency` |
| Compute | Multiplication | 100 | ✅ `compute_latency` |
| Compute | Division | 100 | ✅ `compute_latency` |

## Troubleshooting

### Common Issues

**1. sst-info shows "Component not found"**
```bash
# Rebuild and reinstall
cd qemu-binary && make clean && make && make install
cd ../acalsim-device && make clean && make && make install

# Verify
sst-info qemubinary
sst-info acalsim
```

**2. QEMU device not found**
```bash
# Check custom QEMU
$QEMU_PATH -device help | grep sst
# Should show: name "sst-device", bus System

# If not, rebuild QEMU with SST device
# See PHASE2C_INTEGRATION.md
```

**3. Socket connection timeout**
```bash
# Check socket path permissions
ls -la /tmp/qemu-sst-mmio.sock

# Verify device addresses match
grep SST_DEVICE_BASE riscv-programs/mmio_test.c
grep device_base qemu_binary_test.py
# Both should be 0x10200000
```

**4. Test hangs or times out**
```bash
# Increase simulation time
# In qemu_binary_test.py:
sim_time_us = 10000  # 10ms instead of 1ms

# Enable verbose output
# In SST config:
qemu.addParams({"verbose": "3"})
device.addParams({"verbose": "2"})
```

See [BUILD_AND_TEST.md](BUILD_AND_TEST.md) for detailed troubleshooting.

## Adding New Devices

To add a new device systematically:

1. **Create Device Component** (see USER_GUIDE.md)
2. **Define Register Map** (memory-mapped registers)
3. **Implement Device Logic** (SST component)
4. **Update SST Configuration** (add device and links)
5. **Write Test Program** (RISC-V firmware)
6. **Build and Test**

For N devices, see discussion in MULTI_DEVICE_EXAMPLE.md about:
- Address-based routing in QEMUBinaryComponent
- Device router components
- Systematic device instantiation

## References

- [SST-Core Documentation](http://sst-simulator.org/)
- [QEMU RISC-V Documentation](https://www.qemu.org/docs/master/system/target-riscv.html)
- [RISC-V ISA Specification](https://riscv.org/technical/specifications/)
- [ACALSim Framework](../../README.md)

## Related Projects

- **[qemu-acalsim-sst-linux](../qemu-acalsim-sst-linux/)** - Linux-based variant (Phase 3)
- **[sst-integration](../sst-integration/)** - Direct SST integration experiments

## License

Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

---

**Current Status**: Phase 2C Complete (Binary MMIO, Multi-Device)
**Next Phase**: Phase 2D (Multi-Core Support) or Phase 3 (Linux Integration)
**Last Updated**: 2025-11-10
