# QEMU-ACALSim-SST Linux Integration

Full Linux kernel boot with VirtIO device integration for SST communication. This variant enables realistic system simulation with kernel drivers, user-space applications, and standard Linux interfaces.

## Overview

This directory contains the Linux-based QEMU-SST integration, providing:

- **Full Linux Boot**: Complete Linux operating system running on QEMU RISC-V
- **VirtIO SST Device**: Standard VirtIO device for SST communication
- **Kernel Driver**: Linux kernel module for SST device access
- **User-Space Support**: Standard applications communicating through system calls
- **Multi-Process/Thread**: Full OS-level concurrency and isolation
- **Realistic Modeling**: System call overhead and OS scheduling effects

## Key Features

### Linux Kernel Integration
- Custom device tree for SST device declaration
- VirtIO transport for efficient guest-host communication
- Kernel driver implementing standard character device interface
- Support for multiple processes accessing SST devices

### VirtIO Protocol
- Standard VirtIO queues for bidirectional communication
- Efficient zero-copy data transfer where possible
- Interrupt-driven or polling modes
- Compatible with Linux VirtIO framework

### Application Interface
- Standard `/dev/sst*` character devices
- POSIX I/O operations (open, read, write, ioctl)
- Multi-threaded application support
- Event notification via poll/select

## Directory Structure

```
qemu-acalsim-sst-linux/
├── README.md              # This file
├── GETTING_STARTED.md     # Setup and first simulation guide
├── DEVELOPER_GUIDE.md     # Architecture and development details
├── kernel/                # Linux kernel configuration
│   ├── config             # Kernel .config for RISC-V
│   ├── device-tree/       # Device tree overlays for SST device
│   └── patches/           # Optional kernel patches
├── drivers/               # SST device drivers
│   ├── sst-virtio.c       # VirtIO SST kernel driver
│   ├── sst-device.h       # Driver header definitions
│   └── Makefile           # Kernel module build
├── rootfs/                # Root filesystem contents
│   ├── init               # Init script for simulation
│   └── apps/              # Test applications
│       ├── sst-test.c     # Basic SST device test
│       ├── multi-proc.c   # Multi-process test
│       └── Makefile
├── virtio-device/         # QEMU VirtIO device implementation
│   ├── virtio-sst.c       # VirtIO SST device for QEMU
│   ├── virtio-sst.h       # Device header
│   └── sst-protocol.h     # SST communication protocol
├── qemu-config/           # QEMU launch configuration
│   ├── run-linux.sh       # Launch script with all parameters
│   └── devicetree.dts     # Device tree source
└── sst-config/            # SST simulation configuration
    ├── linux_basic.py     # Single device test
    ├── linux_multi.py     # Multiple device test
    └── components/        # SST component implementations
```

## Quick Start

### Prerequisites
- QEMU with VirtIO SST device support
- RISC-V Linux kernel (5.15+)
- SST Core framework
- RISC-V toolchain for kernel and applications

### Build and Run
```bash
# Build kernel driver
cd drivers && make

# Build test applications
cd rootfs/apps && make

# Run simulation
cd qemu-config && ./run-linux.sh
```

For detailed instructions, see [GETTING_STARTED.md](GETTING_STARTED.md).

## Differences from Bare-Metal Variants

### Bare-Metal Integration (`qemu-acalsim-sst-baremetal`)
- Custom firmware (crt0.S) for CPU initialization
- Direct hardware access via MMIO
- No operating system overhead
- Simple test programs in C and assembly
- Fixed memory layout and device addresses

### Linux Integration (this variant)
- Full Linux operating system with scheduler
- Standard kernel drivers and device tree
- User-space application support through system calls
- Process isolation and virtual memory
- Dynamic device discovery and management
- Realistic OS overhead modeling

### HSA Protocol (`qemu-acalsim-sst-baremetal-HSA`)
- Specialized for heterogeneous compute modeling
- AQL packet-based job dispatch
- User-mode queue abstraction
- GPU/accelerator kernel execution modeling

## Use Cases

This Linux integration is ideal for:

1. **System-Level Simulation**: Modeling complete software stacks with OS effects
2. **Driver Development**: Testing kernel drivers in simulated hardware
3. **Application Validation**: Running real applications with SST device interaction
4. **Performance Analysis**: Understanding OS overhead in accelerated systems
5. **Multi-Process Scenarios**: Simulating concurrent device access patterns

## Getting Started

For first-time users:
1. See [GETTING_STARTED.md](GETTING_STARTED.md) for setup instructions
2. Review the example applications in `rootfs/apps/`
3. Try the basic simulation: `./qemu-config/run-linux.sh`
4. Examine SST configuration in `sst-config/linux_basic.py`

For developers:
1. Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for architecture details
2. Study the VirtIO device implementation in `virtio-device/`
3. Review the kernel driver in `drivers/sst-virtio.c`
4. See the protocol definition in `virtio-device/sst-protocol.h`

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Complete setup and usage guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: From-scratch architecture tutorial
- **[APP_DEVELOPMENT.md](APP_DEVELOPMENT.md)**: Writing Linux applications for SST devices
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**: Extending the kernel driver and QEMU device
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Single-server vs multi-server deployment
- **[ROOTFS_MANAGEMENT.md](ROOTFS_MANAGEMENT.md)**: Installing packages and managing the filesystem
- **[BUILD_NOTES.md](BUILD_NOTES.md)**: Docker vs cross-compile workflows

## Related Projects

For other simulation modes:
- **Bare-Metal Integration**: See `../qemu-acalsim-sst-baremetal/`
- **HSA Protocol**: See `../qemu-acalsim-sst-baremetal-HSA/`

## References

- [RISC-V Linux](https://github.com/riscv/riscv-linux)
- [QEMU RISC-V Documentation](https://www.qemu.org/docs/master/system/target-riscv.html)
- [VirtIO Specification](https://docs.oasis-open.org/virtio/virtio/v1.1/virtio-v1.1.html)
- [Linux Device Drivers](https://lwn.net/Kernel/LDD3/)
- [Device Tree Specification](https://www.devicetree.org/)

---

**Last Updated**: 2025-11-11
