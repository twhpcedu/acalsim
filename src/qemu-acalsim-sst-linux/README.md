# QEMU-ACALSim-SST Linux Integration (Phase 3)

This directory is reserved for Phase 3 development: Linux-based QEMU-SST integration.

## Overview

Phase 3 will extend the QEMU-SST integration to support full Linux system simulation:

- **Phase 3A**: Basic Linux boot with QEMU
- **Phase 3B**: VirtIO device integration for SST communication
- **Phase 3C**: Linux kernel module for SST device driver
- **Phase 3D**: User-space applications communicating with SST

## Differences from Bare-Metal Variant

The bare-metal variant (`qemu-acalsim-sst-baremetal`) provides:
- Custom firmware (crt0.S) for CPU initialization
- Direct hardware access via MMIO
- No operating system overhead
- Simple test programs in C and assembly

The Linux variant (`qemu-acalsim-sst-linux`) will provide:
- Full Linux operating system
- Standard kernel drivers and device tree
- User-space application support
- System call interface
- Multi-process/multi-threaded support

## Current Status

**Status**: Placeholder directory (Phase 3 not yet started)

**Prerequisites**:
- Complete Phase 2 (bare-metal MMIO integration) ✅
- Build custom QEMU with SST device ✅
- Validate binary MMIO protocol ✅

**Next Steps**:
1. Obtain RISC-V Linux kernel source
2. Configure kernel for QEMU virt machine
3. Add device tree entry for SST device
4. Develop kernel driver for SST device
5. Create root filesystem with test applications
6. Integrate with SST simulation

## Directory Structure (Planned)

```
qemu-acalsim-sst-linux/
├── README.md              # This file
├── kernel/                # Linux kernel configuration
│   ├── config             # Kernel .config for RISC-V
│   ├── device-tree/       # Device tree overlays
│   └── patches/           # Kernel patches if needed
├── drivers/               # SST device drivers
│   ├── sst-device.c       # Kernel module for SST device
│   └── Makefile
├── rootfs/                # Root filesystem
│   ├── init               # Init script
│   └── apps/              # Test applications
├── qemu-config/           # QEMU configuration
│   └── run-linux.sh       # Script to launch QEMU with Linux
└── sst-config/            # SST configuration
    └── linux_test.py      # SST Python configuration for Linux
```

## Related Documentation

- **Bare-Metal Integration**: See `../qemu-acalsim-sst-baremetal/`
- **Phase 2 Design**: `../qemu-acalsim-sst-baremetal/PHASE2C_DESIGN.md`
- **Build Instructions**: `../qemu-acalsim-sst-baremetal/BUILD_AND_TEST.md`

## References

- RISC-V Linux: https://github.com/riscv/riscv-linux
- QEMU RISC-V: https://www.qemu.org/docs/master/system/target-riscv.html
- Device Tree Specification: https://www.devicetree.org/
- Linux Device Drivers: https://lwn.net/Kernel/LDD3/

---

**Last Updated**: 2025-11-10
**Phase**: 3 (Planning)
**Status**: Not Started
