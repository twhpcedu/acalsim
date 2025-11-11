# ACALSim SST Device Components for Linux Integration

This directory builds SST device components for the Linux integration project using shared ACALSim framework source files.

## Overview

The SST device components provide the simulation backend for the VirtIO SST device, handling requests from Linux applications via the kernel driver. The source files are shared infrastructure located in the ACALSim framework:

- **SST Device Sources**: `../../../libs/sst/`
  - `ACALSimDeviceComponent.cc` - Basic SST device with socket communication
  - `ACALSimComputeDeviceComponent.cc` - Compute simulation device
  - `ACALSimMMIODevice.cc` - MMIO device with interrupt support

- **HSA Components**: `../../../libs/HSA/`
  - `HSAHostComponent.cc` - HSA host agent
  - `HSAComputeComponent.cc` - HSA compute agent

- **Headers**: `../../../include/sst/` and `../../../include/HSA/`

## Building

### Prerequisites

- SST-Core installation at `$SST_CORE_HOME`
- C++17 compatible compiler
- All dependencies from GETTING_STARTED.md installed

### Build Commands

```bash
# Set SST environment
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Build the library
make

# Verify build
ls -lh libacalsim.so

# Optional: Install to SST element library directory
make install
```

## Components

The library provides these SST components:

### ACALSimDeviceComponent
Main device component that handles VirtIO SST protocol requests from QEMU.

**Parameters**:
- `socket_path`: Unix domain socket path (e.g., `/tmp/qemu-sst-linux.sock`)
- `device_id`: Device identifier (0-based)
- `verbose`: Enable verbose logging (0 or 1)

**Usage in SST Python**:
```python
device = sst.Component("sst_device_0", "acalsim.ACALSimDeviceComponent")
device.addParams({
    "socket_path": "/tmp/qemu-sst-linux.sock",
    "device_id": 0,
    "verbose": "1"
})
```

### ACALSimComputeDeviceComponent
Compute simulation device for processing COMPUTE requests.

**Parameters**:
- `device_id`: Device identifier matching the main device
- `compute_latency`: Cycles per compute unit
- `verbose`: Enable verbose logging

**Usage**:
```python
compute = sst.Component("compute_0", "acalsim.ACALSimComputeDeviceComponent")
compute.addParams({
    "device_id": 0,
    "compute_latency": "1000",
    "verbose": "1"
})

# Link to main device
link = sst.Link("device_link_0")
link.connect(
    (device, "compute_link", "1ns"),
    (compute, "device_link", "1ns")
)
```

## Architecture

```
Linux Application (userspace)
    |
    v
/dev/sst0 (character device)
    |
    v
virtio-sst.ko (kernel driver)
    |
    v
VirtIO SST Device (QEMU)
    |
    v
Unix Domain Socket
    |
    v
ACALSimDeviceComponent (SST) <--> ACALSimComputeDeviceComponent (SST)
```

## Files

- `Makefile` - Build configuration
- `README.md` - This file
- `libacalsim.so` - Built shared library (generated)
- `*.o` - Object files (generated)

## Standalone Design

This directory is self-contained within the Linux integration project and builds against shared framework sources using relative paths. No dependencies on other QEMU-SST projects.

## Makefile Targets

- `make` or `make all` - Build the component library
- `make install` - Install to SST element library directory
- `make clean` - Remove build artifacts
- `make uninstall` - Remove from SST element library
- `make help` - Show help message

## Integration

The built `libacalsim.so` is referenced by SST Python configuration files in `../sst-config/`:
- `linux_basic.py` - Basic single device configuration
- (add more configurations as needed)

## See Also

- [GETTING_STARTED.md](../GETTING_STARTED.md) - Complete setup guide
- [BUILD_NOTES.md](../BUILD_NOTES.md) - Build architecture details
- [../sst-config/](../sst-config/) - SST Python configurations
- [../../../libs/sst/](../../../libs/sst/) - Shared SST device sources
- [../../../libs/HSA/](../../../libs/HSA/) - Shared HSA component sources
