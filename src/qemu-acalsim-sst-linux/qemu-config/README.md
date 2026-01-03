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

# QEMU Configuration Scripts

This directory contains scripts for configuring and launching QEMU with the VirtIO SST device.

## Files

### setup-qemu-virtio-sst.sh

Automated setup script that integrates the VirtIO SST device into QEMU source tree.

**Usage**:
```bash
./setup-qemu-virtio-sst.sh <qemu-source-dir>
```

**Example**:
```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./setup-qemu-virtio-sst.sh /home/user/qemu-build/qemu
```

**What it does**:
1. Copies VirtIO SST device files (`sst-protocol.h`, `virtio-sst.h`, `virtio-sst.c`) to QEMU source
2. Updates `hw/virtio/meson.build` with correct build configuration
3. Adds `CONFIG_VIRTIO_SST` to `hw/virtio/Kconfig`
4. Verifies all files are properly installed

**After running this script**:
```bash
cd <qemu-source-dir>
mkdir -p build && cd build
../configure --target-list=riscv64-softmmu --enable-virtfs
make -j$(nproc)
```

### run-linux.sh

Launch script for running QEMU with Linux kernel and VirtIO SST device.

**Usage**:
```bash
# Set environment variables
export QEMU=/path/to/qemu-system-riscv64
export KERNEL=/path/to/Image
export INITRD=/path/to/initramfs.cpio.gz
export SOCKET=/tmp/qemu-sst-linux.sock

# Run QEMU
./run-linux.sh
```

See [GETTING_STARTED.md](../GETTING_STARTED.md) for complete setup and usage instructions.

## Requirements

- Docker container: `acalsim-workspace`
- QEMU 6.2.0 source
- RISC-V cross-compilation toolchain
- SST-Core installation

## Troubleshooting

### Setup Script Fails

**Problem**: "QEMU source directory not found"
```bash
# Solution: Verify QEMU source path
ls -la /home/user/qemu-build/qemu
```

**Problem**: "VirtIO device directory not found"
```bash
# Solution: Run from correct location
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
pwd  # Should show qemu-config directory
```

### Build Fails After Setup

**Problem**: Meson configuration error
```bash
# Solution: Clean build directory and reconfigure
cd /home/user/qemu-build/qemu
rm -rf build
mkdir build && cd build
../configure --target-list=riscv64-softmmu --enable-virtfs
```

**Problem**: Missing dependencies
```bash
# Solution: Install required packages (as root)
docker exec -u root acalsim-workspace bash -c "apt-get update && apt-get install -y libcap-ng-dev libattr1-dev"
```

## See Also

- [GETTING_STARTED.md](../GETTING_STARTED.md) - Complete setup guide
- [BUILD_NOTES.md](../BUILD_NOTES.md) - Build architecture details
- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) - Implementation details
