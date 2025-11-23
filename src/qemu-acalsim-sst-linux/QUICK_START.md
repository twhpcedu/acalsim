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

# Quick Start Guide - QEMU with virtio-sst

Complete guide to build and run QEMU with virtio-sst device support on RISC-V.

## Prerequisites

- Ubuntu 22.04 (or compatible)
- Docker container: `acalsim-workspace`
- Minimum 20GB free disk space
- Minimum 4GB RAM

## One-Command Build

```bash
# Build everything (QEMU + Linux kernel)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./build-qemu-7.0.sh && ./build-linux-kernel.sh
```

## Step-by-Step Build

### Step 1: Build QEMU 7.0 with virtio-sst (~6 minutes)

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./build-qemu-7.0.sh
```

**What it does:**
- Installs dependencies (libslirp, meson, etc.)
- Clones QEMU 7.0.0
- Integrates virtio-sst device
- Builds QEMU for RISC-V
- Verifies virtio-sst-device and user networking

**Output:**
- QEMU binary: `/home/user/qemu-build/qemu/build/qemu-system-riscv64`
- Automatically added to PATH in `~/.bashrc`

### Step 2: Build Linux Kernel (~15-30 minutes)

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./build-linux-kernel.sh
```

**What it does:**
- Installs cross-compiler (`gcc-riscv64-linux-gnu`)
- Clones Linux v6.1
- Configures with VirtIO drivers enabled
- Builds kernel for RISC-V

**Output:**
- Kernel image: `/home/user/linux/arch/riscv/boot/Image`

### Step 3: Setup Rootfs

The buildroot rootfs is already built and linked:

```bash
# Verify rootfs exists
ls -lh /home/user/initramfs-buildroot.cpio.gz
# Should show: -> /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz
```

### Step 4: Boot QEMU

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

**QEMU Configuration:**
- Machine: QEMU virt
- CPU: rv64 (RISC-V 64-bit)
- CPUs: 4 cores
- RAM: 8GB
- Rootfs: Buildroot initramfs (89MB compressed)
- Network: user-mode (SLIRP) with DHCP
- SST Socket: `/tmp/qemu-sst-llama.sock`

**Expected Boot Output:**
```
VirtIO SST: Initializing device (socket=/tmp/qemu-sst-llama.sock, id=0)
VirtIO SST: Warning - Failed to connect to SST...
VirtIO SST: Device will work without SST connection
VirtIO SST: Device initialized successfully

OpenSBI v1.0
...
[Linux kernel boot messages]
...
Starting syslogd: OK
Starting klogd: OK
Running sysctl: OK
Populating /dev using udev: done
Starting network: OK
Starting dhcpcd: OK
Starting dropbear sshd: OK

Welcome to ACAL Simulator RISC-V Linux
acalsim-riscv login:
```

**Exit QEMU:** Press `Ctrl-A` then `X`

## Verification

### Verify QEMU Build

```bash
qemu-system-riscv64 --version
# Expected: QEMU emulator version 7.0.0

qemu-system-riscv64 -device help | grep virtio-sst
# Expected: name "virtio-sst-device", bus virtio-bus, desc "VirtIO SST Device"

qemu-system-riscv64 -netdev help | grep user
# Expected: user
```

### Verify Kernel Build

```bash
ls -lh /home/user/linux/arch/riscv/boot/Image
# Expected: ~10-20MB file

file /home/user/linux/arch/riscv/boot/Image
# Expected: Linux kernel ARM64 boot executable Image
```

## Troubleshooting

### QEMU Build Fails

**Error:** `Feature virtfs cannot be enabled`
```bash
# Solution: Install missing dependencies
sudo apt-get install -y libcap-ng-dev libattr1-dev
```

**Error:** `network backend 'user' is not compiled`
```bash
# Solution: Install libslirp before building
sudo apt-get install -y libslirp-dev libslirp0
rm -rf /home/user/qemu-build/qemu/build
cd /home/user/qemu-build/qemu/build
../configure --target-list=riscv64-softmmu --enable-virtfs
make -j$(nproc)
```

**Error:** `virtio-sst-device not found`
```bash
# Solution: QEMU version incompatibility
# The virtio-sst code requires QEMU 7.0.0 (not 7.1+)
# Use the build-qemu-7.0.sh script which uses the correct version
```

### Kernel Build Fails

**Error:** `riscv64-linux-gnu-gcc: command not found`
```bash
# Solution: Install cross-compiler
sudo apt-get install -y gcc-riscv64-linux-gnu
```

**Error:** `No rule to make target 'Image'`
```bash
# Solution: Make sure ARCH and CROSS_COMPILE are set
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)
```

### QEMU Boot Fails

**Error:** `could not load kernel '/home/user/linux/arch/riscv/boot/Image'`
```bash
# Solution: Kernel not built yet
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./build-linux-kernel.sh
```

**Error:** `No such file or directory: /home/user/rootfs-persistent.qcow2`
```bash
# Solution: Create or restore rootfs
# Check if rootfs backup exists
ls -lh /home/user/rootfs*.qcow2
```

## Build Times

| Component | Time | Disk Space |
|-----------|------|------------|
| QEMU 7.0 | ~6 min | ~500MB |
| Linux Kernel | 15-30 min | ~2GB |
| **Total** | **~25 min** | **~2.5GB** |

*Times measured on 8-core ARM64 system*

## File Locations

```
/home/user/
├── qemu-build/
│   └── qemu/
│       └── build/
│           └── qemu-system-riscv64                    # QEMU binary
├── linux/
│   └── arch/riscv/boot/
│       └── Image                                       # Kernel image
├── initramfs-buildroot.cpio.gz                         # Symlink to buildroot rootfs
└── buildroot-llama/buildroot-2024.02/output/images/
    └── rootfs.cpio.gz                                  # Buildroot rootfs (89MB)
```

## Environment Variables

The run scripts support these variables:

```bash
# QEMU binary (default: /home/user/qemu-build/qemu/build/qemu-system-riscv64)
export QEMU_BIN=/path/to/qemu-system-riscv64

# Kernel image (default: /home/user/linux/arch/riscv/boot/Image)
export KERNEL=/path/to/Image

# Initramfs file (default: /home/user/initramfs-buildroot.cpio.gz)
export INITRAMFS=/path/to/initramfs.cpio.gz
```

## Next Steps

After successful boot to login prompt:

1. **Login to Linux**: Use buildroot credentials (typically root with no password)
2. **Test network connectivity**: System already configured via DHCP (10.0.2.15)
3. **SSH access**: Dropbear SSH server is running
4. **Test virtio-sst device**: Check device availability
5. **Run SST simulation**: See `examples/llama-inference/README.md`

## Documentation

- **[BUILD_QEMU_SOP.md](docs/BUILD_QEMU_SOP.md)** - Detailed QEMU build guide
- **[README.md](README.md)** - Project overview
- **[examples/llama-inference/README.md](examples/llama-inference/README.md)** - LLAMA inference example

## Support

For issues:
1. Check [Troubleshooting](#troubleshooting) section above
2. Review build logs in `/home/user/qemu-build/qemu/build/meson-logs/`
3. Check kernel build with `dmesg` after boot

---

**Last Updated**: 2025-11-20
**Tested On**: Ubuntu 22.04 (jammy) in acalsim-workspace Docker container
**QEMU Version**: 7.0.0
**Kernel Version**: 6.1
