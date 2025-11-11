# Build Notes for Linux Integration

## Container-Based Build Workflow

**All components are built inside the `acalsim-workspace` Docker container.**

This document explains the build architecture and how different components are built within the unified container environment.

## Build Environment

### Docker Container
Everything runs in: `acalsim-workspace`

```bash
# Access container
docker exec -it acalsim-workspace bash

# All subsequent commands run inside the container
```

### Required Tools (Install in Container)

```bash
# RISC-V cross-compilation toolchain
apt-get install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu

# Kernel build tools
apt-get install -y bc bison flex libssl-dev libelf-dev libncurses-dev

# QEMU build dependencies
apt-get install -y libglib2.0-dev libpixman-1-dev ninja-build

# General tools
apt-get install -y wget git make
```

## Component Build Matrix

| Component | Tool | Target Arch | Location |
|-----------|------|-------------|----------|
| **QEMU VirtIO Device** | GCC (x86_64) | x86_64 | `/home/user/qemu-build/qemu` |
| **Linux Kernel** | riscv64-linux-gnu-gcc | RISC-V 64 | `/home/user/linux` |
| **Kernel Driver** | riscv64-linux-gnu-gcc | RISC-V 64 | `/home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers` |
| **Test Applications** | riscv64-linux-gnu-gcc | RISC-V 64 | `/home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps` |
| **SST Components** | GCC (x86_64) | x86_64 | `/home/user/projects/acalsim/src/acalsim-device` |
| **BusyBox** | riscv64-linux-gnu-gcc | RISC-V 64 | `/home/user/busybox-1.36.0` |

## Build Workflow

### 1. VirtIO Device (QEMU Component)

**Architecture**: x86_64 (runs on host, simulates RISC-V)
**Compiler**: System GCC
**Build location**: Inside QEMU source tree

```bash
# In container
cd /home/user/qemu-build/qemu

# Copy source files
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/virtio-device/*.h include/hw/virtio/
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/virtio-device/virtio-sst.c hw/virtio/

# Update build configuration
echo "virtio_ss.add(when: 'CONFIG_VIRTIO_SST', if_true: files('virtio-sst.c'))" >> hw/virtio/meson.build

# Build QEMU
cd build
../configure --target-list=riscv64-softmmu
make -j$(nproc)
```

**Output**: `/home/user/qemu-build/qemu/build/qemu-system-riscv64` (x86_64 executable)

### 2. Linux Kernel

**Architecture**: RISC-V 64
**Compiler**: riscv64-linux-gnu-gcc
**Build location**: `/home/user/linux`

```bash
# In container
cd /home/user/linux

# Cross-compile for RISC-V
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)
```

**Output**: `/home/user/linux/arch/riscv/boot/Image` (RISC-V 64 kernel)

### 3. Kernel Driver (virtio-sst.ko)

**Architecture**: RISC-V 64
**Compiler**: riscv64-linux-gnu-gcc
**Build location**: `/home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers`

```bash
# In container
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers

# Build against RISC-V kernel
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- \
     KDIR=/home/user/linux
```

**Output**: `virtio-sst.ko` (RISC-V 64 kernel module)

**Why it failed before**: The Makefile defaults to `/lib/modules/$(uname -r)/build` which is the container's x86_64 kernel, not the RISC-V kernel we're building for.

**Solution**: Always specify `KDIR=/home/user/linux` to point to the RISC-V kernel source.

### 4. Test Applications

**Architecture**: RISC-V 64
**Compiler**: riscv64-linux-gnu-gcc
**Build location**: `/home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps`

```bash
# In container
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps

# Cross-compile
make CROSS_COMPILE=riscv64-linux-gnu-
```

**Output**: `sst-test` (RISC-V 64 executable)

### 5. SST Components

**Architecture**: x86_64 (simulation runs on host)
**Compiler**: System GCC/G++
**Build location**: `/home/user/projects/acalsim/src/acalsim-device`

```bash
# In container
cd /home/user/projects/acalsim/src/acalsim-device

# Set SST environment
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Build
make clean && make
```

**Output**: `libacalsim.so` (x86_64 shared library)

## Understanding the Build Failure

When you ran `make` in the `drivers/` directory, you saw:

```
make: *** /lib/modules/6.10.14-linuxkit/build: No such file or directory.  Stop.
```

### Why This Happened

1. **Makefile default**: Without `KDIR`, the Makefile uses `/lib/modules/$(uname -r)/build`
2. **Container kernel**: The container runs a linuxkit kernel (Docker's minimal kernel)
3. **Missing headers**: This kernel doesn't have development headers installed
4. **Wrong architecture**: Even if headers existed, they'd be x86_64, not RISC-V

### Correct Build Command

```bash
# Always specify KDIR when building the driver
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- \
     KDIR=/home/user/linux
```

This tells the build system:
- `ARCH=riscv`: Target RISC-V architecture
- `CROSS_COMPILE=riscv64-linux-gnu-`: Use RISC-V cross-compiler
- `KDIR=/home/user/linux`: Use our RISC-V kernel source for headers

## Quick Reference: Build All Components

Save this script in the container as `/home/user/build-all-linux.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Building All Linux Integration Components"
echo "=========================================="

# 1. Build QEMU
echo ""
echo "[1/5] Building QEMU..."
cd /home/user/qemu-build/qemu/build
ninja || make -j$(nproc)

# 2. Build Linux Kernel
echo ""
echo "[2/5] Building Linux Kernel..."
cd /home/user/linux
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)

# 3. Build Kernel Driver
echo ""
echo "[3/5] Building Kernel Driver..."
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers
make clean
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- KDIR=/home/user/linux

# 4. Build Test Applications
echo ""
echo "[4/5] Building Test Applications..."
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps
make clean
make CROSS_COMPILE=riscv64-linux-gnu-

# 5. Build SST Components
echo ""
echo "[5/5] Building SST Components..."
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

cd /home/user/projects/acalsim/src/acalsim-device
make clean && make

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "QEMU:        /home/user/qemu-build/qemu/build/qemu-system-riscv64"
echo "Kernel:      /home/user/linux/arch/riscv/boot/Image"
echo "Driver:      /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko"
echo "Test App:    /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps/sst-test"
echo "SST Library: /home/user/projects/acalsim/src/acalsim-device/libacalsim.so"
echo "=========================================="
```

Make it executable and run:

```bash
chmod +x /home/user/build-all-linux.sh
/home/user/build-all-linux.sh
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│  acalsim-workspace Docker Container (x86_64)        │
│                                                      │
│  ┌────────────────┐  ┌──────────────────┐          │
│  │  QEMU          │  │  SST Simulator   │          │
│  │  (x86_64)      │  │  (x86_64)        │          │
│  │                │  │                  │          │
│  │  Runs RISC-V   │  │  Cycle-accurate  │          │
│  │  Linux guest   │  │  modeling        │          │
│  └────────┬───────┘  └────────┬─────────┘          │
│           │                    │                     │
│           └──── Unix Socket ───┘                     │
│                                                      │
│  ┌──────────────────────────────────────────┐      │
│  │  Cross-compiled RISC-V Components:       │      │
│  │  - Linux Kernel (Image)                  │      │
│  │  - Kernel Driver (virtio-sst.ko)         │      │
│  │  - Test Applications (sst-test)          │      │
│  │  - Root Filesystem (initramfs.cpio.gz)   │      │
│  └──────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

## File Verification

After building, verify architectures:

```bash
# QEMU should be x86_64
file /home/user/qemu-build/qemu/build/qemu-system-riscv64
# Output: ELF 64-bit LSB executable, x86-64

# Kernel should be RISC-V
file /home/user/linux/arch/riscv/boot/Image
# Output: data (or RISC-V 64-bit)

# Kernel module should be RISC-V
file /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko
# Output: ELF 64-bit LSB relocatable, RISC-V

# Test app should be RISC-V
file /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps/sst-test
# Output: ELF 64-bit LSB executable, RISC-V

# SST library should be x86_64
file /home/user/projects/acalsim/src/acalsim-device/libacalsim.so
# Output: ELF 64-bit LSB shared object, x86-64
```

## Troubleshooting

### Problem: Cannot find riscv64-linux-gnu-gcc

**Solution**: Install cross-compiler in container:
```bash
apt-get update
apt-get install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

### Problem: Kernel build fails with missing dependencies

**Solution**: Install kernel build tools:
```bash
apt-get install -y bc bison flex libssl-dev libelf-dev libncurses-dev
```

### Problem: QEMU build fails

**Solution**: Install QEMU dependencies:
```bash
apt-get install -y libglib2.0-dev libpixman-1-dev ninja-build
```

### Problem: Permission denied creating device nodes

**Solution**: Run with sudo or use fakeroot:
```bash
# When creating initramfs
fakeroot sh -c '
    cd /home/user/rootfs
    mknod -m 666 dev/null c 1 3
    mknod -m 666 dev/console c 5 1
    find . | cpio -o -H newc | gzip > /home/user/initramfs.cpio.gz
'
```

## Development Workflow

### Modify and Rebuild Kernel Driver

```bash
# Edit driver
vim /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers/sst-virtio.c

# Rebuild
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- KDIR=/home/user/linux

# Rebuild initramfs with new driver
cd /home/user/rootfs
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko .
find . | cpio -o -H newc | gzip > /home/user/initramfs.cpio.gz

# Restart QEMU (in Terminal 2)
```

### Modify and Rebuild Test Application

```bash
# Edit app
vim /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps/sst-test.c

# Rebuild
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps
make CROSS_COMPILE=riscv64-linux-gnu-

# Rebuild initramfs
cd /home/user/rootfs
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps/sst-test apps/
find . | cpio -o -H newc | gzip > /home/user/initramfs.cpio.gz

# Restart QEMU
```

## Summary

- **All builds happen in the container**: No host dependencies needed
- **Two architectures**: x86_64 for QEMU/SST, RISC-V for guest OS
- **Key requirement**: Always specify `KDIR` when building kernel driver
- **Development cycle**: Edit → Build → Update initramfs → Test in QEMU

See [GETTING_STARTED.md](GETTING_STARTED.md) for complete setup and usage instructions.
