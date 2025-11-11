# Getting Started with Linux SST Integration

This guide walks you through setting up and running your first Linux-based QEMU-ACALSim-SST simulation **entirely within the acalsim-workspace Docker container**.

## Overview

The Linux integration enables full operating system simulation with:
- Complete Linux kernel boot on RISC-V
- VirtIO device for SST communication
- Kernel driver creating `/dev/sst*` devices
- User-space applications accessing SST via standard I/O
- Realistic OS overhead modeling

**All components are built and run inside the `acalsim-workspace` Docker container.**

## Prerequisites

### Docker Container Setup

Ensure the `acalsim-workspace` container is running:

```bash
# Check if container is running
docker ps | grep acalsim-workspace

# If not running, start it
docker start acalsim-workspace

# Access the container
docker exec -it acalsim-workspace bash
```

### Install Required Tools in Container

All tools are installed **inside the Docker container**:

```bash
# Enter the container
docker exec -it acalsim-workspace bash

# Install RISC-V cross-compilation toolchain
sudo apt-get update
sudo apt-get install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu \
                   binutils-riscv64-linux-gnu

# Install kernel build tools
sudo apt-get install -y bc bison flex libssl-dev libelf-dev \
                   libncurses-dev

# Install QEMU build dependencies (including VirtFS support)
sudo apt-get install -y libglib2.0-dev libpixman-1-dev ninja-build \
                   libcap-ng-dev libattr1-dev

# Install rootfs build tools
sudo apt-get install -y cpio

# Verify toolchain
riscv64-linux-gnu-gcc --version
```

## Quick Start

All commands below are run **inside the container**:

```bash
docker exec -it acalsim-workspace bash
```

### 1. Build QEMU with VirtIO SST Device

**Automated Setup (Recommended)**:

```bash
# Run the setup script to integrate VirtIO SST device
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./setup-qemu-virtio-sst.sh /home/user/qemu-build/qemu

# Configure and build QEMU
cd /home/user/qemu-build/qemu
mkdir -p build && cd build
../configure --target-list=riscv64-softmmu --enable-virtfs
make -j$(nproc)

# Verify QEMU built successfully
./qemu-system-riscv64 --version
```

The setup script automatically:
- Copies VirtIO SST device files to QEMU source tree
- Updates `hw/virtio/meson.build` with proper ordering
- Adds `CONFIG_VIRTIO_SST` to `hw/virtio/Kconfig`
- Verifies all files are correctly installed

<details>
<summary>Manual Setup (Alternative)</summary>

If you prefer manual setup or need to troubleshoot:

```bash
cd /home/user/qemu-build/qemu

# Copy VirtIO SST device files
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/virtio-device/sst-protocol.h \
   include/hw/virtio/
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/virtio-device/virtio-sst.h \
   include/hw/virtio/
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/virtio-device/virtio-sst.c \
   hw/virtio/

# Update meson.build - IMPORTANT: Must add BEFORE specific_ss.add_all line
# Find the line: specific_ss.add_all(when: 'CONFIG_VIRTIO', if_true: virtio_ss)
# Insert BEFORE it: virtio_ss.add(when: 'CONFIG_VIRTIO_SST', if_true: files('virtio-sst.c'))
sed -i "/^specific_ss.add_all.*CONFIG_VIRTIO.*virtio_ss/i virtio_ss.add(when: 'CONFIG_VIRTIO_SST', if_true: files('virtio-sst.c'))" \
    hw/virtio/meson.build

# Add to Kconfig
cat >> hw/virtio/Kconfig <<EOF

config VIRTIO_SST
    bool
    default y
    depends on VIRTIO
EOF

# Configure and build
mkdir -p build && cd build
../configure --target-list=riscv64-softmmu --enable-virtfs
make -j$(nproc)
```

</details>

### 2. Build RISC-V Linux Kernel

```bash
cd /home/user

# Clone Linux kernel (if not already present)
if [ ! -d linux ]; then
    git clone --depth 1 --branch v6.1 https://github.com/torvalds/linux.git
fi

cd linux

# Configure for RISC-V with VirtIO
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig

# Enable VirtIO devices
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- menuconfig
# Or use sed to enable required options:
sed -i 's/# CONFIG_VIRTIO_PCI is not set/CONFIG_VIRTIO_PCI=y/' .config
sed -i 's/# CONFIG_VIRTIO_BLK is not set/CONFIG_VIRTIO_BLK=y/' .config

# Build kernel
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)

# Kernel image is at: arch/riscv/boot/Image
ls -lh arch/riscv/boot/Image
```

### 3. Build Kernel Driver

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers

# Build against the kernel we just built
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- \
     KDIR=/home/user/linux

# Output: virtio-sst.ko
ls -lh virtio-sst.ko
```

### 4. Build Test Applications

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps

# Cross-compile for RISC-V
make CROSS_COMPILE=riscv64-linux-gnu-

# Output: sst-test (RISC-V binary)
file sst-test  # Should show: RISC-V 64-bit LSB executable
```

### 5. Build Root Filesystem

```bash
cd /home/user

# Download and build BusyBox
wget https://busybox.net/downloads/busybox-1.36.0.tar.bz2
tar xf busybox-1.36.0.tar.bz2
cd busybox-1.36.0

# Configure for static build
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig
# Enable static build
sed -i 's/# CONFIG_STATIC is not set/CONFIG_STATIC=y/' .config

# Build BusyBox
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- install

# Create rootfs structure
mkdir -p /home/user/rootfs
cd /home/user/rootfs
cp -a /home/user/busybox-1.36.0/_install/* .

# Create directory structure
mkdir -p bin sbin etc proc sys dev lib apps

# Copy init script
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/init .
chmod +x init

# Copy kernel driver
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko .

# Copy test applications
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps/sst-test apps/
chmod +x apps/sst-test
```

**Create device nodes** (requires root privileges, run this from host or as root in container):
```bash
# From host machine:
docker exec -u root acalsim-workspace bash -c "cd /home/user/rootfs && mknod -m 666 dev/null c 1 3 && mknod -m 666 dev/console c 5 1"

# OR inside container as root:
# sudo mknod -m 666 dev/null c 1 3
# sudo mknod -m 666 dev/console c 5 1
```

**Create initramfs** (back as regular user):
```bash
cd /home/user/rootfs
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz

echo "Initramfs created: /home/user/initramfs.cpio.gz"
ls -lh /home/user/initramfs.cpio.gz
```

### 6. Build SST Components

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/acalsim-device

# Set SST environment
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Build SST device components
make clean && make

# Verify library built
ls -lh libacalsim.so
```

**Note**: The SST device components are built locally but use shared ACALSim framework source files from `../../../libs/sst/` and `../../../libs/HSA/`. See `acalsim-device/README.md` for details.

### 7. Run Your First Simulation

**Terminal 1 - Start SST** (in container):
```bash
docker exec -it acalsim-workspace bash

cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/sst-config

# Set environment
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Run SST configuration
sst linux_basic.py
```

**Terminal 2 - Start QEMU** (in same container):
```bash
docker exec -it acalsim-workspace bash

cd /home/user

# Set paths
export QEMU=/home/user/qemu-build/qemu/build/qemu-system-riscv64
export KERNEL=/home/user/linux/arch/riscv/boot/Image
export INITRD=/home/user/initramfs.cpio.gz
export SOCKET=/tmp/qemu-sst-linux.sock

# Launch QEMU
$QEMU \
    -machine virt \
    -cpu rv64 \
    -m 2G \
    -smp 4 \
    -nographic \
    -kernel $KERNEL \
    -initrd $INITRD \
    -append "console=ttyS0 earlycon=sbi" \
    -device virtio-sst-device,socket=$SOCKET,device-id=0 \
    -serial mon:stdio
```

### 8. Test SST Device

Once Linux boots in QEMU, you should see:

```
============================================
  ACALSim Linux SST Integration
  Initializing system...
============================================
Mounting filesystems...
Loading virtio-sst kernel module...
Waiting for SST device...
SST device found: /dev/sst0
...
```

Run the test application:

```bash
# In QEMU console (Terminal 2)
/apps/sst-test

# Output:
============================================
  SST Device Test Application
============================================

Opening device: /dev/sst0
Device opened successfully (fd=3)

[TEST] NOOP Request
  Status: OK
  PASSED

[TEST] ECHO Request
  Status: OK
  Echo data: "Hello SST!"
  PASSED

...

Test Summary: 5/5 PASSED
```

## Common Workflows

### Quick Script for Container Setup

Save this as `setup-container.sh` on your host:

```bash
#!/bin/bash
# Setup script - Run on HOST

docker exec acalsim-workspace bash -c "
    # Install toolchains
    apt-get update && apt-get install -y \
        gcc-riscv64-linux-gnu \
        g++-riscv64-linux-gnu \
        binutils-riscv64-linux-gnu \
        bc bison flex libssl-dev libelf-dev libncurses-dev \
        libglib2.0-dev libpixman-1-dev ninja-build \
        libcap-ng-dev libattr1-dev wget cpio

    echo 'Toolchains installed successfully'
    riscv64-linux-gnu-gcc --version
"
```

### Quick Build Script

Save this as `build-all.sh` in the container:

```bash
#!/bin/bash
# Build script - Run INSIDE container

set -e

echo "Building all components..."

# Build QEMU
echo "1. Building QEMU..."
cd /home/user/qemu-build/qemu/build
ninja

# Build Linux kernel
echo "2. Building Linux kernel..."
cd /home/user/linux
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)

# Build kernel driver
echo "3. Building kernel driver..."
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- \
     KDIR=/home/user/linux clean all

# Build test apps
echo "4. Building test applications..."
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps
make CROSS_COMPILE=riscv64-linux-gnu- clean all

# Build SST components
echo "5. Building SST components..."
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/acalsim-device
make clean && make

echo "All components built successfully!"
```

### Testing Device Connectivity

```bash
# Inside QEMU Linux
ls -l /dev/sst*
lsmod | grep virtio_sst
dmesg | grep virtio-sst
```

### Running Custom Applications

Create app in container:

```bash
# On host, edit file
cat > /path/to/acalsim-workspace/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps/my-app.c <<'EOF'
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include "../../virtio-device/sst-protocol.h"

int main() {
    int fd = open("/dev/sst0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct SSTRequest req = {
        .type = SST_REQ_COMPUTE,
        .payload.compute.compute_units = 1000
    };

    write(fd, &req, sizeof(req));

    struct SSTResponse resp;
    read(fd, &resp, sizeof(resp));

    printf("Simulated cycles: %lu\n", resp.payload.compute.cycles);

    close(fd);
    return 0;
}
EOF

# Build in container
docker exec acalsim-workspace bash -c "
    cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/rootfs/apps
    riscv64-linux-gnu-gcc -o my-app my-app.c -I../../virtio-device
    file my-app
"

# Rebuild initramfs with new app
# (Follow step 5 again)
```

## Troubleshooting

### Issue: QEMU doesn't start

**Error**: `Failed to open socket: /tmp/qemu-sst-linux.sock`

**Solution**: Start SST first in Terminal 1, then QEMU in Terminal 2.

### Issue: Device not found in Linux

**Check these in order:**

1. **Is VirtIO device configured?**
   ```bash
   # In QEMU command, verify:
   -device virtio-sst-device,socket=/tmp/qemu-sst-linux.sock
   ```

2. **Is kernel driver loaded?**
   ```bash
   # In QEMU Linux
   lsmod | grep virtio_sst
   # If not loaded:
   insmod /virtio-sst.ko
   ```

3. **Is SST connected?**
   ```bash
   # Check SST terminal for "Connected" message
   ```

### Issue: Kernel module won't load

**Error**: `Unknown symbol in module`

**Solution**: Rebuild driver against exact kernel you're running:
```bash
# In container
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers
make clean
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- KDIR=/home/user/linux
```

### Issue: Can't access container

```bash
# Check if container is running
docker ps | grep acalsim-workspace

# If not running
docker start acalsim-workspace

# If container doesn't exist
# Recreate it following your container setup instructions
```

## Next Steps

- Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for architecture details
- Explore SST configurations in `sst-config/`
- Study VirtIO protocol in `virtio-device/sst-protocol.h`
- Try modifying test applications
- Experiment with multi-device setups
- Compare with bare-metal variant: `../qemu-acalsim-sst-baremetal/`

## Container Management Tips

```bash
# Save container state (if you made significant changes)
docker commit acalsim-workspace acalsim-workspace:linux-ready

# Check container disk usage
docker exec acalsim-workspace df -h

# Clean up build artifacts
docker exec acalsim-workspace bash -c "
    cd /home/user/linux && make clean
    cd /home/user/qemu-build/qemu/build && ninja clean
"
```

---

**All commands in this guide are run inside the `acalsim-workspace` Docker container** unless explicitly marked as "on host".
