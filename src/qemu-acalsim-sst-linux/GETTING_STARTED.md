# Getting Started with Linux SST Integration

This guide walks you through setting up and running your first Linux-based QEMU-ACALSim-SST simulation.

## Overview

The Linux integration enables full operating system simulation with:
- Complete Linux kernel boot on RISC-V
- VirtIO device for SST communication
- Kernel driver creating `/dev/sst*` devices
- User-space applications accessing SST via standard I/O
- Realistic OS overhead modeling

## Prerequisites

### Required Software

1. **QEMU with VirtIO SST Device**
   - QEMU 7.0+ with RISC-V support
   - Custom VirtIO SST device (see Integration section)

2. **Linux Kernel**
   - RISC-V Linux kernel 5.15+
   - VirtIO support enabled
   - SST kernel driver compiled

3. **SST Core Framework**
   - SST Core 11.0+
   - SST device components built

4. **Cross-compilation Tools**
   - RISC-V GCC toolchain
   - Kernel build tools

### Installation

#### Install RISC-V Toolchain

**Ubuntu/Debian:**
```bash
sudo apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

**macOS (via Homebrew):**
```bash
brew tap riscv/riscv
brew install riscv-tools
```

#### Install QEMU

```bash
# From package manager (may lack VirtIO SST device)
sudo apt-get install qemu-system-misc

# Or build from source with VirtIO SST device
# See "Integrating VirtIO Device" section below
```

#### Install SST Core

```bash
# Follow SST installation guide
# http://sst-simulator.org/SSTPages/SSTMainDocumentation/
```

## Quick Start

### 1. Integrate VirtIO Device into QEMU

Copy the VirtIO SST device files into QEMU source:

```bash
# Set QEMU source directory
export QEMU_SRC=/path/to/qemu

# Copy device files
cp virtio-device/sst-protocol.h $QEMU_SRC/include/hw/virtio/
cp virtio-device/virtio-sst.h $QEMU_SRC/include/hw/virtio/
cp virtio-device/virtio-sst.c $QEMU_SRC/hw/virtio/

# Update build configuration
# Add to $QEMU_SRC/hw/virtio/meson.build:
echo "virtio_ss.add(when: 'CONFIG_VIRTIO_SST', if_true: files('virtio-sst.c'))" \
    >> $QEMU_SRC/hw/virtio/meson.build

# Add to $QEMU_SRC/hw/virtio/Kconfig:
cat >> $QEMU_SRC/hw/virtio/Kconfig <<EOF
config VIRTIO_SST
    bool
    default y
    depends on VIRTIO
EOF

# Rebuild QEMU
cd $QEMU_SRC/build
../configure --target-list=riscv64-softmmu
make -j$(nproc)
```

### 2. Build Linux Kernel

```bash
# Get RISC-V Linux kernel
git clone https://github.com/torvalds/linux.git
cd linux
git checkout v6.1  # Or latest stable

# Configure for RISC-V
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- menuconfig

# Enable VirtIO support:
#   Device Drivers -> Virtio drivers -> PCI driver for virtio devices
#   Device Drivers -> Block devices -> Virtio block driver

# Build kernel
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)

# Output: arch/riscv/boot/Image (uncompressed kernel)
cp arch/riscv/boot/Image ../vmlinux
```

### 3. Build Kernel Driver

```bash
cd drivers

# Build for current kernel
make KDIR=/path/to/linux

# Or cross-compile
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- \
     KDIR=/path/to/linux

# Output: virtio-sst.ko
```

### 4. Build Test Applications

```bash
cd rootfs/apps

# Cross-compile for RISC-V
make CROSS_COMPILE=riscv64-linux-gnu-

# Output: sst-test
```

### 5. Create Root Filesystem

Create a minimal root filesystem with BusyBox:

```bash
# Download and build BusyBox
wget https://busybox.net/downloads/busybox-1.36.0.tar.bz2
tar xf busybox-1.36.0.tar.bz2
cd busybox-1.36.0

# Configure for static build
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- menuconfig
# Enable: Settings -> Build static binary

# Build
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- install

# Create rootfs structure
mkdir -p rootfs
cd rootfs
mkdir -p bin sbin etc proc sys dev lib apps

# Copy BusyBox
cp -a ../_install/* .

# Copy init script
cp /path/to/qemu-acalsim-sst-linux/rootfs/init init
chmod +x init

# Copy kernel driver
cp /path/to/qemu-acalsim-sst-linux/drivers/virtio-sst.ko .

# Copy test applications
cp /path/to/qemu-acalsim-sst-linux/rootfs/apps/sst-test apps/

# Create initramfs
find . | cpio -o -H newc | gzip > ../initramfs.cpio.gz
```

### 6. Build SST Components

```bash
# Use existing ACALSim device components
cd ../../acalsim-device
make

# Verify libacalsim.so is built
ls -l libacalsim.so
```

### 7. Run Your First Simulation

**Terminal 1 - Start SST:**
```bash
cd sst-config

# Set environment
export SST_CORE_HOME=/path/to/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Run SST configuration
sst linux_basic.py
```

**Terminal 2 - Start QEMU:**
```bash
cd qemu-config

# Set paths
export QEMU=/path/to/qemu/build/qemu-system-riscv64
export KERNEL=/path/to/vmlinux
export INITRD=/path/to/initramfs.cpio.gz

# Launch QEMU
./run-linux.sh
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
# In QEMU console
/apps/sst-test

# Output:
============================================
  SST Device Test Application
============================================

Opening device: /dev/sst0
Device opened successfully (fd=3)

[TEST] NOOP Request
  Sending NOOP request...
  Status: OK
  User data: 0x1234
  PASSED

[TEST] ECHO Request
  Sending ECHO request: "Hello SST!"
  Status: OK
  Echo data: "Hello SST!"
  PASSED

...
```

## Common Workflows

### Testing Device Connectivity

```bash
# Check if device exists
ls -l /dev/sst*

# Check kernel module
lsmod | grep virtio_sst

# View kernel messages
dmesg | grep virtio-sst

# Simple test
echo "test" > /dev/sst0
```

### Running Compute Simulations

Create a custom application:

```c
#include <fcntl.h>
#include <unistd.h>
#include "sst-protocol.h"

int main() {
    int fd = open("/dev/sst0", O_RDWR);

    struct SSTRequest req = {
        .type = SST_REQ_COMPUTE,
        .payload.compute.compute_units = 1000
    };

    write(fd, &req, sizeof(req));

    struct SSTResponse resp;
    read(fd, &resp, sizeof(resp));

    printf("Simulated cycles: %lu\n",
           resp.payload.compute.cycles);

    close(fd);
    return 0;
}
```

Compile and run:

```bash
riscv64-linux-gnu-gcc -o my-app my-app.c \
    -I../../virtio-device

# Copy to rootfs and run in QEMU
```

### Multi-Device Setup

Configure multiple SST devices:

**SST configuration** (`linux_multi.py`):
```python
for i in range(4):
    dev = sst.Component(f"sst_device_{i}",
                       "acalsim.ACALSimDeviceComponent")
    dev.addParams({
        "socket_path": f"/tmp/qemu-sst-{i}.sock",
        "device_id": i
    })
```

**QEMU launch**:
```bash
$QEMU ... \
    -device virtio-sst-device,socket=/tmp/qemu-sst-0.sock,device-id=0 \
    -device virtio-sst-device,socket=/tmp/qemu-sst-1.sock,device-id=1 \
    -device virtio-sst-device,socket=/tmp/qemu-sst-2.sock,device-id=2 \
    -device virtio-sst-device,socket=/tmp/qemu-sst-3.sock,device-id=3
```

Access from Linux:
```bash
/apps/sst-test /dev/sst0  # Device 0
/apps/sst-test /dev/sst1  # Device 1
...
```

## Troubleshooting

### QEMU doesn't start

**Error**: `Failed to open socket: /tmp/qemu-sst-linux.sock`

**Solution**: Start SST first, it creates the socket server.

### Device not found in Linux

**Error**: `/dev/sst0: No such file or directory`

**Possible causes**:
1. VirtIO device not configured in QEMU
   - Check QEMU command line has `-device virtio-sst-device`
2. Kernel driver not loaded
   - Run `insmod /virtio-sst.ko`
   - Check `dmesg | grep virtio-sst`
3. SST not connected
   - Check SST is running
   - Verify socket path matches

### Request timeout

**Error**: `virtio-sst: Request timed out`

**Solution**:
- Ensure SST is processing requests
- Check SST component configuration
- Verify socket communication is working
- Check SST logs for errors

### Kernel build failures

**Error**: `virtio_device_id` undeclared

**Solution**: Make sure VirtIO support is enabled in kernel config:
```bash
make menuconfig
# Enable: Device Drivers -> Virtio drivers
```

## Next Steps

- Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for architecture details
- Explore example SST configurations in `sst-config/`
- Review VirtIO protocol in `virtio-device/sst-protocol.h`
- Study kernel driver implementation in `drivers/sst-virtio.c`
- Try bare-metal variant: `../qemu-acalsim-sst-baremetal/`
- Try HSA variant: `../qemu-acalsim-sst-baremetal-HSA/`

## Additional Resources

- [Linux VirtIO Documentation](https://www.kernel.org/doc/html/latest/driver-api/virtio/virtio.html)
- [QEMU RISC-V Documentation](https://www.qemu.org/docs/master/system/target-riscv.html)
- [SST Documentation](http://sst-simulator.org/SSTPages/SSTMainDocumentation/)
- [RISC-V Specifications](https://riscv.org/technical/specifications/)

---

**Need Help?**
- Check kernel logs: `dmesg | grep -i sst`
- Check QEMU logs: Run with `-d guest_errors,unimp`
- Check SST output for connection status
- Review component README files for detailed information
