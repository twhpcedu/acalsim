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

# Debian RISC-V Setup Guide for QEMU

## Overview

This guide shows how to switch from Buildroot to Debian RISC-V for PyTorch development.

## Prerequisites

- ✅ QEMU 7.0.0 with virtio-sst (already built)
- ✅ Linux kernel 6.18.0-rc6 (already built)
- ✅ Docker container with internet access
- 📦 ~20GB free disk space
- ⏱️ ~2-3 hours for initial setup

---

## Method 1: Using Pre-built Debian RISC-V (Recommended)

### Step 1: Download Debian RISC-V Root Filesystem

```bash
# Enter Docker container
docker exec -it acalsim-workspace bash

# Create working directory
mkdir -p /home/user/debian-riscv
cd /home/user/debian-riscv

# Download Debian sid (unstable) RISC-V rootfs
# Note: Debian has the most up-to-date RISC-V support
wget https://gitlab.com/api/v4/projects/giomasce%2Fdqib/jobs/artifacts/master/download?job=convert_riscv64-virt -O artifacts.zip

# Or use this alternative source:
wget http://ftp.ports.debian.org/debian-ports/dists/sid/main/installer-riscv64/current/images/netboot/mini.iso

# Or build from debootstrap (shown below)
```

**Note**: As of 2025, Debian RISC-V support is still in the ports repository. Let me provide a more reliable method using debootstrap.

---

## Method 2: Build Debian RISC-V Using debootstrap (More Reliable)

### Step 1: Install debootstrap and QEMU User Emulation

```bash
# Inside Docker container
sudo apt-get update
sudo apt-get install -y debootstrap qemu-user-static binfmt-support debian-ports-archive-keyring

# Verify RISC-V binfmt support
ls /proc/sys/fs/binfmt_misc/qemu-riscv64
```

### Step 2: Create Debian RISC-V Root Filesystem

```bash
# Create directory for Debian rootfs
mkdir -p /home/user/debian-riscv/rootfs
cd /home/user/debian-riscv

# Use debootstrap to create Debian sid RISC-V rootfs
# This downloads and installs packages (~2GB download)
sudo debootstrap --arch=riscv64 \
    --include=systemd,udev,kmod,iproute2,iputils-ping,net-tools,openssh-server,wget,curl,ca-certificates,vim,git,build-essential,python3,python3-pip,python3-dev,cmake,ninja-build \
    --keyring=/usr/share/keyrings/debian-ports-archive-keyring.gpg \
    sid \
    rootfs \
    http://deb.debian.org/debian-ports

# This will take 20-40 minutes depending on network speed
```

### Step 3: Configure the Debian System

```bash
# Chroot into the new system (using QEMU user emulation)
sudo chroot rootfs /bin/bash

# Inside chroot:

# Set root password
passwd
# Enter password: root (or your choice)

# Set hostname
echo "acalsim-debian-riscv" > /etc/hostname

# Configure network (DHCP)
cat > /etc/network/interfaces << 'EOF'
auto lo
iface lo inet loopback

auto eth0
iface eth0 inet dhcp
EOF

# Enable systemd services
systemctl enable systemd-networkd
systemctl enable ssh

# Create a user (optional)
useradd -m -s /bin/bash user
passwd user
# Enter password for user

# Add user to sudo group
apt-get install -y sudo
usermod -aG sudo user

# Exit chroot
exit
```

### Step 4: Create QCOW2 Disk Image from Rootfs

```bash
# Create 20GB disk image
qemu-img create -f qcow2 /home/user/debian-riscv64.qcow2 20G

# Format and populate the disk
# We'll use a helper script

cat > /home/user/debian-riscv/create-disk.sh << 'EOFSCRIPT'
#!/bin/bash
set -e

ROOTFS_DIR="/home/user/debian-riscv/rootfs"
DISK_IMAGE="/home/user/debian-riscv64.qcow2"
TEMP_DIR="/tmp/debian-disk-mount"

echo "Creating disk image..."

# Load NBD kernel module
sudo modprobe nbd max_part=8 || echo "NBD module already loaded or not available"

# Connect qcow2 image to NBD device
sudo qemu-nbd --connect=/dev/nbd0 "$DISK_IMAGE" || {
    echo "ERROR: Failed to connect NBD. Trying alternative method..."
    # Alternative: Use loop device method (see below)
    exit 1
}

# Create partition table
echo "Creating partition..."
sudo parted /dev/nbd0 mklabel gpt
sudo parted /dev/nbd0 mkpart primary ext4 1MiB 100%

# Wait for partition to appear
sleep 2

# Format partition
echo "Formatting partition..."
sudo mkfs.ext4 /dev/nbd0p1

# Mount partition
mkdir -p "$TEMP_DIR"
sudo mount /dev/nbd0p1 "$TEMP_DIR"

# Copy rootfs
echo "Copying rootfs (this may take several minutes)..."
sudo cp -a "$ROOTFS_DIR"/* "$TEMP_DIR/"

# Sync and unmount
sync
sudo umount "$TEMP_DIR"
sudo qemu-nbd --disconnect /dev/nbd0

echo "✓ Disk image created successfully: $DISK_IMAGE"
EOFSCRIPT

chmod +x /home/user/debian-riscv/create-disk.sh

# Run the script
/home/user/debian-riscv/create-disk.sh
```

---

## Alternative Method: Direct TAR Archive (If NBD Fails)

If NBD doesn't work in your Docker container, use this simpler method:

```bash
# Create tar archive of rootfs
cd /home/user/debian-riscv
sudo tar czf debian-riscv64-rootfs.tar.gz -C rootfs .

# Then boot using initramfs method (similar to buildroot)
# Or use virtio-9p to share the directory
```

---

## Step 5: Create Boot Script for Debian

```bash
cat > /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/run_qemu_debian.sh << 'EOF'
#!/bin/bash
# Boot Debian RISC-V in QEMU

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
DEBIAN_DISK=${DEBIAN_DISK:-/home/user/debian-riscv64.qcow2}
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

echo "Booting Debian RISC-V..."
echo "Press Ctrl-A then X to exit QEMU"
echo ""

exec $QEMU_BIN \
    -M virt \
    -cpu rv64 \
    -smp 4 \
    -m 8G \
    -kernel "$KERNEL" \
    -append "console=ttyS0 root=/dev/vda1 rw rootwait" \
    -drive file="$DEBIAN_DISK",if=none,id=rootfs,format=qcow2 \
    -device virtio-blk-device,drive=rootfs \
    -netdev user,id=net0 \
    -device virtio-net-device,netdev=net0 \
    -device virtio-sst-device,socket=$SOCKET_PATH \
    -nographic
EOF

chmod +x /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/run_qemu_debian.sh
```

---

## Step 6: Boot Debian

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian.sh
```

Expected output:
```
[    0.000000] Booting Linux on hartid 0
...
Debian GNU/Linux bookworm/sid acalsim-debian-riscv ttyS0

acalsim-debian-riscv login: root
Password: [enter your password]

root@acalsim-debian-riscv:~#
```

---

## Step 7: Install PyTorch Dependencies in Debian

Once logged into Debian:

```bash
# Update package lists
apt update
apt upgrade -y

# Install development tools
apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop

# Install Python and development packages
apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-yaml \
    python3-setuptools \
    python3-wheel

# Install BLAS/LAPACK
apt install -y \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev

# Install additional build dependencies
apt install -y \
    libffi-dev \
    libssl-dev \
    ninja-build \
    patchelf \
    ccache

# Verify Python
python3 --version
pip3 --version

# Install Python packages for PyTorch
pip3 install typing-extensions pyyaml cffi numpy
```

---

## Step 8: Build PyTorch in Debian

```bash
# Clone PyTorch
cd /root
git clone --depth 1 --recursive https://github.com/pytorch/pytorch
cd pytorch

# Set environment variables for minimal build
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0
export MAX_JOBS=4

# Build and install PyTorch (4-8 hours on RISC-V)
python3 setup.py install

# Test PyTorch
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

---

## Comparison: Debian vs Buildroot

| Aspect | Debian | Buildroot |
|--------|--------|-----------|
| **Setup Time** | 2-3 hours | 2-4 hours (already done) |
| **Rootfs Size** | 2-5GB | 500MB |
| **Package Manager** | apt (easy) | Limited |
| **PyTorch Build** | Easier (more packages) | Possible (with updates) |
| **Boot Time** | Slower | Faster |
| **Disk Space** | 20GB+ | 10GB |
| **Updates** | Easy (apt) | Rebuild needed |

---

## Troubleshooting

### NBD Module Not Available in Docker

```bash
# Use alternative: Create rootfs as initramfs
cd /home/user/debian-riscv
sudo tar czf debian-initramfs.cpio.gz -C rootfs .

# Boot with initramfs instead of disk
# Modify run_qemu_debian.sh to use:
#   -initrd /home/user/debian-riscv/debian-initramfs.cpio.gz
#   -append "console=ttyS0"
```

### Debootstrap Fails with GPG Errors

```bash
# Install Debian ports keyring
sudo apt-get install -y debian-ports-archive-keyring

# Use --no-check-gpg (less secure, but works)
sudo debootstrap --no-check-gpg --arch=riscv64 sid rootfs http://deb.debian.org/debian-ports
```

### QEMU Won't Boot Debian

```bash
# Check kernel has EXT4 support
grep CONFIG_EXT4 /home/user/linux/.config

# Verify disk image
qemu-img info /home/user/debian-riscv64.qcow2

# Try with more verbose output
# Add to boot command: -append "console=ttyS0 root=/dev/vda1 rw rootwait debug"
```

---

## Quick Commands Summary

```bash
# Full Debian setup (all-in-one)
docker exec -it acalsim-workspace bash -c "
cd /home/user && \
sudo apt-get install -y debootstrap qemu-user-static && \
mkdir -p debian-riscv && \
cd debian-riscv && \
sudo debootstrap --arch=riscv64 --include=systemd,python3,python3-pip,git,build-essential sid rootfs http://deb.debian.org/debian-ports && \
sudo tar czf /home/user/debian-initramfs.cpio.gz -C rootfs . && \
echo 'Debian rootfs created'
"

# Boot Debian
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian.sh
```

---

## Recommendation

**Before switching to Debian:**

1. Try rebuilding buildroot with PyTorch support (I already added packages)
2. Attempt PyTorch build on buildroot first
3. Only switch to Debian if buildroot PyTorch build fails

**Debian is better if:**
- You need frequent package installations
- You want easier dependency management
- Pre-built RISC-V packages are available
- You don't mind larger rootfs size

---

**Created**: 2025-11-20
**Tested**: Debian sid RISC-V with QEMU 7.0.0
