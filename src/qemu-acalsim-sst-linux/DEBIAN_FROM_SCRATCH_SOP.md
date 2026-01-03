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

# SOP: Boot Debian RISC-V Linux from Scratch

## Prerequisites

- Docker container `acalsim-workspace` running
- QEMU 7.0.0 with virtio-sst built (in `/home/user/qemu-build/qemu/build/`)
- Linux kernel built (in `/home/user/linux/arch/riscv/boot/Image`)
- Internet access for downloading Debian DQIB image

---

## Step 1: Download Debian DQIB RISC-V Image

```bash
# Enter Docker container
docker exec -it acalsim-workspace bash

# Create directory for Debian
mkdir -p /home/user/debian-riscv
cd /home/user/debian-riscv

# Download Debian Quick Image Baker (DQIB) artifacts
wget https://gitlab.com/giomasce/dqib/-/jobs/artifacts/master/download?job=convert_riscv64-virt -O dqib_riscv64.zip

# Extract the archive
unzip dqib_riscv64.zip

# The extracted directory contains:
# - image.qcow2 (Debian root filesystem ~443MB)
# - kernel (Linux kernel ~33MB)
# - initrd (Initial ramdisk ~50MB)
# - openssh_host_* (SSH keys)
# - readme.txt (Documentation)

# Verify files
ls -lh dqib_riscv64-virt/
```

**Expected output:**
```
total 526M
-rw-r--r-- 1 user user 443M Nov 20 12:00 image.qcow2
-rw-r--r-- 1 user user  33M Nov 20 12:00 kernel
-rw-r--r-- 1 user user  50M Nov 20 12:00 initrd
-rw-r--r-- 1 user user  411 Nov 20 12:00 openssh_host_ecdsa_key
-rw-r--r-- 1 user user  102 Nov 20 12:00 openssh_host_ecdsa_key.pub
-rw-r--r-- 1 user user  399 Nov 20 12:00 openssh_host_ed25519_key
-rw-r--r-- 1 user user   94 Nov 20 12:00 openssh_host_ed25519_key.pub
-rw-r--r-- 1 user user 2.5K Nov 20 12:00 openssh_host_rsa_key
-rw-r--r-- 1 user user  566 Nov 20 12:00 openssh_host_rsa_key.pub
-rw-r--r-- 1 user user 1.2K Nov 20 12:00 readme.txt
```

---

## Step 2: Boot Debian RISC-V

```bash
# Navigate to llama-inference directory
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference

# Boot Debian DQIB
./run_qemu_debian_dqib.sh
```

**Expected boot output:**
```
Booting DQIB Debian RISC-V Linux...
Press Ctrl-A then X to exit QEMU

Login credentials:
  root / root
  debian / debian

SSH access: ssh -p 2222 debian@localhost

[    0.000000] Linux version 6.1.0 ...
[    0.000000] Machine model: riscv-virtio,qemu
...
[   OK   ] Started OpenBSD Secure Shell server.
[   OK   ] Reached target Multi-User System.

Debian GNU/Linux bookworm/sid debian ttyS0

debian login: _
```

---

## Step 3: Login and Configure System

### Option A: Login as root (Full privileges)

```
debian login: root
Password: root

root@debian:~#
```

### Option B: Login as debian user (Standard user)

```
debian login: debian
Password: debian

debian@debian:~$
```

---

## Step 4: Configure Sudo (Recommended)

**Login as root first, then run:**

```bash
# Update package lists
apt update

# Install sudo
apt install -y sudo

# Add debian user to sudo group
usermod -aG sudo debian

# Configure passwordless sudo (optional but convenient)
echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian
chmod 440 /etc/sudoers.d/debian

# Fix hostname resolution warning (optional)
echo "127.0.0.1 debian" >> /etc/hosts

# Verify sudo works
su - debian -c "sudo whoami"
# Should print: root
```

**One-liner version:**
```bash
apt update && apt install -y sudo && usermod -aG sudo debian && echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian && chmod 440 /etc/sudoers.d/debian && echo "127.0.0.1 debian" >> /etc/hosts && echo "✓ Sudo configured"
```

---

## Step 5: Install Development Tools

**As debian user with sudo:**

```bash
# Switch to debian user
su - debian

# Update package lists
sudo apt update
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux

# Install Python development packages
sudo apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-yaml \
    python3-setuptools \
    python3-wheel

# Install libraries for PyTorch
sudo apt install -y \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libffi-dev \
    libssl-dev \
    ninja-build \
    patchelf \
    ccache

# Verify installations
python3 --version
pip3 --version
cmake --version
git --version
```

---

## Step 6: Install PyTorch (Optional)

### Option 1: Try Pre-built Package (Fast)

```bash
# Search for PyTorch in Debian repositories
apt search pytorch

# If available, install
sudo apt install -y python3-torch python3-torch-cpu
```

### Option 2: Build from Source (4-8 hours)

```bash
# Install Python dependencies
pip3 install typing-extensions pyyaml requests packaging

# Clone PyTorch
cd /home/debian
git clone --depth 1 --branch v2.1.0 --recursive https://github.com/pytorch/pytorch
cd pytorch

# Configure build environment
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0
export MAX_JOBS=4
export USE_OPENBLAS=1

# Build and install (4-8 hours)
python3 setup.py install

# Test installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
```

**For detailed PyTorch build instructions, see:** `BUILD_PYTORCH_IN_QEMU.md`

---

## SSH Access (Alternative to Console)

The DQIB image has SSH port forwarding configured:

```bash
# From your host machine (outside Docker)
ssh -p 2222 debian@localhost
# Password: debian

# Or as root
ssh -p 2222 root@localhost
# Password: root
```

---

## Comparison: Debian DQIB vs Buildroot

| Feature | Debian DQIB | Buildroot |
|---------|-------------|-----------|
| **Setup Time** | 5-10 minutes | Already configured |
| **Download Size** | ~328MB (ZIP) | N/A (built locally) |
| **Rootfs Size** | ~443MB (qcow2) | ~89MB (cpio.gz) |
| **Boot Script** | `./run_qemu_debian_dqib.sh` | `./run_qemu_initramfs.sh` |
| **Package Manager** | ✅ apt (easy) | ❌ Rebuild needed |
| **Login** | root/root or debian/debian | root (no password) |
| **SSH Access** | ✅ Port 2222 | ❌ Not configured |
| **Persistence** | ✅ qcow2 disk | ❌ initramfs (volatile) |
| **PyTorch** | May have pre-built | Build from source |
| **Best For** | Development, experiments | Embedded, minimal |

---

## Boot Scripts Reference

### Debian DQIB Boot Script
```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian_dqib.sh
```

**Script contents (`run_qemu_debian_dqib.sh`):**
```bash
#!/bin/bash
# Boot DQIB Debian RISC-V image

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
DQIB_DIR="/home/user/debian-riscv/dqib_riscv64-virt"
KERNEL="$DQIB_DIR/kernel"
INITRD="$DQIB_DIR/initrd"
DEBIAN_DISK="$DQIB_DIR/image.qcow2"
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

exec $QEMU_BIN \
    -M virt \
    -cpu rv64 \
    -smp 4 \
    -m 8G \
    -kernel "$KERNEL" \
    -initrd "$INITRD" \
    -append "root=LABEL=rootfs console=ttyS0" \
    -drive file="$DEBIAN_DISK",if=none,id=hd,format=qcow2 \
    -device virtio-blk-device,drive=hd \
    -netdev user,id=net0,hostfwd=tcp:127.0.0.1:2222-:22 \
    -device virtio-net-device,netdev=net0 \
    -device virtio-sst-device,socket=$SOCKET_PATH \
    -object rng-random,filename=/dev/urandom,id=rng \
    -device virtio-rng-device,rng=rng \
    -nographic
```

### Buildroot Boot Script (Alternative)
```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

---

## Troubleshooting

### DQIB Download Fails

```bash
# Check internet connectivity
ping -c 3 google.com

# Try alternative download URL
wget https://gitlab.com/api/v4/projects/giomasce%2Fdqib/jobs/artifacts/master/download?job=convert_riscv64-virt -O dqib_riscv64.zip

# Or download manually and copy to Docker container
docker cp dqib_riscv64.zip acalsim-workspace:/home/user/debian-riscv/
```

### Boot Fails - Kernel Not Found

```bash
# Verify DQIB files exist
ls -lh /home/user/debian-riscv/dqib_riscv64-virt/

# Check paths in boot script
cat /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/run_qemu_debian_dqib.sh

# Verify DQIB_DIR path matches
DQIB_DIR="/home/user/debian-riscv/dqib_riscv64-virt"
```

### Login Fails - Wrong Password

**Default credentials:**
- Username: `root`, Password: `root`
- Username: `debian`, Password: `debian`

If passwords don't work, the DQIB image may have been modified. Try both users.

### Sudo Gives Hostname Warning

```bash
# Fix with:
echo "127.0.0.1 debian" >> /etc/hosts
```

### SSH Connection Refused

```bash
# Check SSH service status inside Debian
sudo systemctl status ssh

# Start SSH if not running
sudo systemctl start ssh

# Check port forwarding in boot script
# Should have: -netdev user,id=net0,hostfwd=tcp:127.0.0.1:2222-:22
```

---

## Quick Reference Commands

```bash
# Download DQIB
mkdir -p /home/user/debian-riscv && cd /home/user/debian-riscv
wget https://gitlab.com/giomasce/dqib/-/jobs/artifacts/master/download?job=convert_riscv64-virt -O dqib_riscv64.zip
unzip dqib_riscv64.zip

# Boot Debian
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian_dqib.sh

# Configure sudo (run inside Debian as root)
apt update && apt install -y sudo && usermod -aG sudo debian && echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian && chmod 440 /etc/sudoers.d/debian

# Exit QEMU
# Press: Ctrl-A, then X
```

---

## Related Documentation

- **BUILD_PYTORCH_IN_QEMU.md** - Detailed PyTorch build guide
- **PYTORCH_INSTALLATION_SUMMARY.md** - Quick PyTorch overview
- **DEBIAN_SUDO_SETUP.md** - Comprehensive sudo configuration
- **SWITCH_TO_DEBIAN_QUICK.md** - Debian vs Buildroot comparison
- **run_qemu_debian_dqib.sh** - Debian boot script
- **run_qemu_initramfs.sh** - Buildroot boot script (alternative)

---

**Created**: 2025-11-20
**DQIB Version**: master (riscv64-virt)
**QEMU Version**: 7.0.0 with virtio-sst
**Linux Kernel**: Provided by DQIB (6.1.0)
**Debian Version**: Bookworm/Sid (testing/unstable)
