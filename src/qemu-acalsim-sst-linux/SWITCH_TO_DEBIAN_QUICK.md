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

# Quick Guide: Switch to Debian RISC-V

## TL;DR - Three Commands

```bash
# 1. Run setup script (1-2 hours)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./setup_debian_riscv.sh

# 2. Boot Debian
./run_qemu_debian.sh

# 3. Login (root/root) and use apt
apt update
apt install <whatever you need>
```

---

## Step-by-Step

### 1. Create Debian Rootfs

```bash
docker exec -it acalsim-workspace bash

cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./setup_debian_riscv.sh
```

**What it does:**
- Installs debootstrap
- Downloads Debian sid RISC-V (~2GB)
- Configures system (hostname, network, password)
- Creates initramfs (~600MB compressed)

**Time:** 1-2 hours (mostly downloading)

### 2. Boot Debian RISC-V

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_debian.sh
```

**Login:**
- Username: `root`
- Password: `root`

### 3. Install Packages in Debian

```bash
# Inside Debian

# Update package lists
apt update

# Install whatever you need
apt install python3-torch  # If RISC-V build exists
apt install build-essential cmake git
apt install python3-pip python3-numpy
```

---

## What's Included

### Pre-installed in Debian:
- ✅ Python 3 (latest in sid)
- ✅ pip, NumPy
- ✅ GCC, build tools
- ✅ CMake, Ninja
- ✅ Git, Vim, wget, curl
- ✅ OpenBLAS
- ✅ SSH server
- ✅ Network tools

### Easy to Install:
```bash
apt install <package>
```

---

## Files Created

```
/home/user/
├── debian-riscv/
│   └── rootfs/                    # Full Debian filesystem
├── debian-initramfs.cpio.gz       # Bootable initramfs (~600MB)
└── projects/.../llama-inference/
    ├── setup_debian_riscv.sh      # Automated setup
    └── run_qemu_debian.sh         # Boot script
```

---

## Comparison

| Feature | Debian | Buildroot |
|---------|--------|-----------|
| **Setup** | 1-2 hours | Already done |
| **Size** | ~2GB | ~500MB |
| **Packages** | `apt install` | Rebuild |
| **PyTorch** | May have pre-built | Build from source |
| **Ease** | ★★★★★ | ★★★☆☆ |
| **Control** | ★★★☆☆ | ★★★★★ |

---

## Boot Commands Comparison

### Buildroot (Current):
```bash
./run_qemu_initramfs.sh
# Login: root (no password)
```

### Debian (New):
```bash
./run_qemu_debian.sh
# Login: root / password: root
```

---

## PyTorch in Debian

### Option 1: Try Pre-built (Fast)
```bash
# Inside Debian
apt update
apt search pytorch
apt install python3-torch  # If available for RISC-V
```

### Option 2: Build from Source (Slow)
```bash
# Inside Debian
apt install -y build-essential cmake git python3-dev libopenblas-dev

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

export USE_CUDA=0
python3 setup.py install
```

---

## Troubleshooting

### Setup script fails with "debootstrap not found"
```bash
sudo apt-get update
sudo apt-get install -y debootstrap qemu-user-static
```

### Boot fails with kernel panic
```bash
# Check initramfs exists
ls -lh /home/user/debian-initramfs.cpio.gz

# Verify kernel has required features
grep -E 'CONFIG_EXT4|CONFIG_INITRAMFS' /home/user/linux/.config
```

### Can't login
```bash
# Default credentials:
# Username: root
# Password: root

# If password doesn't work, recreate rootfs:
rm -rf /home/user/debian-riscv
./setup_debian_riscv.sh
```

---

## Switching Back to Buildroot

```bash
# Just use the buildroot boot script
./run_qemu_initramfs.sh
```

Both can coexist - you can switch between them anytime!

---

## My Recommendation

### Try Buildroot First
Since I already added PyTorch support to buildroot:

```bash
# Rebuild buildroot with PyTorch packages (~30-60 min)
cd /home/user/buildroot-llama/buildroot-2024.02
# Add PyTorch packages to config (see PYTORCH_SUPPORT_ADDED.md)
make -j$(nproc)
```

### Switch to Debian If:
1. ❌ Buildroot PyTorch build fails
2. ❌ You need many packages via apt
3. ✅ Pre-built PyTorch RISC-V packages exist
4. ✅ You prefer standard Linux environment

---

## Quick Decision Guide

**Choose Debian if you answer YES to:**
- Need to install packages frequently?
- Want standard Linux experience?
- Prefer `apt install` over rebuilding?
- Have 10GB+ disk space?

**Choose Buildroot if you answer YES to:**
- Want minimal, optimized system?
- Need reproducible environment?
- Prefer full control over packages?
- Already invested time in buildroot?

---

**Created**: 2025-11-20
**Scripts**: `setup_debian_riscv.sh`, `run_qemu_debian.sh`
**Documentation**: `DEBIAN_SETUP_GUIDE.md`
