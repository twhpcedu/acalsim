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

# Rebuild Everything From Scratch

This guide shows how to completely clean and rebuild the entire RISC-V Linux environment from scratch.

## Quick Clean & Rebuild (Recommended)

```bash
# Enter the Docker container
docker exec -it acalsim-workspace bash

# Navigate to project directory
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux

# Clean everything
rm -rf /home/user/qemu-build
rm -rf /home/user/linux
rm -rf /home/user/buildroot-llama/buildroot-2024.02
rm -f /home/user/initramfs-buildroot.cpio.gz

# Rebuild QEMU (takes ~6 minutes)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./build-qemu-7.0.sh

# Rebuild Linux kernel (takes ~15-30 minutes)
./build-linux-kernel.sh

# Rebuild buildroot rootfs (if needed, takes ~30-60 minutes)
# Note: You may already have buildroot built, check first:
ls -lh /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz

# If not present, rebuild buildroot:
cd /home/user/buildroot-llama/buildroot-2024.02
make clean
make -j$(nproc)

# Create initramfs symlink
ln -sf /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz \
       /home/user/initramfs-buildroot.cpio.gz

# Test boot
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

## Step-by-Step Clean & Rebuild

### Step 1: Clean QEMU Build

```bash
# Remove entire QEMU build directory
rm -rf /home/user/qemu-build

# Or just clean the build (keeps source):
cd /home/user/qemu-build/qemu/build
ninja clean  # or: make clean
```

### Step 2: Clean Linux Kernel Build

```bash
# Remove entire Linux source directory
rm -rf /home/user/linux

# Or just clean the build (keeps source and config):
cd /home/user/linux
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- clean

# Or deep clean (removes all generated files):
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- mrproper
```

### Step 3: Clean Buildroot Rootfs

```bash
# Option 1: Clean buildroot build (keeps config)
cd /home/user/buildroot-llama/buildroot-2024.02
make clean

# Option 2: Deep clean (removes all output)
make distclean

# Option 3: Remove entire buildroot directory
rm -rf /home/user/buildroot-llama/buildroot-2024.02
```

### Step 4: Rebuild QEMU

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./build-qemu-7.0.sh
```

**What it does:**
- Installs dependencies (libslirp, meson, etc.)
- Clones QEMU 7.0.0
- Integrates virtio-sst device
- Configures and builds QEMU
- Verifies build

**Build time:** ~6 minutes
**Output:** `/home/user/qemu-build/qemu/build/qemu-system-riscv64`

### Step 5: Rebuild Linux Kernel

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./build-linux-kernel.sh
```

**What it does:**
- Installs cross-compiler (gcc-riscv64-linux-gnu)
- Clones Linux v6.1
- Configures with VirtIO drivers
- Builds kernel

**Build time:** ~15-30 minutes
**Output:** `/home/user/linux/arch/riscv/boot/Image`

### Step 6: Rebuild Buildroot Rootfs (If Needed)

```bash
cd /home/user/buildroot-llama/buildroot-2024.02

# Check if .config exists, if not restore it
ls -la .config

# Clean and rebuild
make clean
make -j$(nproc)
```

**Build time:** ~30-60 minutes
**Output:** `/home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz`

### Step 7: Create Initramfs Symlink

```bash
ln -sf /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz \
       /home/user/initramfs-buildroot.cpio.gz
```

### Step 8: Verify Build

```bash
# Check QEMU
qemu-system-riscv64 --version
# Expected: QEMU emulator version 7.0.0

# Check virtio-sst device
qemu-system-riscv64 -device help | grep virtio-sst
# Expected: name "virtio-sst-device", bus virtio-bus

# Check networking
qemu-system-riscv64 -netdev help | grep user
# Expected: user

# Check kernel
ls -lh /home/user/linux/arch/riscv/boot/Image
# Expected: ~10-20MB file

# Check initramfs
ls -lh /home/user/initramfs-buildroot.cpio.gz
# Expected: ~89MB file
```

### Step 9: Test Boot

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

**Expected output:**
```
Welcome to ACAL Simulator RISC-V Linux
acalsim-riscv login:
```

## Clean Only Dependencies (Keep Builds)

If you just want to force rebuild without cleaning:

```bash
# Force QEMU rebuild
cd /home/user/qemu-build/qemu/build
ninja clean
../configure --target-list=riscv64-softmmu --enable-virtfs
make -j$(nproc)

# Force kernel rebuild
cd /home/user/linux
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- clean
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j$(nproc)
```

## Nuclear Option - Clean Everything

```bash
# WARNING: This removes ALL builds
rm -rf /home/user/qemu-build
rm -rf /home/user/linux
rm -rf /home/user/buildroot-llama
rm -f /home/user/initramfs*.cpio.gz
rm -f /home/user/rootfs*.qcow2

# Then rebuild using the scripts
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./build-qemu-7.0.sh
./build-linux-kernel.sh

# For buildroot, you'll need to follow the buildroot SOP again
```

## Troubleshooting Rebuild Issues

### Build Script Fails

```bash
# Check if running inside Docker
docker ps | grep acalsim-workspace

# Enter Docker if needed
docker exec -it acalsim-workspace bash

# Verify you're in the right directory
pwd
# Should be: /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
```

### Missing Dependencies

```bash
# Reinstall all dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    libglib2.0-dev \
    libpixman-1-dev \
    libslirp-dev \
    libslirp0 \
    libcap-ng-dev \
    libattr1-dev \
    gcc-riscv64-linux-gnu \
    flex \
    bison \
    bc \
    libssl-dev \
    libelf-dev
```

### Disk Space Issues

```bash
# Check disk space
df -h /home/user

# Clean old builds if needed
du -sh /home/user/qemu-build
du -sh /home/user/linux
du -sh /home/user/buildroot-llama

# Clean build artifacts
cd /home/user/qemu-build/qemu/build && ninja clean
cd /home/user/linux && make clean
```

## Build Time Summary

| Component | Clean Time | Build Time | Total |
|-----------|------------|------------|-------|
| QEMU 7.0 | 1 sec | ~6 min | ~6 min |
| Linux Kernel | 1 sec | 15-30 min | 15-30 min |
| Buildroot | 1 sec | 30-60 min | 30-60 min |
| **Total** | **~3 sec** | **~50-100 min** | **~50-100 min** |

*Note: Buildroot is only needed if you don't have a working rootfs already*

## One-Liner Clean & Rebuild

```bash
# Complete clean and rebuild (no buildroot)
docker exec -it acalsim-workspace bash -c "
  rm -rf /home/user/qemu-build /home/user/linux && \
  cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config && \
  ./build-qemu-7.0.sh && \
  ./build-linux-kernel.sh && \
  echo 'Build complete! Test with: cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference && ./run_qemu_initramfs.sh'
"
```

## Files That Won't Be Affected

These configuration files are preserved:
- `qemu-config/build-qemu-7.0.sh` (build script)
- `qemu-config/build-linux-kernel.sh` (build script)
- `examples/llama-inference/run_qemu_initramfs.sh` (boot script)
- All source code in the repository

## Post-Rebuild Verification

```bash
# Verify all components
echo "=== QEMU Check ==="
qemu-system-riscv64 --version

echo "=== Kernel Check ==="
ls -lh /home/user/linux/arch/riscv/boot/Image

echo "=== Initramfs Check ==="
ls -lh /home/user/initramfs-buildroot.cpio.gz

echo "=== Boot Test ==="
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
timeout 30 ./run_qemu_initramfs.sh 2>&1 | grep -A 3 "Welcome to ACAL"
```

---

**Last Updated**: 2025-11-20
**Tested On**: Ubuntu 22.04 in acalsim-workspace Docker container
