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

# QEMU Build Instructions for ACALSIM

This document provides step-by-step instructions to build QEMU with virtio-sst device support.

## Prerequisites

Install required dependencies:

```bash
sudo apt update
sudo apt install -y \
  software-properties-common \
  ninja-build meson build-essential \
  libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev \
  pkg-config python3 git

# Network support dependency
sudo apt-get install -y libslirp-dev libslirp0
```

## Option 1: Build QEMU 7.0 (Recommended - Compatible with virtio-sst)

This uses QEMU 7.0.0, the last version before the `virtio_init()` API changed, making it compatible with the existing virtio-sst device code.

**Why QEMU 7.0?** The virtio-sst code uses the old `virtio_init()` API which was changed in QEMU 7.1+. QEMU 7.0.0 is the last stable version with the compatible API.

```bash
# Create build directory
mkdir -p /home/user/qemu-build
cd /home/user/qemu-build

# Clone QEMU and checkout version 7.0.0
git clone https://github.com/qemu/qemu.git
cd qemu
git checkout v7.0.0

# Integrate virtio-sst device into QEMU source
# Method 1: Using setup script (may need adjustments for QEMU 7.0)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./setup-qemu-virtio-sst.sh /home/user/qemu-build/qemu

# If setup script fails, use Method 2: Manual integration
# 1. Copy device files
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/virtio-device/sst-protocol.h \
   /home/user/qemu-build/qemu/include/hw/virtio/
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/virtio-device/virtio-sst.h \
   /home/user/qemu-build/qemu/include/hw/virtio/
cp /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/virtio-device/virtio-sst.c \
   /home/user/qemu-build/qemu/hw/virtio/

# 2. Fix header include order (QEMU requires qemu/osdep.h first)
sed -i '/#include "qemu\/osdep.h"/d' /home/user/qemu-build/qemu/include/hw/virtio/virtio-sst.h
sed -i '1i #include "qemu/osdep.h"' /home/user/qemu-build/qemu/hw/virtio/virtio-sst.c

# 3. Update meson.build to include virtio-sst
# Add after the line with CONFIG_VHOST_USER_SCMI or similar
cd /home/user/qemu-build/qemu/hw/virtio
# Find a good spot (after other virtio devices) and add:
sed -i '/system_virtio_ss.add(when.*CONFIG_VIRTIO/a system_virtio_ss.add(when: '\''CONFIG_VIRTIO_SST'\'', if_true: files('\''virtio-sst.c'\''))' meson.build

# 4. Add Kconfig entry
cat >> /home/user/qemu-build/qemu/hw/virtio/Kconfig << 'EOF'

config VIRTIO_SST
    bool
    default y
    depends on VIRTIO
EOF

# Configure and build QEMU
cd /home/user/qemu-build/qemu
mkdir -p build
cd build
../configure --target-list=riscv64-softmmu --enable-virtfs
make -j$(nproc)

# Add QEMU to PATH
export PATH=/home/user/qemu-build/qemu/build:$PATH
echo 'export PATH=/home/user/qemu-build/qemu/build:$PATH' >> ~/.bashrc

# Verify installation
qemu-system-riscv64 --version
qemu-system-riscv64 -device help | grep virtio-sst
qemu-system-riscv64 -netdev help | grep user
```

**Expected output:**
- Version: QEMU emulator version 7.0.0
- Device list should include `virtio-sst-device`
- Network backends should include `user`

## Option 2: Build Latest QEMU (Requires virtio-sst fixes)

⚠️ **Warning:** The current virtio-sst code is NOT compatible with QEMU 10.x+. This requires code updates.

```bash
# Create build directory
mkdir -p /home/user/qemu-build
cd /home/user/qemu-build

# Clone latest QEMU
git clone https://github.com/qemu/qemu.git
cd qemu

# DO NOT integrate virtio-sst yet - it needs API updates for QEMU 10.x

# Build QEMU without virtio-sst (networking only)
mkdir -p build
cd build
../configure --target-list=riscv64-softmmu --enable-virtfs
make -j$(nproc)

# Verify
qemu-system-riscv64 --version
qemu-system-riscv64 -netdev help | grep user
```

## Option 3: Build with Updated virtio-sst (Future)

This requires updating the virtio-sst device code for QEMU 10.x API changes:

**Required fixes:**
1. Fix `virtio_init()` - signature changed, no longer takes device name
2. Fix `DEFINE_PROP_END_OF_LIST()` - should be without parentheses
3. Fix `set_status` callback - now returns `int` instead of `void`
4. Fix `class_init` - parameter should be `const void*` not `void*`
5. Remove target-specific includes that cause poisoning errors

*Code updates not included yet - requires development work*

## Boot QEMU with virtio-sst

After successful build, boot the persistent disk:

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_persistent.sh
```

## Troubleshooting

### Issue: `virtio-sst-device not found`
**Cause:** Device not integrated or QEMU built without it

**Solution:**
- Verify files copied: `ls /home/user/qemu-build/qemu/hw/virtio/virtio-sst.c`
- Check build system updated correctly
- Rebuild from clean state: `cd build && make clean && make -j$(nproc)`

### Issue: `network backend 'user' is not compiled into this binary`
**Cause:** libslirp not installed before configure

**Solution:**
```bash
sudo apt-get install -y libslirp-dev libslirp0
cd /home/user/qemu-build/qemu
rm -rf build
mkdir build && cd build
../configure --target-list=riscv64-softmmu --enable-virtfs
make -j$(nproc)
```

### Issue: Compilation errors in virtio-sst.c
**Cause:** QEMU version incompatibility

**Solution:** Use QEMU 8.2.0 (Option 1) instead of latest

## Current Status

As of 2025-11-20:
- ✅ QEMU 7.0.0 is compatible with virtio-sst device (TESTED - last version before API changes)
- ✅ Automated build script verified working: `qemu-config/build-qemu-7.0.sh`
- ❌ QEMU 7.1+ - 10.1+ require virtio-sst code updates (virtio_init API changed)
- ✅ Network support (SLIRP) works in all versions when libslirp is installed
- ⚠️ API breaking change: QEMU 7.1.0 dropped the `name` parameter from `virtio_init()`
- ✅ Build verified in acalsim-workspace Docker container on 2025-11-20

## Recommended Approach

**Use Option 1 (QEMU 7.0.0)** for immediate compatibility with existing virtio-sst code without modifications.

## API Compatibility Notes

The virtio-sst device code uses these APIs that changed in QEMU 7.1+:
- `virtio_init(vdev, "name", device_id, config_size)` → `virtio_init(vdev, device_id, config_size)`
- Other minor API changes in device properties and callbacks

To use newer QEMU versions, the virtio-sst code must be updated to match the new API.
