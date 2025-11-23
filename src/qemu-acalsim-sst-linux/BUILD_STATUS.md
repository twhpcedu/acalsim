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

# Build Status Report - QEMU with virtio-sst

**Date**: 2025-11-20
**Status**: ✅ **QEMU & Kernel Build SUCCESSFUL**

---

## ✅ Successfully Completed

### 1. QEMU 7.0.0 Build
- **Status**: ✅ **WORKING**
- **Location**: `/home/user/qemu-build/qemu/build/qemu-system-riscv64`
- **Features**:
  - ✅ virtio-sst-device available and initializing
  - ✅ User-mode networking (SLIRP) working
  - ✅ VirtIO block device working
  - ✅ VirtIO network device working
  - ✅ Boots OpenSBI successfully
- **Build Time**: ~6 minutes
- **Build Script**: `qemu-config/build-qemu-7.0.sh`

### 2. Linux Kernel v6.1 Build
- **Status**: ✅ **WORKING**
- **Location**: `/home/user/linux/arch/riscv/boot/Image`
- **Features**:
  - ✅ RISC-V 64-bit support
  - ✅ VirtIO drivers enabled
  - ✅ Networking support
  - ✅ Block device support
  - ✅ 9P filesystem support
  - ✅ Boots successfully in QEMU
- **Build Time**: ~15-30 minutes
- **Build Script**: `qemu-config/build-linux-kernel.sh`

### 3. Boot Process
- **Status**: ✅ **BOOTS TO LOGIN PROMPT**
- **Verified**:
  - ✅ OpenSBI loads
  - ✅ Kernel boots
  - ✅ Root filesystem mounts
  - ✅ Init process starts (/init)
  - ✅ virtio-sst-device initializes (warns about missing SST connection - expected)
  - ✅ Network device eth0 detected and configured
  - ✅ All system services start successfully
  - ✅ **Login prompt displayed**

### 4. Buildroot Rootfs
- **Status**: ✅ **WORKING PERFECTLY**
- **Location**: `/home/user/initramfs-buildroot.cpio.gz` → `/home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz`
- **Size**: 89MB compressed, 233MB uncompressed
- **Features**:
  - ✅ RISC-V native binaries
  - ✅ Full system utilities (syslog, udev, networking)
  - ✅ SSH server (dropbear)
  - ✅ DHCP client (udhcpc + dhcpcd)
  - ✅ Network stack (IPv4 + IPv6)
  - ✅ Proper init system with service management

---

## ✅ All Issues Resolved

### Rootfs Now Working
- **Status**: ✅ **FULLY WORKING** with buildroot rootfs
- **Solution**: Using buildroot 2024.02 initramfs with proper RISC-V binaries
- **Features**:
  - ✅ All system services start successfully
  - ✅ Network configured via DHCP (10.0.2.15)
  - ✅ IPv6 auto-configuration working
  - ✅ SSH daemon (dropbear) running
  - ✅ Syslog and kernel logging working
  - ✅ Udev device management working
  - ✅ **Login prompt reached successfully**

**Previous Issue** (now resolved): The old `rootfs-python-persistent.qcow2` had permission issues due to being created in a restricted environment.

**Current Solution**: Using buildroot's RISC-V initramfs directly (`/home/user/initramfs-buildroot.cpio.gz`)

---

## 📊 Build Summary

| Component | Status | Build Time | Size |
|-----------|--------|------------|------|
| QEMU 7.0.0 | ✅ Working | ~6 min | 500MB |
| Linux Kernel v6.1 | ✅ Working | 15-30 min | ~2GB source |
| Buildroot Rootfs | ✅ Working | Complete | 89MB |
| **Overall** | **✅ FULLY WORKING** | **~25 min** | **~2.5GB** |

---

## 🎯 What Was Fixed

### Original Problems:
1. ❌ `virtio-sst-device not found` - QEMU was too new (v10.1)
2. ❌ `network backend 'user' not found` - Missing libslirp
3. ❌ Kernel not available - Not built yet
4. ❌ Incompatible QEMU API - virtio-sst code written for old API
5. ❌ Rootfs permission errors - Old qcow2 image had capability issues
6. ❌ No login prompt - Services couldn't start

### Solutions Applied:
1. ✅ Downgraded to QEMU 7.0.0 (last compatible version)
2. ✅ Installed libslirp-dev before building
3. ✅ Built Linux kernel v6.1 with VirtIO drivers
4. ✅ Fixed header include order (qemu/osdep.h)
5. ✅ Updated meson.build for QEMU 7.0 structure
6. ✅ Added virtio-sst to Kconfig
7. ✅ Used buildroot initramfs with proper RISC-V binaries
8. ✅ **Successfully reached login prompt**

---

## 📁 File Locations

```
/home/user/
├── qemu-build/qemu/build/
│   └── qemu-system-riscv64              # ✅ QEMU binary (working)
├── linux/arch/riscv/boot/
│   └── Image                             # ✅ Kernel image (working)
├── initramfs-buildroot.cpio.gz           # ✅ Symlink to buildroot rootfs (WORKING)
├── buildroot-llama/buildroot-2024.02/output/images/
│   └── rootfs.cpio.gz                    # ✅ Buildroot rootfs (89MB)
├── rootfs-persistent.qcow2               # Symlink to old rootfs (deprecated)
└── rootfs-python-persistent.qcow2        # ⚠️ Old rootfs with permission issues (deprecated)
```

---

## 🚀 How to Use

### Boot QEMU (Recommended - Working):
```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

### Expected Output:
```
VirtIO SST: Initializing device (socket=/tmp/qemu-sst-llama.sock, id=0)
VirtIO SST: Warning - Failed to connect to SST at /tmp/qemu-sst-llama.sock: No such file or directory
VirtIO SST: Device will work without SST connection
VirtIO SST: Device initialized successfully

OpenSBI v1.0
   ____                    _____ ____ _____
  / __ \                  / ____|  _ \_   _|
  ...

[    2.469751] Run /init as init process
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

**Exit QEMU**: Press `Ctrl-A` then `X`

### Verify Build:
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
```

---

## 📚 Documentation Created

1. **`qemu-config/build-qemu-7.0.sh`**
   - Automated QEMU 7.0.0 build script
   - Installs dependencies
   - Integrates virtio-sst device
   - Verifies build

2. **`qemu-config/build-linux-kernel.sh`**
   - Automated kernel build script
   - Installs cross-compiler
   - Configures VirtIO drivers
   - Builds for RISC-V

3. **`docs/BUILD_QEMU_SOP.md`**
   - Complete build instructions
   - API compatibility notes
   - Troubleshooting guide
   - Multiple build options

4. **`QUICK_START.md`**
   - Quick reference guide
   - Step-by-step instructions
   - Verification steps
   - Common issues

5. **`BUILD_STATUS.md`** (this file)
   - Build status summary
   - Known issues
   - What was fixed

---

## ✅ Conclusion

**The complete RISC-V Linux system is now fully operational!**

All issues have been resolved:
- ✅ QEMU 7.0.0 built with virtio-sst device
- ✅ User-mode networking enabled
- ✅ Linux kernel built and boots successfully
- ✅ Buildroot rootfs with all services working
- ✅ **System boots to login prompt**
- ✅ Network configured (IPv4 + IPv6)
- ✅ SSH server running

**The system is ready for llama inference and other workloads.**

---

**Build Verified On**: 2025-11-20
**Platform**: Ubuntu 22.04 (jammy) in acalsim-workspace Docker container
**Architecture**: ARM64 (aarch64)
**QEMU Version**: 7.0.0
**Kernel Version**: 6.1
