# RootFS Management and Package Installation

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

## Table of Contents

1. [Understanding the Filesystem Architecture](#understanding-the-filesystem-architecture)
2. [Method 1: Pre-Boot Modification (Recommended)](#method-1-pre-boot-modification-recommended)
3. [Method 2: Runtime Modification and Persistence](#method-2-runtime-modification-and-persistence)
4. [Installing Specific Packages](#installing-specific-packages)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Understanding the Filesystem Architecture

### What is initramfs?

The QEMU-SST Linux integration uses an **initramfs (Initial RAM Filesystem)** - a compressed filesystem loaded into RAM at boot time.

```
┌─────────────────────────────────────────────────────────────┐
│                     Boot Process                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. QEMU loads Linux kernel (Image)                         │
│           ↓                                                  │
│  2. Kernel decompresses initramfs.cpio.gz                    │
│           ↓                                                  │
│  3. Filesystem mounted in RAM (/dev/ram)                     │
│           ↓                                                  │
│  4. System boots with in-memory filesystem                   │
│                                                              │
│  ⚠️  ALL CHANGES LOST ON SHUTDOWN unless persisted          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Concepts:**

| Component | Location | Description |
|-----------|----------|-------------|
| **rootfs/** | `/home/user/rootfs` (in container) | Source directory for filesystem content |
| **initramfs.cpio.gz** | `/home/user/initramfs.cpio.gz` | Compressed archive loaded by QEMU |
| **Runtime FS** | Inside QEMU Linux | In-memory filesystem (changes lost on reboot) |

**Important**: The initramfs is **ephemeral** - any changes made during runtime (package installs, file modifications) are lost when QEMU shuts down unless you rebuild the initramfs.

---

## Method 1: Pre-Boot Modification (Recommended)

This is the **recommended approach** for installing packages - modify the rootfs directory, then rebuild initramfs.

### Step 1: Prepare Your Rootfs Directory

```bash
# In Docker container
docker exec -it acalsim-workspace bash

# Navigate to rootfs
cd /home/user/rootfs

# Current structure:
# rootfs/
# ├── bin/         # BusyBox binaries
# ├── sbin/        # System binaries
# ├── usr/         # User programs
# ├── apps/        # Your custom applications
# ├── lib/         # Shared libraries
# ├── dev/         # Device files
# ├── proc/        # Process info (empty)
# ├── sys/         # Sysfs (empty)
# └── init         # Init script
```

### Step 2: Install Packages

#### Option A: Cross-Compile from Source

For compiled programs (recommended for performance):

```bash
# Example: Install a custom tool
cd /home/user/rootfs/apps

# Cross-compile for RISC-V
riscv64-linux-gnu-gcc -static -o mytool mytool.c

# Verify it's RISC-V binary
file mytool
# Output: mytool: ELF 64-bit LSB executable, UCB RISC-V, ...
```

#### Option B: Copy Pre-Built RISC-V Binaries

If you have pre-compiled RISC-V binaries:

```bash
# Copy to rootfs
cp /path/to/riscv64-binary /home/user/rootfs/apps/

# Make executable
chmod +x /home/user/rootfs/apps/riscv64-binary
```

#### Option C: Install Python (via Buildroot/Custom Build)

For Python and complex packages, you need a **full RISC-V Python build**:

```bash
# Option 1: Use Buildroot (recommended)
cd /home/user
git clone https://github.com/buildroot/buildroot
cd buildroot

# Configure for RISC-V with Python
make menuconfig
# Select:
# - Target Architecture: RISCV
# - Target Architecture Variant: RV64GC
# - Toolchain: Buildroot toolchain
# - Interpreter languages and scripting: Python3
# - Python3 modules: (select what you need)

# Build (this takes 30min - 2 hours)
make -j$(nproc)

# Extract Python from output
cp output/target/usr/bin/python3 /home/user/rootfs/usr/bin/
cp -r output/target/usr/lib/python3.* /home/user/rootfs/usr/lib/

# Copy required shared libraries
cp output/target/lib/libc.so.6 /home/user/rootfs/lib/
cp output/target/lib/libpthread.so.0 /home/user/rootfs/lib/
# ... (copy all dependencies shown by ldd)
```

**Alternative**: Use pre-built RISC-V root filesystem from distributions:
- [Debian RISC-V port](https://wiki.debian.org/RISC-V)
- [Fedora RISC-V](https://fedoraproject.org/wiki/Architectures/RISC-V)

### Step 3: Rebuild initramfs

After modifying rootfs, rebuild the initramfs:

```bash
# In container at /home/user/rootfs
cd /home/user/rootfs

# Create compressed initramfs
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz

# Verify size
ls -lh /home/user/initramfs.cpio.gz
```

**Important**: The initramfs size affects boot time and memory usage. Keep it under 500MB for reasonable performance.

### Step 4: Boot with New initramfs

```bash
# QEMU will automatically use the updated initramfs.cpio.gz
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./run-linux.sh

# Or run SST tests
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/tests
./test_runner.sh NOOP
```

---

## Method 2: Runtime Modification and Persistence

If you need to install packages **while Linux is running**, you can save changes back to initramfs.

### Step 1: Boot Linux and Install Packages

```bash
# Start QEMU-SST Linux
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/qemu-config
./run-linux.sh

# In Linux console, make your changes:
mkdir -p /apps/myproject
echo "Hello World" > /apps/myproject/data.txt

# If you have a package manager (requires full rootfs):
apk add python3 py3-pip  # Alpine Linux
# OR
apt-get install python3 python3-pip  # Debian/Ubuntu
```

### Step 2: Export Filesystem from Running Linux

While Linux is still running, from **another terminal**:

```bash
# Connect to QEMU serial console via Unix socket or
# use QEMU monitor commands to access filesystem

# Method A: Use QEMU's virtiofs or 9pfs to share a directory
# (requires QEMU configured with shared folder)

# Method B: Copy files via SSH/SCP
# (requires networking and SSH daemon in Linux)

# Method C: Use QEMU's savevm/loadvm for full state
# (info in QEMU documentation)
```

### Step 3: Manual Extraction and Rebuild

If you can't access the running filesystem, you need to:

1. **Extract current initramfs**:
```bash
# In container
mkdir /tmp/extract
cd /tmp/extract
gunzip -c /home/user/initramfs.cpio.gz | cpio -idmv

# Now /tmp/extract contains the full filesystem
```

2. **Make your changes**:
```bash
# Add files, install packages, etc.
cp /path/to/new/files /tmp/extract/apps/
```

3. **Rebuild initramfs**:
```bash
cd /tmp/extract
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz
```

---

## Installing Specific Packages

### Python 3

**Challenge**: Python requires shared libraries and is ~50MB+ with standard library.

#### Solution 1: Minimal Python (MicroPython)

```bash
# Build MicroPython for RISC-V (much smaller, ~300KB)
cd /home/user
git clone https://github.com/micropython/micropython
cd micropython
git submodule update --init

# Build for RISC-V
cd ports/unix
make CROSS_COMPILE=riscv64-linux-gnu- \
     CC=riscv64-linux-gnu-gcc \
     LD=riscv64-linux-gnu-ld

# Copy to rootfs
cp build-standard/micropython /home/user/rootfs/usr/bin/python

# Rebuild initramfs
cd /home/user/rootfs
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz
```

#### Solution 2: Full CPython (via Buildroot)

See "Option C" in Method 1, Step 2 above.

### PyTorch

**Challenge**: PyTorch is large (~500MB) and requires many dependencies.

**Recommendation**: Use a **minimal inference-only build** or **ONNX Runtime** instead.

#### Option A: PyTorch Mobile Lite

```bash
# Build PyTorch Lite for RISC-V (on host with cross-compiler)
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Configure for RISC-V minimal build
export CROSS_COMPILE=riscv64-linux-gnu-
export BUILD_LITE_INTERPRETER=1
python3 setup.py build

# Copy libtorch to rootfs
cp build/lib/libtorch_lite.so /home/user/rootfs/usr/lib/
```

#### Option B: ONNX Runtime (Lighter Alternative)

```bash
# ONNX Runtime is smaller and more portable
# Build or download RISC-V binary
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-riscv64.tar.gz
tar -xzf onnxruntime-riscv64.tar.gz

# Copy to rootfs
cp onnxruntime-riscv64/lib/*.so /home/user/rootfs/usr/lib/
cp -r onnxruntime-riscv64/include /home/user/rootfs/usr/

# Rebuild initramfs
cd /home/user/rootfs
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz
```

### LLAMA 2 7B Model

**Challenge**: LLAMA 2 7B model is ~13GB - **far too large for initramfs**.

**Solution**: Use **persistent storage** via:

#### Option 1: Virtual Disk Image

```bash
# Create a 20GB virtual disk image
qemu-img create -f qcow2 /home/user/models.qcow2 20G

# Attach to QEMU (modify run-linux.sh)
qemu-system-riscv64 \
    # ... existing options ...
    -drive file=/home/user/models.qcow2,if=virtio,format=qcow2 \
    # ... rest of command

# In Linux, mount the disk
mkdir /mnt/models
mount /dev/vda /mnt/models

# Copy model to disk (from host via networking or shared folder)
cp llama-2-7b.bin /mnt/models/
```

#### Option 2: VirtIO-9p Shared Folder

```bash
# Modify run-linux.sh to add shared folder:
qemu-system-riscv64 \
    # ... existing options ...
    -virtfs local,path=/host/models,mount_tag=models,security_model=none,id=models \
    # ... rest of command

# In Linux, mount shared folder
mkdir /mnt/models
mount -t 9p -o trans=virtio models /mnt/models

# Access models directly from host filesystem
ls /mnt/models/llama-2-7b.bin
```

#### Option 3: Network Mount (NFS/SMB)

```bash
# Configure networking in QEMU (requires network stack)
# Mount NFS share from host
mount -t nfs 192.168.1.100:/exports/models /mnt/models
```

**Recommended Approach for LLAMA**:
- Use VirtIO-9p shared folder for development
- Use virtual disk image for production/benchmarking
- Keep model files on host filesystem, not in initramfs

---

## Best Practices

### 1. Keep initramfs Small

```bash
# Check initramfs size before and after changes
ls -lh /home/user/initramfs.cpio.gz

# Target: < 100MB for fast boot
# Acceptable: < 500MB for full Linux environment
# Avoid: > 1GB (slow boot, high memory usage)
```

### 2. Separate Code from Data

```
rootfs/
├── apps/           # Your programs (10-50MB)
├── lib/            # Shared libraries (50-100MB)
└── [minimal Linux] # (50-100MB)

/mnt/models/        # Large datasets (mounted at runtime)
└── llama-2-7b.bin  # (13GB - NOT in initramfs!)
```

### 3. Use Static Linking for Small Programs

```bash
# Static binary = no shared library dependencies
riscv64-linux-gnu-gcc -static -o myapp myapp.c

# Verify no dynamic dependencies
riscv64-linux-gnu-readelf -d myapp
# Should show: "There is no dynamic section in this file"
```

### 4. Strip Debug Symbols

```bash
# Reduce binary size by removing debug symbols
riscv64-linux-gnu-strip /home/user/rootfs/apps/*

# Before strip: 2.5MB
# After strip:  500KB
```

### 5. Compression Testing

```bash
# Test different compression levels
cd /home/user/rootfs

# Fast compression (default, level 6)
find . | cpio -o -H newc 2>/dev/null | gzip > /tmp/initramfs-fast.cpio.gz

# Best compression (level 9)
find . | cpio -o -H newc 2>/dev/null | gzip -9 > /tmp/initramfs-best.cpio.gz

# Compare
ls -lh /tmp/initramfs-*.cpio.gz
```

### 6. Version Control Your RootFS

```bash
# Keep rootfs changes in Git
cd /home/user/rootfs
git init
git add .
git commit -m "Initial rootfs with Python 3.11"

# After installing new packages
git add apps/ lib/
git commit -m "Add PyTorch Lite for RISC-V"
```

---

## Troubleshooting

### Problem: "initramfs is too large, boot is slow"

**Solution**:
```bash
# Check size
du -sh /home/user/rootfs

# Identify large files
du -h /home/user/rootfs | sort -rh | head -20

# Remove unnecessary files
rm -rf /home/user/rootfs/usr/share/doc
rm -rf /home/user/rootfs/usr/share/man
```

### Problem: "Python module not found"

**Solution**:
```bash
# Check Python path in Linux
echo $PYTHONPATH

# Add to /init script in rootfs
echo 'export PYTHONPATH=/usr/lib/python3.11:/apps/python-modules' >> /home/user/rootfs/init

# Rebuild initramfs
cd /home/user/rootfs
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz
```

### Problem: "Shared library not found"

**Solution**:
```bash
# Check library dependencies
riscv64-linux-gnu-ldd /home/user/rootfs/apps/myapp

# Copy missing libraries to rootfs
cp /usr/riscv64-linux-gnu/lib/libmissing.so.1 /home/user/rootfs/lib/

# Update LD_LIBRARY_PATH in /init
echo 'export LD_LIBRARY_PATH=/lib:/usr/lib' >> /home/user/rootfs/init
```

### Problem: "Out of memory in QEMU"

**Solution**:
```bash
# Increase QEMU memory in run-linux.sh
qemu-system-riscv64 \
    -m 4G \  # Increase from default (512M or 1G)
    # ... rest of options
```

### Problem: "Can't persist changes across reboots"

**Solution**:
```bash
# Use persistent disk image (see LLAMA section above)
# OR always rebuild initramfs after making changes

# Quick rebuild script:
cat > /home/user/rebuild-initramfs.sh <<'EOF'
#!/bin/bash
cd /home/user/rootfs
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz
echo "initramfs rebuilt: $(ls -lh /home/user/initramfs.cpio.gz)"
EOF
chmod +x /home/user/rebuild-initramfs.sh

# Use it:
/home/user/rebuild-initramfs.sh
```

---

## Quick Reference

### Rebuild initramfs

```bash
cd /home/user/rootfs
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz
```

### Extract initramfs

```bash
mkdir /tmp/extract && cd /tmp/extract
gunzip -c /home/user/initramfs.cpio.gz | cpio -idmv
```

### Add a new application

```bash
# Compile
riscv64-linux-gnu-gcc -static -o myapp myapp.c

# Add to rootfs
cp myapp /home/user/rootfs/apps/

# Rebuild
cd /home/user/rootfs
find . | cpio -o -H newc 2>/dev/null | gzip > /home/user/initramfs.cpio.gz
```

### Check filesystem size

```bash
du -sh /home/user/rootfs
ls -lh /home/user/initramfs.cpio.gz
```

---

## Related Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - Initial setup and first boot
- [APP_DEVELOPMENT.md](APP_DEVELOPMENT.md) - Writing applications for Linux
- [BUILD_NOTES.md](BUILD_NOTES.md) - Cross-compilation details
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview

---

## Additional Resources

- [Linux initramfs documentation](https://www.kernel.org/doc/html/latest/filesystems/ramfs-rootfs-initramfs.html)
- [BusyBox project](https://busybox.net/) - Minimal Linux utilities
- [Buildroot](https://buildroot.org/) - Build custom Linux systems
- [RISC-V toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
