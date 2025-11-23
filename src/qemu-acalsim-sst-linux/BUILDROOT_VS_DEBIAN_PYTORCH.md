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

# Buildroot vs Debian for PyTorch - Analysis & Recommendation

## Your Questions Answered

### Q1: Does the current Linux kernel work with Debian?
**Answer: ✅ YES!**

Your kernel (Linux 6.18.0-rc6) has all required support:
- ✅ EXT4 filesystem (Debian uses this)
- ✅ VirtIO block device (for disk images)
- ✅ 9P filesystem (for host sharing)
- ✅ Network stack fully configured

### Q2: Should I use Buildroot or Debian for PyTorch?
**Answer: Both can work, but they have different trade-offs.**

## Comparison: Buildroot vs Debian

### Option 1: Updated Buildroot (Recommended for Your Use Case)

#### ✅ Pros:
1. **Already built and working** - You have it now
2. **Optimized for embedded** - Smaller, faster boot
3. **Custom-built** - Only what you need
4. **No package conflicts** - Clean environment
5. **Good for research/simulation** - Controlled environment
6. **Faster builds** - Incremental builds work well
7. **I just added PyTorch support** - Now includes:
   - OpenBLAS (linear algebra)
   - patchelf (wheel building)
   - ninja (fast build tool)
   - ccache (build caching)
   - Python cffi, pycparser

#### ⚠️ Cons:
1. **Longer initial setup** (2-4 hours build time)
2. **Limited package manager** - Can't `apt install` easily
3. **Manual dependency management** - Need to rebuild for new packages
4. **Smaller community** - Less PyTorch+Buildroot documentation

#### Recommendation:
**Use Buildroot if:**
- You want full control
- You're doing research/benchmarking
- You need a minimal, reproducible environment
- You're okay with longer build times for customization

---

### Option 2: Debian RISC-V

#### ✅ Pros:
1. **Full package manager** - `apt install` anything
2. **Huge package repository** - Pre-built packages
3. **Easier to use** - Standard Linux experience
4. **More documentation** - Common for PyTorch development
5. **Pre-built PyTorch** - May have RISC-V packages
6. **Easier debugging** - Standard tools available
7. **Quick to get started** - Download and boot

#### ⚠️ Cons:
1. **Much larger** - 2-5GB+ rootfs vs 233MB buildroot
2. **Slower boot** - More services to start
3. **More bloat** - Many unnecessary packages
4. **Less optimized** - General purpose, not embedded
5. **May need more RAM/disk** - 10GB+ disk, 8GB RAM minimum

#### Recommendation:
**Use Debian if:**
- You want quick experimentation
- You need lots of different packages
- You prefer standard Linux development
- You have plenty of disk space (15GB+)

---

## My Recommendation: **Start with Updated Buildroot**

### Why Buildroot is Better for Your Case:

1. **✅ Already Working**
   - You have it built and booting
   - Login prompt working
   - Network configured

2. **✅ Now Has PyTorch Support** (I just added)
   - OpenBLAS for linear algebra
   - All build tools (clang, cmake, ninja)
   - Python 3.11.8 with NumPy
   - patchelf for wheel building

3. **✅ Good for Simulation/Research**
   - Minimal overhead
   - Reproducible environment
   - Fast boot times
   - Known dependencies

4. **✅ Incremental Rebuild**
   - Adding PyTorch packages: ~30-60 min
   - Not starting from scratch

### When to Switch to Debian:

Switch to Debian **only if** you encounter these issues:
1. ❌ PyTorch build fails with missing dependencies
2. ❌ You need many additional packages frequently
3. ❌ You want pre-built PyTorch binaries (if available)
4. ❌ Build times become too long (4+ hour rebuilds)

---

## Updated Buildroot Configuration

### What I Added to `setup_buildroot_python.sh`:

```bash
# PyTorch build dependencies
BR2_PACKAGE_OPENBLAS=y          # Linear algebra library (CRITICAL)
BR2_PACKAGE_PATCHELF=y          # Modify ELF binaries (for wheels)
BR2_PACKAGE_NINJA=y             # Fast build system
BR2_CCACHE=y                    # Compilation cache

# Additional Python packages for PyTorch
BR2_PACKAGE_PYTHON_CFFI=y       # C Foreign Function Interface
BR2_PACKAGE_PYTHON_PYCPARSER=y  # C parser for CFFI
```

### Already Included (from before):
```bash
BR2_PACKAGE_PYTHON3=y           # Python 3.11.8
BR2_PACKAGE_PYTHON_PIP=y        # Package installer
BR2_PACKAGE_PYTHON_SETUPTOOLS=y # Build system
BR2_PACKAGE_PYTHON_NUMPY=y      # Numerical library
BR2_PACKAGE_LLVM=y              # LLVM compiler
BR2_PACKAGE_CLANG=y             # Clang C++ compiler
BR2_PACKAGE_CMAKE=y             # CMake build system
BR2_PACKAGE_GIT=y               # Git version control
BR2_PACKAGE_MAKE=y              # Make build tool
```

---

## Next Steps with Updated Buildroot

### Option A: Rebuild Buildroot with PyTorch Support (Recommended)

```bash
cd /home/user/buildroot-llama/buildroot-2024.02

# Clean previous build (keeps config)
make clean

# Reload updated config (includes PyTorch packages)
make acalsim_riscv64_defconfig

# Rebuild (will build new packages: ~30-60 minutes)
make -j$(nproc)

# Update initramfs symlink
ln -sf /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz \
       /home/user/initramfs-buildroot.cpio.gz

# Test boot
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

### Option B: Run New Buildroot Script (Clean Build)

```bash
# This will build everything from scratch (2-4 hours)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./setup_buildroot_python.sh
```

### Option C: Try Current Buildroot First

You could try building PyTorch with the current buildroot and see what fails:

```bash
# Boot current system
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh

# Inside QEMU:
# 1. Check what's available
python3 --version
python3 -c "import numpy; print(numpy.__version__)"

# 2. Try installing PyTorch dependencies via pip
pip3 install typing-extensions pyyaml cffi

# 3. Attempt to build PyTorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python3 setup.py install

# If it fails, note what's missing, then rebuild buildroot
```

---

## Debian Alternative (If Needed)

### How to Use Debian Instead:

1. **Download Debian RISC-V rootfs**
```bash
# This is larger (2-5GB) and takes longer to download
wget https://cdimage.debian.org/cdimage/ports/latest/riscv64/debian-sid-riscv64-rootfs.tar.xz
```

2. **Create disk image**
```bash
qemu-img create -f qcow2 debian-riscv64.qcow2 20G
# Then format and populate (complex process)
```

3. **Boot with Debian**
```bash
# Similar to run_qemu_persistent.sh but with Debian disk
```

4. **Install PyTorch**
```bash
# Inside Debian
apt update
apt install python3-pip python3-dev build-essential
pip3 install torch --no-binary torch
```

---

## Summary: My Recommendation

### **Start with Updated Buildroot** (I already modified the script)

**Reasoning:**
1. ✅ Already have it working
2. ✅ I added all PyTorch dependencies
3. ✅ Faster, lighter, more controlled
4. ✅ Good for research/simulation
5. ✅ Can always switch to Debian later if needed

**Build Time:**
- Incremental rebuild with PyTorch packages: **~30-60 minutes**
- PyTorch compilation in QEMU: **4-8 hours** (RISC-V is slow)

**Disk Space:**
- Buildroot rootfs: ~500MB
- PyTorch source + build: ~10-15GB
- Total needed: ~20GB

### **Switch to Debian Only If:**
1. Buildroot PyTorch build fails repeatedly
2. You need many additional packages
3. You want easier package management
4. Pre-built RISC-V PyTorch packages exist

---

## Files Modified

- ✅ `setup_buildroot_python.sh` - Added PyTorch support
- 📄 `setup_buildroot_python.sh.pre-pytorch` - Backup before changes
- 📄 `BUILDROOT_VS_DEBIAN_PYTORCH.md` - This document

---

**Recommendation**: Rebuild buildroot with PyTorch support (~30-60 min), then try building PyTorch. Switch to Debian only if you encounter insurmountable issues.
