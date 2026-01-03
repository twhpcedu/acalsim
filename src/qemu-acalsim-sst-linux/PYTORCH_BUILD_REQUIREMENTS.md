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

# PyTorch Build Requirements Check

## Your Buildroot Configuration Analysis

Based on the buildroot configuration at `configs/acalsim_riscv64_defconfig`:

### ✅ Already Included (Good for PyTorch!)

#### Python Environment
- ✅ **Python 3.11.8** - Perfect! (PyTorch requires Python 3.8+)
- ✅ **pip** - Package installer
- ✅ **setuptools** - Build system
- ✅ **NumPy** - Core numerical library (required by PyTorch)

#### Python Core Modules
- ✅ **bzip2** - Compression support
- ✅ **readline** - Command line editing
- ✅ **SSL/OpenSSL** - HTTPS support (needed for downloading packages)
- ✅ **sqlite** - Database support
- ✅ **zlib** - Compression
- ✅ **pyexpat** - XML parsing
- ✅ **curses** - Terminal UI

#### C/C++ Compilers & Build Tools
- ✅ **Clang/LLVM** - C/C++ compiler (PyTorch can use this)
- ✅ **Binutils** - Binary utilities (ld, as, etc.)
- ✅ **Make** - Build automation
- ✅ **CMake** - Build system (PyTorch uses CMake!)
- ✅ **GCC with C++ support** - Alternative compiler
- ✅ **Fortran support** - For numerical libraries

#### Version Control & Tools
- ✅ **Git 2.43.0** - For cloning PyTorch repo
- ✅ **Vim** - Text editor
- ✅ **Patch** - Apply patches
- ✅ **diffutils** - Diff tools

#### Compression & Archive Tools
- ✅ **tar, gzip, bzip2, xz** - Archive extraction

#### Networking
- ✅ **DHCP client (dhcpcd)** - Auto network config
- ✅ **SSH server (Dropbear)** - Remote access
- ✅ **wget** - Download files
- ✅ **CA certificates** - HTTPS trust

#### System Tools
- ✅ **htop** - Process monitor
- ✅ **GDB** - Debugger
- ✅ **strace** - System call tracer

#### Required Libraries
- ✅ **libffi** - Foreign function interface (Python needs this)
- ✅ **OpenSSL** - Crypto library
- ✅ **readline** - Line editing
- ✅ **ncurses** - Terminal UI library

### ⚠️ Additional Packages Recommended for PyTorch Build

PyTorch has additional build dependencies that are **NOT** in your current buildroot config:

#### Critical Missing Packages
1. ❌ **BLAS/LAPACK libraries**
   - Options: OpenBLAS, ATLAS, or MKL
   - PyTorch needs these for linear algebra
   - **Recommendation**: Add `BR2_PACKAGE_OPENBLAS=y`

2. ❌ **patchelf** (for wheel building)
   - Needed to modify RPATH in compiled libraries
   - **Recommendation**: Add `BR2_PACKAGE_PATCHELF=y`

3. ❌ **ninja** (build tool)
   - PyTorch uses Ninja for faster builds
   - **Recommendation**: Add `BR2_PACKAGE_NINJA=y`

#### Optional but Helpful
4. ⚠️ **ccache** (compilation cache)
   - Speeds up rebuilds significantly
   - **Recommendation**: Add `BR2_CCACHE=y`

5. ⚠️ **Python typing extensions**
   - `BR2_PACKAGE_PYTHON_TYPING_EXTENSIONS=y`
   - PyTorch uses type hints

6. ⚠️ **Python pyyaml**
   - `BR2_PACKAGE_PYTHON_PYYAML=y`
   - PyTorch config files

7. ⚠️ **Python cffi**
   - `BR2_PACKAGE_PYTHON_CFFI=y`
   - C foreign function interface for Python

8. ⚠️ **Increased memory/disk**
   - Building PyTorch from source requires ~8GB RAM and ~15GB disk
   - Your QEMU is configured with 8GB RAM ✅
   - Check available disk space in rootfs

## Updated Buildroot Configuration for PyTorch

Add these lines to `configs/acalsim_riscv64_defconfig`:

```bash
# Additional packages for PyTorch build
BR2_PACKAGE_OPENBLAS=y
BR2_PACKAGE_PATCHELF=y
BR2_PACKAGE_NINJA=y
BR2_CCACHE=y

# Additional Python packages
BR2_PACKAGE_PYTHON_TYPING_EXTENSIONS=y
BR2_PACKAGE_PYTHON_PYYAML=y
BR2_PACKAGE_PYTHON_CFFI=y

# For better performance
BR2_PACKAGE_PYTHON3_PYEXPAT=y
```

## Summary

### Current Status
- ✅ **Python 3.11.8** with pip, setuptools, NumPy
- ✅ **Git, Vim, Make, CMake** all present
- ✅ **Network support** fully configured
- ✅ **Clang/LLVM toolchain** for compilation
- ✅ **All core Python modules** for development

### What You Need to Add
To build PyTorch from source, you should add:
1. **OpenBLAS** (critical for linear algebra)
2. **patchelf** (for building wheels)
3. **ninja** (faster build tool)
4. **ccache** (optional, speeds up builds)
5. Additional Python packages (typing-extensions, pyyaml, cffi)

## Verification Commands

Once booted in QEMU, verify what's available:

```bash
# Check Python version
python3 --version
# Expected: Python 3.11.8

# Check pip
pip3 --version

# Check NumPy
python3 -c "import numpy; print(numpy.__version__)"

# Check compiler
clang --version
gcc --version

# Check build tools
make --version
cmake --version
git --version

# Check available disk space
df -h
# Need at least 15GB free for PyTorch build

# Check available memory
free -h
# Need at least 8GB RAM (you have this in QEMU config)
```

## Recommendation

### Option 1: Rebuild with Additional Packages (Recommended)
```bash
cd /home/user/buildroot-llama/buildroot-2024.02

# Update config
cat >> configs/acalsim_riscv64_defconfig << 'EOF'
# Additional packages for PyTorch build
BR2_PACKAGE_OPENBLAS=y
BR2_PACKAGE_PATCHELF=y
BR2_PACKAGE_NINJA=y
BR2_CCACHE=y
BR2_PACKAGE_PYTHON_TYPING_EXTENSIONS=y
BR2_PACKAGE_PYTHON_PYYAML=y
BR2_PACKAGE_PYTHON_CFFI=y
EOF

# Reload config
make acalsim_riscv64_defconfig

# Rebuild (will only build new packages)
make -j$(nproc)

# Create new initramfs symlink
ln -sf /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz \
       /home/user/initramfs-buildroot.cpio.gz
```

### Option 2: Try with Current Config (May Work)
Your current buildroot already has most of what's needed. You could try building PyTorch with what you have and install missing packages via pip inside QEMU:

```bash
# Inside QEMU Linux
pip3 install typing-extensions pyyaml cffi
```

However, **OpenBLAS is critical** and should be in the rootfs, not installed via pip.

## Build Time Estimate

- **Current buildroot**: Already built ✅
- **Adding new packages**: ~30-60 minutes additional build time
- **PyTorch from source in QEMU**: 4-8 hours (RISC-V is slower)

---

**Recommendation**: Add OpenBLAS at minimum before attempting PyTorch build.
