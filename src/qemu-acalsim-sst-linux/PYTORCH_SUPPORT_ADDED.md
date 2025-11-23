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

# PyTorch Support Added to Buildroot Script

## Summary

✅ **I've updated the `setup_buildroot_python.sh` script with PyTorch build dependencies.**

## What Was Added

### Critical Packages for PyTorch:
1. **BR2_PACKAGE_OPENBLAS=y** - Linear algebra library (BLAS/LAPACK)
2. **BR2_PACKAGE_PATCHELF=y** - Modify ELF binaries (for building wheels)
3. **BR2_PACKAGE_NINJA=y** - Fast build system (PyTorch uses this)
4. **BR2_CCACHE=y** - Compilation cache (speeds up rebuilds)

### Python Packages:
5. **BR2_PACKAGE_PYTHON_CFFI=y** - C Foreign Function Interface
6. **BR2_PACKAGE_PYTHON_PYCPARSER=y** - C parser for CFFI

## Already Included (No Changes Needed):
- ✅ Python 3.11.8
- ✅ pip, setuptools, NumPy
- ✅ Clang/LLVM, GCC with C++
- ✅ CMake, Make, Git
- ✅ Network support (DHCP, SSH, wget)
- ✅ All compression tools

## Files Modified

```
setup_buildroot_python.sh           ← Updated with PyTorch packages
setup_buildroot_python.sh.backup    ← Original version (before Step 4/5 fix)
setup_buildroot_python.sh.pre-pytorch ← Version before PyTorch additions
```

## How to Rebuild with PyTorch Support

### Option 1: Incremental Rebuild (Faster - Recommended)
```bash
cd /home/user/buildroot-llama/buildroot-2024.02

# Update config to include PyTorch packages
cat >> configs/acalsim_riscv64_defconfig << 'EOF'
# PyTorch build dependencies
BR2_PACKAGE_OPENBLAS=y
BR2_PACKAGE_PATCHELF=y
BR2_PACKAGE_NINJA=y
BR2_CCACHE=y
BR2_PACKAGE_PYTHON_CFFI=y
BR2_PACKAGE_PYTHON_PYCPARSER=y
EOF

# Reload config
make acalsim_riscv64_defconfig

# Build only new packages (~30-60 minutes)
make -j$(nproc)

# Update initramfs
ln -sf /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz \
       /home/user/initramfs-buildroot.cpio.gz

# Boot and test
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

### Option 2: Full Clean Build (Slower)
```bash
# Clean everything
cd /home/user/buildroot-llama/buildroot-2024.02
make clean

# Run updated script (2-4 hours)
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./setup_buildroot_python.sh
```

## Verification Inside QEMU

Once booted, verify PyTorch dependencies:

```bash
# Check Python and NumPy
python3 --version
python3 -c "import numpy; print(numpy.__version__)"

# Check if OpenBLAS is available
ls /usr/lib/libopenblas* || ls /usr/lib/riscv64-linux-gnu/libopenblas*

# Check build tools
clang --version
cmake --version
ninja --version
patchelf --version

# Check disk space (need ~15GB for PyTorch build)
df -h

# Check memory
free -h
```

## Next Steps: Building PyTorch

Once buildroot is rebuilt with PyTorch support:

```bash
# Inside QEMU Linux:

# 1. Install additional Python packages via pip
pip3 install typing-extensions pyyaml

# 2. Clone PyTorch
git clone --depth 1 --recursive https://github.com/pytorch/pytorch
cd pytorch

# 3. Set environment variables
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0

# 4. Build PyTorch (4-8 hours on RISC-V)
python3 setup.py install

# 5. Test PyTorch
python3 -c "import torch; print(torch.__version__)"
```

## Build Time Estimates

| Task | Time |
|------|------|
| Incremental buildroot rebuild | 30-60 min |
| Full buildroot rebuild | 2-4 hours |
| PyTorch compilation in QEMU | 4-8 hours |
| **Total (incremental)** | **5-9 hours** |
| **Total (full rebuild)** | **6-12 hours** |

## Comparison: Buildroot vs Debian

### Buildroot (Recommended) ✅
- ✅ Already working
- ✅ Now has PyTorch support
- ✅ Smaller, faster
- ✅ Good for research
- ⚠️ Longer build time

### Debian (Alternative)
- ✅ Easy package management (apt)
- ✅ More packages available
- ✅ Standard Linux
- ⚠️ Much larger (2-5GB vs 500MB)
- ⚠️ Need to setup from scratch

**Recommendation**: Use updated buildroot. Switch to Debian only if buildroot PyTorch build fails.

## Backup Files

If you need to revert:

```bash
# Revert to version before PyTorch additions
cp setup_buildroot_python.sh.pre-pytorch setup_buildroot_python.sh

# Revert to original version (with Steps 4/5 issues)
cp setup_buildroot_python.sh.backup setup_buildroot_python.sh
```

---

**Status**: ✅ Script updated and ready to build
**Next Action**: Rebuild buildroot with PyTorch support (~30-60 min)
**Created**: 2025-11-20
