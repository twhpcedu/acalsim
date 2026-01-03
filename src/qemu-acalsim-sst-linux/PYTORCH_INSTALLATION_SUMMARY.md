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

# PyTorch Installation for RISC-V - Quick Guide

## Answer: You Build PyTorch Inside QEMU

PyTorch doesn't have pre-compiled RISC-V binaries, so you:
1. ✅ Rebuild buildroot with PyTorch dependencies (30-60 min)
2. ✅ Boot into QEMU
3. ✅ Build PyTorch from source inside QEMU (4-8 hours)

---

## Quick Start

### 1. Rebuild Buildroot (Do This Once)

```bash
cd /home/user/buildroot-llama/buildroot-2024.02

# Add PyTorch dependencies
cat >> configs/acalsim_riscv64_defconfig << 'EOF'
BR2_PACKAGE_OPENBLAS=y
BR2_PACKAGE_PATCHELF=y
BR2_PACKAGE_NINJA=y
BR2_CCACHE=y
BR2_PACKAGE_PYTHON_CFFI=y
BR2_PACKAGE_PYTHON_PYCPARSER=y
EOF

# Rebuild
make acalsim_riscv64_defconfig
make -j$(nproc)

# Update initramfs
ln -sf /home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.cpio.gz \
       /home/user/initramfs-buildroot.cpio.gz
```

### 2. Boot QEMU

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_initramfs.sh
```

### 3. Build PyTorch Inside QEMU

Once logged into Linux:

```bash
# Install Python dependencies
pip3 install typing-extensions pyyaml requests packaging

# Clone PyTorch
cd /root
git clone --depth 1 --branch v2.1.0 --recursive https://github.com/pytorch/pytorch
cd pytorch

# Configure build
export USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0 USE_DISTRIBUTED=0
export BUILD_TEST=0 MAX_JOBS=4 USE_OPENBLAS=1

# Build (4-8 hours)
python3 setup.py install

# Test
python3 -c "import torch; print(torch.__version__)"
```

---

## What Gets Installed in Buildroot

### From Buildroot Packages:
- ✅ Python 3.11.8
- ✅ NumPy (pre-built)
- ✅ OpenBLAS (linear algebra library)
- ✅ GCC, Clang, CMake, Ninja
- ✅ Git, pip, setuptools
- ✅ patchelf, ccache

### Built Inside QEMU:
- PyTorch itself (source build)

---

## Timeline

| Step | Time | What Happens |
|------|------|--------------|
| Buildroot rebuild | 30-60 min | Adds PyTorch dependencies |
| Boot QEMU | < 1 min | Start Linux |
| Clone PyTorch | 5-10 min | Download source |
| Build PyTorch | **4-8 hours** | Compile PyTorch |
| **Total** | **~5-9 hours** | Complete installation |

---

## Why Not Add PyTorch to Buildroot Directly?

**Technical Reasons:**
1. PyTorch isn't in buildroot's package repository
2. RISC-V support is experimental
3. Build requires source compilation anyway
4. Easier to debug inside running system

**Practical Reasons:**
- Building inside QEMU gives you `pip` for dependencies
- You can test incrementally
- Easier to fix issues when they arise
- Can save/resume if something fails

---

## What Dependencies Do

| Package | Why PyTorch Needs It |
|---------|---------------------|
| **OpenBLAS** | Matrix operations (CRITICAL - won't work without it) |
| **NumPy** | PyTorch uses NumPy arrays internally |
| **CMake** | PyTorch build system |
| **Ninja** | Faster alternative to Make |
| **patchelf** | Fix library paths in built binaries |
| **ccache** | Cache compiled objects (speeds up rebuilds) |
| **CFFI** | Python C Foreign Function Interface |

---

## Verification Steps

After PyTorch builds, verify it works:

```bash
# Import test
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# Basic tensor operations
python3 << 'EOF'
import torch
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = torch.mm(x, y)
print(f"Matrix multiplication works: {z.shape}")
EOF

# Simple neural network
python3 << 'EOF'
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
x = torch.randn(5, 10)
output = model(x)
print(f"Neural network works: {output.shape}")
EOF
```

---

## Alternative: Pre-built Wheel (If Available)

If someone provides a pre-built PyTorch wheel for RISC-V:

```bash
# Inside QEMU
pip3 install torch-2.1.0-cp311-cp311-linux_riscv64.whl
```

**But**: No official RISC-V wheels exist yet, so source build is required.

---

## Space Requirements

| Component | Size |
|-----------|------|
| Buildroot rootfs | ~500MB |
| PyTorch source | ~1-2GB |
| Build artifacts | ~5-10GB |
| Installed PyTorch | ~500MB-1GB |
| **Total needed** | **~15GB** |

---

## Documentation

- 📄 **BUILD_PYTORCH_IN_QEMU.md** - Detailed build instructions
- 📄 **PYTORCH_BUILD_REQUIREMENTS.md** - What's needed and why
- 📄 **PYTORCH_SUPPORT_ADDED.md** - What I added to buildroot

---

## Quick Decision Guide

**Q: Can I just `pip install torch`?**
A: No - no pre-built RISC-V wheels exist.

**Q: Why not add PyTorch to buildroot package?**
A: Too complex, easier to build inside QEMU.

**Q: How long does it take?**
A: 5-9 hours total (mostly PyTorch compilation).

**Q: Can I speed it up?**
A: Use ccache (already configured), reduce MAX_JOBS if RAM limited.

**Q: What if build fails?**
A: Check BUILD_PYTORCH_IN_QEMU.md troubleshooting section.

---

**Summary**: Rebuild buildroot with dependencies (30-60 min), then build PyTorch from source inside QEMU (4-8 hours). Total: ~5-9 hours.
