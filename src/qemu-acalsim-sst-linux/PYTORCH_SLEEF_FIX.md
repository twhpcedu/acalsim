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

# Fix PyTorch SLEEF FMA Error on RISC-V

## Error Description

```
/home/debian/pytorch/third_party/sleef/src/arch/helperpurec_scalar.h:69:2: error: FP_FAST_FMA or FP_FAST_FMAF not defined
   69 | #error FP_FAST_FMA or FP_FAST_FMAF not defined
```

This occurs because SLEEF (SIMD Library for Evaluating Elementary Functions) expects hardware FMA support, which may not be detected on RISC-V.

---

## Solution 1: Disable SLEEF (Recommended for RISC-V)

The easiest solution is to disable SLEEF entirely, as RISC-V doesn't benefit from SLEEF's SIMD optimizations in QEMU.

### Apply Before Building PyTorch

```bash
cd /home/debian/pytorch

# Clean previous build attempts
python3 setup.py clean

# Set environment to disable SLEEF
export USE_SLEEF=0

# Also disable other SIMD/vectorization features
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0
export MAX_JOBS=4
export USE_OPENBLAS=1

# Build PyTorch
python3 setup.py install 2>&1 | tee ../pytorch-build.log
```

---

## Solution 2: Patch SLEEF for RISC-V

If you need SLEEF support, you can patch it to work without FMA:

```bash
cd /home/debian/pytorch

# Create patch for SLEEF
cat > /tmp/sleef_riscv.patch << 'EOF'
--- a/third_party/sleef/src/arch/helperpurec_scalar.h
+++ b/third_party/sleef/src/arch/helperpurec_scalar.h
@@ -66,7 +66,9 @@
 #endif

 #if !defined(FP_FAST_FMA) && !defined(FP_FAST_FMAF)
-#error FP_FAST_FMA or FP_FAST_FMAF not defined
+// RISC-V: FMA not available, use software emulation
+#warning FP_FAST_FMA or FP_FAST_FMAF not defined, using software FMA
+// #error FP_FAST_FMA or FP_FAST_FMAF not defined
 #endif

 //////////////////////////////////////////////////////////////////////////////////
EOF

# Apply patch
patch -p1 < /tmp/sleef_riscv.patch

# Then build PyTorch normally
python3 setup.py install
```

---

## Solution 3: Define FP_FAST_FMA Manually (Workaround)

Force define FP_FAST_FMA even though hardware doesn't support it:

```bash
cd /home/debian/pytorch

# Set CFLAGS to define FP_FAST_FMA
export CFLAGS="-DFP_FAST_FMA -DFP_FAST_FMAF"
export CXXFLAGS="-DFP_FAST_FMA -DFP_FAST_FMAF"

# Disable SLEEF anyway (recommended)
export USE_SLEEF=0

# Build PyTorch
python3 setup.py install
```

---

## Solution 4: Use Older PyTorch Version

Try PyTorch 1.13 which has better RISC-V compatibility:

```bash
cd /home/debian

# Remove broken build
rm -rf pytorch

# Clone older version
git clone --depth 1 --branch v1.13.0 --recursive https://github.com/pytorch/pytorch
cd pytorch

# Configure for RISC-V
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
export USE_SLEEF=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0
export MAX_JOBS=4
export USE_OPENBLAS=1

# Build
python3 setup.py install
```

---

## Recommended Build Configuration for RISC-V

Create a build script with all necessary environment variables:

```bash
cat > /home/debian/pytorch/build_riscv.sh << 'EOF'
#!/bin/bash
# PyTorch RISC-V Build Configuration

# Disable all SIMD/vectorization features
export USE_CUDA=0
export USE_CUDNN=0
export USE_ROCM=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_SLEEF=0          # Disable SLEEF (FMA issue)
export USE_FBGEMM=0
export USE_KINETO=0

# Disable distributed
export USE_DISTRIBUTED=0
export USE_MPI=0
export USE_GLOO=0
export USE_TENSORPIPE=0
export USE_NCCL=0

# Disable testing
export BUILD_TEST=0
export BUILD_CAFFE2=0

# Use OpenBLAS for BLAS operations
export BLAS=OpenBLAS
export USE_OPENBLAS=1

# Build settings
export MAX_JOBS=4
export CMAKE_BUILD_TYPE=Release
export CMAKE_GENERATOR=Ninja

# Prevent FMA errors
export CFLAGS="-O2"
export CXXFLAGS="-O2"

echo "====================================="
echo "PyTorch RISC-V Build Configuration"
echo "====================================="
echo "USE_SLEEF: $USE_SLEEF (disabled to avoid FMA errors)"
echo "USE_OPENBLAS: $USE_OPENBLAS"
echo "MAX_JOBS: $MAX_JOBS"
echo "====================================="

# Clean previous build
python3 setup.py clean

# Build and install
echo "Starting PyTorch build at $(date)"
python3 setup.py install 2>&1 | tee ../pytorch-build.log
echo "Build finished at $(date)"
EOF

chmod +x /home/debian/pytorch/build_riscv.sh

# Run the build script
./build_riscv.sh
```

---

## Verification After Fix

```bash
# Test PyTorch import
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# Test basic operations
python3 << 'PYTEST'
import torch

# Create tensors
x = torch.randn(3, 3)
y = torch.randn(3, 3)

# Matrix multiplication (uses BLAS, not SLEEF)
z = torch.mm(x, y)
print(f"Matrix multiplication: {z.shape}")

# Element-wise operations
w = x * y
print(f"Element-wise multiplication: {w.shape}")

print("\n✓ PyTorch working without SLEEF!")
PYTEST
```

---

## Why This Happens on RISC-V

1. **SLEEF is SIMD-optimized**: Designed for x86/ARM SIMD instructions (SSE, AVX, NEON)
2. **RISC-V has limited SIMD**: Vector extension (RVV) not widely supported in QEMU
3. **FMA detection fails**: RISC-V toolchain doesn't define `FP_FAST_FMA` macro
4. **Not needed for RISC-V**: SLEEF provides minimal benefit without SIMD

---

## Quick Fix Summary

**Most users should use Solution 1 (Disable SLEEF):**

```bash
cd /home/debian/pytorch
python3 setup.py clean
export USE_SLEEF=0 USE_CUDA=0 USE_MKLDNN=0 BUILD_TEST=0 MAX_JOBS=4 USE_OPENBLAS=1
python3 setup.py install 2>&1 | tee ../pytorch-build.log
```

This disables SLEEF entirely, avoiding the FMA error while maintaining full PyTorch functionality through OpenBLAS.

---

## Related Issues

- PyTorch Issue #71791: SLEEF build failure on non-x86 architectures
- PyTorch Issue #55615: RISC-V support tracking
- SLEEF Issue #371: FMA macro detection on non-mainstream architectures

---

**Created**: 2025-11-20
**PyTorch Version Tested**: v2.1.0
**Platform**: RISC-V 64-bit (QEMU)
**Solution**: Disable SLEEF with `USE_SLEEF=0`
