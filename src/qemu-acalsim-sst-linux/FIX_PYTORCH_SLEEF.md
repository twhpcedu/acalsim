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

# URGENT FIX: PyTorch SLEEF FMA Error

## The Problem

Your PyTorch build keeps failing with SLEEF FMA errors even after setting `USE_SLEEF=0` because CMake cached the old configuration in the `build/` directory.

## The Solution - Copy and Paste This Entire Block

**Stop the current build (Ctrl+C), then run this entire script in one go:**

```bash
cd /home/debian/pytorch

# Stop any running builds
pkill -9 -f "python3 setup.py" || true

# NUCLEAR CLEAN - Remove ALL build artifacts
echo "Cleaning build artifacts..."
rm -rf build/
rm -rf torch.egg-info/
rm -rf dist/
rm -rf *.egg-info/
python3 setup.py clean || true
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# Verify build directory is gone
if [ -d "build" ]; then
    echo "ERROR: Failed to delete build/ directory"
    exit 1
fi

echo "✓ Build directory cleaned"

# Set ALL required environment variables
export USE_SLEEF=0
export USE_CUDA=0
export USE_CUDNN=0
export USE_ROCM=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_FBGEMM=0
export USE_KINETO=0
export USE_DISTRIBUTED=0
export USE_MPI=0
export USE_GLOO=0
export USE_TENSORPIPE=0
export USE_NCCL=0
export BUILD_TEST=0
export BUILD_CAFFE2=0
export BLAS=OpenBLAS
export USE_OPENBLAS=1
export MAX_JOBS=4
export CMAKE_BUILD_TYPE=Release

echo ""
echo "====================================="
echo "PyTorch RISC-V Build Configuration"
echo "====================================="
echo "USE_SLEEF: $USE_SLEEF (DISABLED)"
echo "USE_OPENBLAS: $USE_OPENBLAS"
echo "MAX_JOBS: $MAX_JOBS"
echo "====================================="
echo ""
echo "Starting clean build at $(date)"
echo "This will take 4-8 hours..."
echo ""

# Start the build
python3 setup.py install 2>&1 | tee ../pytorch-build-clean.log
```

## Alternative: Use pip to Force Clean Install

If the above still fails, try using pip which handles dependencies better:

```bash
cd /home/debian/pytorch

# Clean everything
rm -rf build/ dist/ *.egg-info/

# Set environment
export USE_SLEEF=0 USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0
export USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0
export USE_DISTRIBUTED=0 BUILD_TEST=0
export MAX_JOBS=4 USE_OPENBLAS=1

# Use pip to install
pip3 install -v . 2>&1 | tee ../pytorch-build-pip.log
```

## Check if SLEEF is Being Built

After starting the new build, watch the first 100 lines of output:

```bash
# In another terminal
tail -f ../pytorch-build-clean.log | head -100
```

You should see CMake configuration output. Look for lines mentioning SLEEF:
- ❌ BAD: `-- Building SLEEF` or `-- SLEEF: ON`
- ✅ GOOD: No mention of SLEEF, or `-- SLEEF: OFF`

## If It Still Tries to Build SLEEF

The issue might be hardcoded in CMakeLists.txt. Apply this patch:

```bash
cd /home/debian/pytorch

# Patch CMakeLists.txt to disable SLEEF
sed -i 's/option(USE_SLEEF "Use SLEEF" ON)/option(USE_SLEEF "Use SLEEF" OFF)/' CMakeLists.txt

# Clean and rebuild
rm -rf build/
export USE_SLEEF=0 MAX_JOBS=4 USE_OPENBLAS=1 BUILD_TEST=0
python3 setup.py install 2>&1 | tee ../pytorch-build-patched.log
```

## Nuclear Option: Patch SLEEF to Skip FMA Check

If nothing else works, patch SLEEF to ignore the FMA error:

```bash
cd /home/debian/pytorch

# Backup the file
cp third_party/sleef/src/arch/helperpurec_scalar.h third_party/sleef/src/arch/helperpurec_scalar.h.bak

# Comment out the error
sed -i 's/#error FP_FAST_FMA or FP_FAST_FMAF not defined/\/\/ #error FP_FAST_FMA or FP_FAST_FMAF not defined/' third_party/sleef/src/arch/helperpurec_scalar.h

# Define the macros manually
sed -i '69a #define FP_FAST_FMA 1\n#define FP_FAST_FMAF 1' third_party/sleef/src/arch/helperpurec_scalar.h

# Clean and rebuild
rm -rf build/
export MAX_JOBS=4 USE_OPENBLAS=1 BUILD_TEST=0
python3 setup.py install 2>&1 | tee ../pytorch-build-patched.log
```

## Verify the Fix Worked

After cleaning and restarting the build, you should see:

```
[1/1455] Building CXX object ...
[2/1455] Building CXX object ...
...
```

And it should get past step 311 (where it was failing) without any SLEEF errors.

## Common Mistakes

1. **Not stopping the build first** - Ctrl+C to stop it
2. **Not actually deleting build/** - Run `rm -rf build/` explicitly
3. **Setting variables after the build starts** - Variables must be set BEFORE running setup.py
4. **Not cleaning completely** - Use the full clean commands above

---

**Created**: 2025-11-20
**Issue**: SLEEF FMA error on RISC-V
**Solution**: Complete clean + USE_SLEEF=0 + rebuild
