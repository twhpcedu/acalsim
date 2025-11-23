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

# Complete Guide: Building PyTorch on RISC-V Debian

This guide provides step-by-step instructions for building PyTorch 2.4 from source on RISC-V architecture running Debian Linux.

## Prerequisites

### System Requirements
- RISC-V 64-bit system (physical or QEMU)
- Debian sid (unstable) or similar
- At least 32GB RAM (recommended for linking)
- At least 100GB disk space
- Python 3.13

### Required Packages
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    clang \
    llvm \
    lld \
    libopenblas-dev \
    libomp-dev \
    ccache
```

### Python Dependencies
```bash
pip3 install --break-system-packages \
    numpy \
    typing-extensions \
    pyyaml \
    requests \
    packaging \
    sympy \
    filelock \
    jinja2 \
    networkx \
    fsspec
```

## Step 1: Clone PyTorch Repository

```bash
cd ~
git clone --depth 1 --branch v2.4.0 --recursive https://github.com/pytorch/pytorch
cd pytorch
```

**Important**: Use `--recursive` to clone all submodules. This is essential.

## Step 2: Apply RISC-V Compatibility Patches

### Patch 1: SLEEF FMA Fix
RISC-V doesn't define `FP_FAST_FMA` macros, so we need to patch SLEEF:

```bash
# Edit third_party/sleef/src/arch/helperpurec_scalar.h
# Around line 69, change:
#   #error FP_FAST_FMA or FP_FAST_FMAF not defined
# To:
#   // #error FP_FAST_FMA or FP_FAST_FMAF not defined
#   #define FP_FAST_FMA 1
#   #define FP_FAST_FMAF 1

sed -i 's/#error FP_FAST_FMA or FP_FAST_FMAF not defined/\/\/ #error FP_FAST_FMA or FP_FAST_FMAF not defined\n#define FP_FAST_FMA 1\n#define FP_FAST_FMAF 1/' \
    third_party/sleef/src/arch/helperpurec_scalar.h
```

### Patch 2: CMake Warning Suppression
Suppress compiler warnings that cause build failures:

```bash
cat > /tmp/cmake_patch.txt << 'EOF'
# Disable problematic warnings for RISC-V/Clang
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=dangling-reference -Wno-error=tautological-compare -Wno-unknown-warning-option")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error -Wno-unknown-warning-option")
EOF

sed -i '/^project(Torch CXX C)/r /tmp/cmake_patch.txt' CMakeLists.txt
```

### Patch 3: Fix Missing Headers (if needed)
Some files may need additional headers:

```bash
# Add cstdint to files that need it
for file in \
    caffe2/utils/string_utils.cc \
    c10/util/ThreadLocalDebugInfo.h \
    torch/csrc/jit/passes/quantization/quantization_type.h
do
    if [ -f "$file" ]; then
        sed -i '1i #include <cstdint>' "$file"
    fi
done

# Add stdexcept to files that need it
for file in \
    torch/csrc/jit/runtime/logging.cpp \
    torch/csrc/lazy/core/multi_wait.cpp
do
    if [ -f "$file" ]; then
        sed -i '1i #include <stdexcept>' "$file"
    fi
done
```

### Patch 4: cpuinfo Syscall Fix (if needed)
```bash
# Add syscall headers to cpuinfo/src/api.c
if [ -f "third_party/cpuinfo/src/api.c" ]; then
    sed -i '1i #ifndef _GNU_SOURCE\n#define _GNU_SOURCE\n#endif\n#include <unistd.h>\n#include <sys/syscall.h>' \
        third_party/cpuinfo/src/api.c
fi
```

### Patch 5: Flatbuffers Const Issue (if needed)
```bash
# Fix const-qualified assignment in flatbuffers
if [ -f "third_party/flatbuffers/include/flatbuffers/stl_emulation.h" ]; then
    sed -i 's/const size_type count_;/size_type count_;/' \
        third_party/flatbuffers/include/flatbuffers/stl_emulation.h
fi
```

## Step 3: Set Build Environment Variables

```bash
# Disable features not needed for CPU-only RISC-V build
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
export USE_SLEEF=0
export USE_DISTRIBUTED=0
export USE_FBGEMM=0
export USE_KINETO=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0

# Enable OpenBLAS for CPU math operations
export USE_OPENBLAS=1

# Build configuration
export BUILD_TEST=0           # Skip tests to save time and space
export MAX_JOBS=4             # Adjust based on available CPU cores
export CMAKE_BUILD_TYPE=Release

# Optional: Enable ccache for faster rebuilds
export USE_CCACHE=1
```

**Important**: Adjust `MAX_JOBS` based on your system:
- 4-8 cores: `MAX_JOBS=4`
- 8-16 cores: `MAX_JOBS=8`
- More RAM = can use more jobs

## Step 4: Build PyTorch

### Method 1: Using pip install (Recommended)

This method properly handles the Python bindings:

```bash
cd ~/pytorch

# Clean any previous builds
rm -rf build

# Build and install in editable/development mode
pip3 install --break-system-packages --no-build-isolation --no-deps -v -e .
```

The build will take **several hours** (4-8 hours depending on hardware).

### Method 2: Using setup.py (Alternative)

```bash
cd ~/pytorch

# Clean any previous builds
rm -rf build

# Build PyTorch
python3 setup.py build 2>&1 | tee ../pytorch-build.log

# After build completes, fix installation paths
cd build
find . -name "cmake_install.cmake" -exec sed -i \
    's|/usr/local/lib/python3.13/dist-packages|/home/debian/.local/lib/python3.13/site-packages|g' {} \;

# Install the built libraries
cmake --build . --target install --config Release

# Set up editable install
cd ~/pytorch
python3 setup.py develop --user
```

## Step 5: Verify Installation

```bash
# Test import
python3 << 'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch location: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test basic tensor operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y

print(f"Tensor addition test: {z.tolist()}")
assert z.tolist() == [5.0, 7.0, 9.0], "Tensor operation failed!"

print("\n✓ PyTorch is working correctly!")
EOF
```

Expected output:
```
PyTorch version: 2.4.0a0+git...
PyTorch location: /home/debian/pytorch/torch/__init__.py
CUDA available: False
Tensor addition test: [5.0, 7.0, 9.0]

✓ PyTorch is working correctly!
```

## Step 6: Set Up Environment (Optional)

Add to your `~/.bashrc`:

```bash
# PyTorch environment
export LD_LIBRARY_PATH=~/pytorch/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=~/pytorch:$PYTHONPATH
```

## Troubleshooting

### Issue 1: Linker Out of Memory
**Error**: `ld.lld: signal 7 (Bus error)`

**Solution**: Increase RAM allocation. PyTorch linking requires at least 16GB, recommended 32GB.

If using QEMU:
```bash
# Edit your QEMU launch script and increase -m parameter:
-m 32G
```

### Issue 2: Disk Space Full
**Error**: `No space left on device`

**Solution**: Ensure at least 100GB free space. The build directory can consume 50-70GB.

### Issue 3: CMake Install Permission Denied
**Error**: `file INSTALL cannot make directory "/usr/local/...": Permission denied`

**Solution**: Fix cmake_install.cmake files:
```bash
cd ~/pytorch/build
find . -name "cmake_install.cmake" -exec sed -i \
    's|/usr/local/lib/python3.13/dist-packages|$HOME/.local/lib/python3.13/site-packages|g' {} \;
```

### Issue 4: Module 'torch._C' Not Found
**Error**: `Failed to load PyTorch C extensions`

**Solution**: Use `python3 setup.py develop --user` instead of manual installation.

### Issue 5: cpuinfo Architecture Not Supported
**Warning**: `Error in cpuinfo: processor architecture is not supported`

**Impact**: This is a harmless warning. PyTorch works correctly and falls back to safe defaults.

### Issue 6: Old Build Directory Paths
**Error**: `CMakeCache.txt directory is different than the directory where CMakeCache.txt was created`

**Solution**: Clean rebuild:
```bash
cd ~/pytorch
rm -rf build
# Then rebuild using Method 1 or Method 2 above
```

## Build Time Expectations

On QEMU with 4 cores @ 32GB RAM:
- **Configuration**: 5-10 minutes
- **Compilation**: 4-6 hours
- **Linking**: 1-2 hours
- **Installation**: 5-10 minutes

**Total**: ~5-8 hours

On native RISC-V hardware (varies significantly):
- Modern server: 1-3 hours
- Development board: 6-12 hours

## Testing PyTorch Functionality

### Basic Tests

```python
import torch

# 1. Tensor creation
a = torch.randn(3, 3)
print(f"Random tensor:\n{a}")

# 2. Matrix multiplication
b = torch.randn(3, 3)
c = torch.matmul(a, b)
print(f"Matrix multiplication result:\n{c}")

# 3. Neural network layers
import torch.nn as nn

linear = nn.Linear(10, 5)
x = torch.randn(2, 10)
y = linear(x)
print(f"Linear layer output shape: {y.shape}")

# 4. Gradient computation
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(f"Gradient of x^2 at x=2: {x.grad}")
```

### Advanced Tests

```python
# 5. Simple neural network training
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Training loop
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("\n✓ Training test completed successfully!")
```

## Performance Notes

### RISC-V Specific Considerations

1. **No SIMD optimizations**: RISC-V vector extensions (RVV) are not yet fully supported in PyTorch
2. **CPU-only**: No GPU acceleration available on most RISC-V systems
3. **OpenBLAS**: Provides optimized BLAS operations for matrix math
4. **Memory bandwidth**: Often the limiting factor for large models

### Performance Tips

1. **Use smaller batch sizes**: RISC-V systems typically have less memory bandwidth
2. **Reduce model size**: Quantization and pruning are more important on RISC-V
3. **Enable OpenBLAS threading**: Set `export OPENBLAS_NUM_THREADS=4` (adjust to your cores)
4. **Use mixed precision (if supported)**: Can reduce memory usage

## Known Limitations

1. **No CUDA support**: CPU-only build
2. **No MKL-DNN**: Intel optimizations not available
3. **Limited distributed training**: Some distributed features disabled
4. **cpuinfo warnings**: Expected due to incomplete RISC-V support
5. **Performance**: 2-10x slower than x86_64 with AVX2/AVX-512

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| PyTorch | 2.4.0 | Recommended for Python 3.13 |
| Python | 3.13.x | Fully supported in PyTorch 2.4+ |
| Debian | sid/unstable | Required for Python 3.13 |
| GCC/Clang | 13+ | Required for C++17 features |
| CMake | 3.18+ | Required by PyTorch build |

**Python Version Compatibility**:
- PyTorch 2.0-2.1: Python 3.11 only
- PyTorch 2.2-2.3: Python 3.11-3.12
- PyTorch 2.4+: Python 3.11-3.13

## Complete Build Script

Here's a complete script that automates the entire build process:

```bash
#!/bin/bash
# Complete PyTorch RISC-V Build Script
set -e

echo "======================================"
echo "PyTorch RISC-V Complete Build Script"
echo "======================================"

# 1. Install dependencies
echo "Step 1: Installing dependencies..."
sudo apt update
sudo apt install -y \
    build-essential git cmake ninja-build \
    python3 python3-dev python3-pip \
    clang llvm lld libopenblas-dev ccache

pip3 install --break-system-packages \
    numpy typing-extensions pyyaml requests packaging sympy filelock jinja2 networkx fsspec

# 2. Clone repository
echo "Step 2: Cloning PyTorch..."
cd ~
if [ ! -d "pytorch" ]; then
    git clone --depth 1 --branch v2.4.0 --recursive https://github.com/pytorch/pytorch
fi
cd pytorch

# 3. Apply patches
echo "Step 3: Applying patches..."

# SLEEF FMA patch
sed -i 's/#error FP_FAST_FMA or FP_FAST_FMAF not defined/\/\/ #error FP_FAST_FMA or FP_FAST_FMAF not defined\n#define FP_FAST_FMA 1\n#define FP_FAST_FMAF 1/' \
    third_party/sleef/src/arch/helperpurec_scalar.h

# CMake warnings patch
cat > /tmp/cmake_patch.txt << 'EOF'
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=dangling-reference -Wno-error=tautological-compare -Wno-unknown-warning-option")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error -Wno-unknown-warning-option")
EOF
sed -i '/^project(Torch CXX C)/r /tmp/cmake_patch.txt' CMakeLists.txt

# 4. Set environment
echo "Step 4: Setting build environment..."
export USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0 USE_SLEEF=0
export USE_DISTRIBUTED=0 USE_FBGEMM=0 USE_KINETO=0
export USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0
export USE_OPENBLAS=1 BUILD_TEST=0 MAX_JOBS=4

# 5. Build and install
echo "Step 5: Building PyTorch (this will take several hours)..."
rm -rf build
pip3 install --break-system-packages --no-build-isolation --no-deps -v -e . 2>&1 | tee ../pytorch-build.log

# 6. Test
echo "Step 6: Testing installation..."
python3 << 'EOF'
import torch
print(f"✓ PyTorch {torch.__version__} installed successfully!")
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
assert (x + y).tolist() == [5.0, 7.0, 9.0]
print("✓ All tests passed!")
EOF

echo ""
echo "======================================"
echo "PyTorch build completed successfully!"
echo "======================================"
```

Save this as `build_pytorch_riscv.sh`, make it executable, and run:
```bash
chmod +x build_pytorch_riscv.sh
./build_pytorch_riscv.sh
```

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/stable/
- **PyTorch GitHub**: https://github.com/pytorch/pytorch
- **RISC-V**: https://riscv.org/
- **Issue Tracker**: https://github.com/pytorch/pytorch/issues

## Credits

Built and tested on:
- QEMU RISC-V 64-bit
- Debian sid (unstable)
- Python 3.13.9
- PyTorch 2.4.0

---

**Last Updated**: 2025-11-23
**Tested PyTorch Version**: 2.4.0a0+gitd990dad
