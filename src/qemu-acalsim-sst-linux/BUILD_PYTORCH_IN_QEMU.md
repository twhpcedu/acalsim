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

# Building PyTorch in QEMU RISC-V (Buildroot)

## Prerequisites

1. Buildroot rebuilt with PyTorch dependencies
2. Booted into QEMU with buildroot initramfs
3. At least 15GB free space in rootfs
4. 8GB RAM (already configured in QEMU)

## Step-by-Step Build Process

### 1. Verify Environment

```bash
# Check Python
python3 --version
# Expected: Python 3.11.8

# Check NumPy
python3 -c "import numpy; print(numpy.__version__)"

# Check OpenBLAS
ls /usr/lib/libopenblas* || ls /usr/lib/*/libopenblas*

# Check build tools
cmake --version
ninja --version
git --version

# Check disk space
df -h
# Need at least 15GB free
```

### 2. Install Python Dependencies

```bash
# Install via pip
pip3 install \
    typing-extensions \
    pyyaml \
    requests \
    packaging \
    sympy \
    filelock \
    networkx \
    jinja2 \
    fsspec

# Verify
pip3 list
```

### 3. Clone PyTorch

```bash
# Create working directory
mkdir -p /root/pytorch-build
cd /root/pytorch-build

# Clone PyTorch (shallow clone to save space)
git clone --depth 1 --branch v2.1.0 --recursive https://github.com/pytorch/pytorch
cd pytorch

# Check size
du -sh .
# Should be ~1-2GB
```

### 4. Configure Build Environment

Create a build configuration script:

```bash
cat > /root/pytorch-build/pytorch/build_config.sh << 'EOF'
#!/bin/bash
# PyTorch Build Configuration for RISC-V

# Disable CUDA/GPU features
export USE_CUDA=0
export USE_CUDNN=0
export USE_ROCM=0
export USE_XNNPACK=0

# Disable optional accelerators
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_FBGEMM=0
export USE_KINETO=0

# Disable distributed training
export USE_DISTRIBUTED=0
export USE_MPI=0
export USE_GLOO=0
export USE_TENSORPIPE=0

# Disable testing
export BUILD_TEST=0
export BUILD_CAFFE2=0

# Use OpenBLAS (critical for RISC-V)
export BLAS=OpenBLAS
export USE_OPENBLAS=1

# Build settings
export MAX_JOBS=4
export CMAKE_BUILD_TYPE=Release

# Use Ninja for faster builds
export CMAKE_GENERATOR=Ninja

# Enable verbose output
export VERBOSE=1

echo "PyTorch build environment configured for RISC-V"
echo "Using OpenBLAS: $USE_OPENBLAS"
echo "Max parallel jobs: $MAX_JOBS"
EOF

chmod +x /root/pytorch-build/pytorch/build_config.sh
```

### 5. Build PyTorch

```bash
cd /root/pytorch-build/pytorch

# Load build configuration
source build_config.sh

# Clean any previous build attempts
python3 setup.py clean

# Build and install (THIS WILL TAKE 4-8 HOURS)
echo "Starting PyTorch build at $(date)"
python3 setup.py install 2>&1 | tee ../pytorch-build.log
echo "Build finished at $(date)"
```

**Note**: This will take 4-8 hours on RISC-V. You can monitor progress in the log:
```bash
# In another terminal (if you have SSH access):
tail -f /root/pytorch-build/pytorch-build.log
```

### 6. Verify Installation

```bash
# Test PyTorch import
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test basic operations
python3 << 'PYTEST'
import torch
import numpy as np

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor: {x}")

# Basic operations
y = x * 2
print(f"Result: {y}")

# NumPy interop
arr = np.array([4.0, 5.0, 6.0])
z = torch.from_numpy(arr)
print(f"From NumPy: {z}")

print("\n✓ PyTorch is working!")
PYTEST
```

### 7. Test with Simple Neural Network

```bash
python3 << 'PYTEST'
import torch
import torch.nn as nn

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = SimpleNet()
print(model)

# Test forward pass
x = torch.randn(5, 10)
output = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

print("\n✓ Neural network test passed!")
PYTEST
```

## Build Time Optimization

### Use ccache (Already Included)

```bash
# ccache should be available from buildroot
ccache -s  # Show cache statistics

# Set ccache size
ccache -M 5G
```

### Monitor Build Progress

```bash
# Watch build progress
watch -n 5 'ps aux | grep -E "gcc|g\+\+|clang" | wc -l'

# Check disk usage during build
watch -n 60 'df -h'
```

## Troubleshooting

### Build Fails with "Out of Memory"

```bash
# Reduce parallel jobs
export MAX_JOBS=2

# Or build with no parallelism
export MAX_JOBS=1
```

### Build Fails with Missing Dependencies

```bash
# Install missing Python packages
pip3 install <package-name>
```

### Build Takes Too Long

- **Expected**: 4-8 hours on RISC-V is normal
- **Optimization**: Use ccache (already configured)
- **Alternative**: Build with fewer features (already minimal)

### Disk Space Issues

```bash
# Clean build artifacts
python3 setup.py clean

# Remove git history if needed
cd /root/pytorch-build/pytorch
rm -rf .git
```

## Post-Installation

### Test Installation Location

```bash
python3 -c "import torch; print(torch.__file__)"
```

### Check Installed Size

```bash
du -sh /usr/lib/python3.11/site-packages/torch
# Expected: ~500MB-1GB
```

### Save Built PyTorch (Optional)

If you want to save the built PyTorch for reuse:

```bash
# Create a wheel
cd /root/pytorch-build/pytorch
python3 setup.py bdist_wheel

# Wheel will be in dist/
ls -lh dist/*.whl
```

## Performance Expectations

| Operation | RISC-V Performance |
|-----------|-------------------|
| Tensor creation | Similar to x86 |
| Matrix multiplication | Slower (no SIMD) |
| Neural network training | 5-10x slower than x86 |
| Inference | Acceptable for small models |

## Next Steps

After PyTorch is built:
1. Test with your specific models
2. Profile performance with RISC-V
3. Consider using virtio-sst for simulation integration

---

**Estimated Total Time**: 5-9 hours (30-60 min buildroot + 4-8 hours PyTorch build)
