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

# PyTorch Build Fix Instructions

## Current Status
QEMU Debian RISC-V is running and PyTorch v2.3.0 build with Clang failed due to missing `syscall` declaration in cpuinfo library.

## Fix and Restart Build

**Run these commands in the QEMU Debian console** (copy-paste the entire block):

```bash
# Stop any running build
pkill -f 'python3 setup.py' || echo "No build running"

# Navigate to PyTorch
cd ~/pytorch

# Apply cpuinfo fix (add syscall header)
sed -i '1i #include <sys/syscall.h>' third_party/cpuinfo/src/api.c

# Verify fix was applied
echo "=== Verifying fix ==="
head -5 third_party/cpuinfo/src/api.c

# Clean build
rm -rf build/
python3 setup.py clean 2>&1 | tail -10

# Set Clang environment variables
export CC=clang
export CXX=clang++
export CFLAGS="-w"
export CXXFLAGS="-w"
export LDFLAGS="-fuse-ld=lld"
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0
export USE_FBGEMM=0
export USE_KINETO=0
export USE_NUMPY=1
export MAX_JOBS=2

# Start build in background
echo "=== Starting PyTorch build with Clang ==="
echo "Build started at: $(date)"
nohup python3 setup.py install > /tmp/pytorch_clang_build.log 2>&1 &
echo $! > /tmp/pytorch_build.pid
echo "Build PID: $(cat /tmp/pytorch_build.pid)"

# Wait a few seconds
sleep 5

# Show initial build output
echo ""
echo "=== Latest build output ==="
tail -30 /tmp/pytorch_clang_build.log

echo ""
echo "=== Monitor build with ==="
echo "tail -f /tmp/pytorch_clang_build.log"
```

## Monitor Build Progress

```bash
# Watch build log in real-time
tail -f /tmp/pytorch_clang_build.log

# Check if build is still running
ps -p $(cat /tmp/pytorch_build.pid 2>/dev/null) || echo "Build finished or failed"

# Check last 50 lines of log
tail -50 /tmp/pytorch_clang_build.log
```

## Expected Output

After applying the fix, you should see:
- `#include <sys/syscall.h>` as the first line of `cpuinfo/src/api.c`
- Build starting successfully with Clang compiler
- No more "implicit declaration of function 'syscall'" errors

## Build Time

Expected build time: **12-24 hours** on RISC-V emulation

## Next Steps After Build Completes

Once the build finishes successfully:

```bash
# Verify PyTorch installation
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import torch; print('Build config:', torch.__config__.show())"

# Test basic tensor operations
python3 -c "import torch; x = torch.randn(3, 3); print('Tensor:', x)"
```

## Troubleshooting

If build fails again:
```bash
# Check last errors in log
tail -100 /tmp/pytorch_clang_build.log | grep -i error

# Check if out of memory
dmesg | tail -20

# Check disk space
df -h
```

## Alternative: Quick One-Liner Fix

If you just want to apply the fix and restart quickly:

```bash
pkill -f setup.py; cd ~/pytorch; sed -i '1i #include <sys/syscall.h>' third_party/cpuinfo/src/api.c; rm -rf build/; export CC=clang CXX=clang++ CFLAGS="-w" CXXFLAGS="-w" LDFLAGS="-fuse-ld=lld" USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0 USE_NNPACK=0 USE_QNNPACK=0 USE_DISTRIBUTED=0 BUILD_TEST=0 USE_FBGEMM=0 USE_KINETO=0 USE_NUMPY=1 MAX_JOBS=2; nohup python3 setup.py install > /tmp/pytorch_clang_build.log 2>&1 & echo $! > /tmp/pytorch_build.pid; sleep 3; tail -30 /tmp/pytorch_clang_build.log
```
