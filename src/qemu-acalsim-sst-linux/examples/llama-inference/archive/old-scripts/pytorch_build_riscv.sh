#!/bin/bash
# PyTorch RISC-V Build Script (run inside Debian QEMU)
# This script configures PyTorch to build without SLEEF to avoid FMA errors

set -e

cat <<'INSTRUCTIONS'
=====================================
PyTorch RISC-V Build Script
=====================================

This script fixes the SLEEF FMA error by disabling SLEEF.

Copy and run this inside Debian QEMU:

cd /home/debian/pytorch

# Create build script
cat > build_riscv.sh << 'EOF'
#!/bin/bash
# PyTorch RISC-V Build Configuration

# Disable SLEEF (fixes FMA error)
export USE_SLEEF=0

# Disable all GPU/SIMD features
export USE_CUDA=0
export USE_CUDNN=0
export USE_ROCM=0
export USE_MKLDNN=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_FBGEMM=0
export USE_KINETO=0

# Disable distributed training
export USE_DISTRIBUTED=0
export USE_MPI=0
export USE_GLOO=0
export USE_TENSORPIPE=0
export USE_NCCL=0

# Disable testing
export BUILD_TEST=0
export BUILD_CAFFE2=0

# Use OpenBLAS for linear algebra
export BLAS=OpenBLAS
export USE_OPENBLAS=1

# Build settings
export MAX_JOBS=4
export CMAKE_BUILD_TYPE=Release
export CMAKE_GENERATOR=Ninja

echo "====================================="
echo "PyTorch RISC-V Build Configuration"
echo "====================================="
echo "USE_SLEEF: $USE_SLEEF (disabled - fixes FMA error)"
echo "USE_OPENBLAS: $USE_OPENBLAS"
echo "MAX_JOBS: $MAX_JOBS"
echo "====================================="

# Clean previous build
echo "Cleaning previous build..."
python3 setup.py clean

# Build and install
echo "Starting PyTorch build at $(date)"
python3 setup.py install 2>&1 | tee ../pytorch-build.log
echo "Build finished at $(date)"

# Test installation
echo ""
echo "Testing PyTorch installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed successfully')"
EOF

chmod +x build_riscv.sh

# Run the build
./build_riscv.sh

INSTRUCTIONS
