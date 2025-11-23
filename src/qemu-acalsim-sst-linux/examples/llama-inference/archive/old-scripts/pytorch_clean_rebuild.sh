#!/bin/bash
# Complete PyTorch Clean and Rebuild Script for RISC-V
# Run this inside Debian QEMU to fix SLEEF FMA errors

cat << 'INSTRUCTIONS'
=====================================
PyTorch Complete Clean & Rebuild
=====================================

The SLEEF error persists because CMake cached the old configuration.
You need to completely remove the build directory.

Run these commands inside Debian QEMU:

# Stop the current build (Ctrl+C if still running)

cd /home/debian/pytorch

# COMPLETE CLEAN - Remove all build artifacts
rm -rf build/
rm -rf torch.egg-info/
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# Create build configuration script
cat > build_riscv_clean.sh << 'EOF'
#!/bin/bash
set -e

echo "====================================="
echo "PyTorch RISC-V Clean Build"
echo "====================================="

# CRITICAL: Disable SLEEF to avoid FMA errors
export USE_SLEEF=0

# Disable all GPU/SIMD/vectorization features
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

# Disable testing to speed up build
export BUILD_TEST=0
export BUILD_CAFFE2=0

# Use OpenBLAS for linear algebra
export BLAS=OpenBLAS
export USE_OPENBLAS=1

# Build settings
export MAX_JOBS=4
export CMAKE_BUILD_TYPE=Release
export CMAKE_GENERATOR=Ninja

# Additional flags to ensure clean build
export CMAKE_ARGS="-DUSE_SLEEF=OFF"

echo ""
echo "Configuration:"
echo "  USE_SLEEF: $USE_SLEEF (DISABLED - fixes FMA error)"
echo "  USE_OPENBLAS: $USE_OPENBLAS"
echo "  MAX_JOBS: $MAX_JOBS"
echo "  CMAKE_ARGS: $CMAKE_ARGS"
echo "====================================="
echo ""

# Verify build directory is clean
if [ -d "build" ]; then
    echo "ERROR: build/ directory still exists!"
    echo "Please run: rm -rf build/"
    exit 1
fi

echo "Starting clean build at $(date)"
echo "This will take 4-8 hours..."
echo ""

# Build and install
python3 setup.py install 2>&1 | tee ../pytorch-build-clean.log

echo ""
echo "Build finished at $(date)"
echo ""

# Test installation
echo "Testing PyTorch installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed successfully')" || {
    echo "✗ PyTorch import failed"
    exit 1
}

echo ""
echo "✓ Build completed successfully!"
EOF

chmod +x build_riscv_clean.sh

# Run the clean build
./build_riscv_clean.sh

INSTRUCTIONS
