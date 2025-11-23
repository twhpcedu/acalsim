#!/bin/bash
#
# Fix PyTorch cpuinfo issue and rebuild with Clang
#
# Run this script inside the QEMU Debian environment
#
# Copyright 2023-2025 Playlab/ACAL
# Licensed under the Apache License, Version 2.0
#

set -e

echo "============================================================"
echo "Fixing PyTorch cpuinfo and Restarting Build"
echo "============================================================"
echo ""

# Step 1: Kill any existing build
echo "Step 1: Stopping any running PyTorch build..."
pkill -f 'python3 setup.py' || echo "No build process found (OK)"
echo "✓ Build process stopped"
echo ""

# Step 2: Navigate to PyTorch directory
cd ~/pytorch
echo "Working directory: $(pwd)"
echo ""

# Step 3: Apply cpuinfo fix
echo "Step 2: Applying cpuinfo syscall fix..."
echo ""
echo "File before fix:"
head -5 third_party/cpuinfo/src/api.c

# Add the syscall header at the beginning
sed -i '1i #include <sys/syscall.h>' third_party/cpuinfo/src/api.c

echo ""
echo "File after fix:"
head -5 third_party/cpuinfo/src/api.c
echo "✓ cpuinfo syscall header added"
echo ""

# Step 4: Clean build directory
echo "Step 3: Cleaning build directory..."
rm -rf build/
python3 setup.py clean || echo "Clean completed (warnings OK)"
echo "✓ Build directory cleaned"
echo ""

# Step 5: Set environment variables
echo "Step 4: Setting Clang environment..."
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

echo "Environment variables set:"
echo "  CC=$CC"
echo "  CXX=$CXX"
echo "  CFLAGS=$CFLAGS"
echo "  CXXFLAGS=$CXXFLAGS"
echo "  MAX_JOBS=$MAX_JOBS"
echo "✓ Environment configured"
echo ""

# Step 6: Start build in background
echo "Step 5: Starting PyTorch build with Clang..."
echo ""
echo "Build log: /tmp/pytorch_clang_build.log"
echo "PID file: /tmp/pytorch_build.pid"
echo ""
echo "Build started at: $(date)"
echo ""

nohup python3 setup.py install > /tmp/pytorch_clang_build.log 2>&1 &
echo $! > /tmp/pytorch_build.pid

BUILD_PID=$!
echo "Build process PID: $BUILD_PID"
echo ""

# Wait a few seconds to check if build starts successfully
sleep 5

if ps -p $BUILD_PID > /dev/null; then
    echo "✓ Build is running successfully!"
    echo ""
    echo "Monitor build progress with:"
    echo "  tail -f /tmp/pytorch_clang_build.log"
    echo ""
    echo "Check if still running:"
    echo "  ps -p $BUILD_PID"
    echo ""
    echo "Expected build time: 12-24 hours"
    echo ""
    echo "Latest build output:"
    tail -20 /tmp/pytorch_clang_build.log
else
    echo "✗ Build process stopped unexpectedly!"
    echo ""
    echo "Check log for errors:"
    tail -50 /tmp/pytorch_clang_build.log
    exit 1
fi

echo ""
echo "============================================================"
echo "BUILD RESTART COMPLETE!"
echo "============================================================"
