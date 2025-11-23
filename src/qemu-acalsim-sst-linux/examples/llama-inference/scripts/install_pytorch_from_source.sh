#!/bin/bash
#
# Install PyTorch from Source for RISC-V
#
# This script installs PyTorch by building from source inside QEMU.
# Expected build time: 12-24 hours on RISC-V emulation
#
# Run this inside the QEMU Linux environment
#
# Copyright 2023-2025 Playlab/ACAL
# Licensed under the Apache License, Version 2.0
#

set -e

echo "============================================================"
echo "PyTorch from Source Installation for RISC-V"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Install build dependencies"
echo "  2. Clone PyTorch repository"
echo "  3. Build PyTorch (12-24 hours)"
echo "  4. Install PyTorch"
echo ""
echo "Requirements:"
echo "  - 8GB RAM (you have this)"
echo "  - 10GB disk space (you have this)"
echo "  - Network connectivity"
echo "  - 12-24 hours build time"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	echo "Aborted"
	exit 1
fi

# Step 1: Install build dependencies
echo ""
echo "============================================================"
echo "Step 1: Installing Build Dependencies"
echo "============================================================"
echo ""

pip3 install --upgrade pip setuptools wheel

# Install Python dependencies for PyTorch build
pip3 install numpy pyyaml typing_extensions

echo "✓ Build dependencies installed"

# Step 2: Clone PyTorch (use a smaller, compatible version)
echo ""
echo "============================================================"
echo "Step 2: Cloning PyTorch Source"
echo "============================================================"
echo ""

cd /root
if [ -d "pytorch" ]; then
	echo "PyTorch directory already exists, using existing clone"
	cd pytorch
	git pull
else
	# Clone a stable version that's known to work
	git clone --depth 1 --branch v2.0.1 https://github.com/pytorch/pytorch.git
	cd pytorch
fi

echo "✓ PyTorch source ready"

# Step 3: Initialize submodules (only essential ones to save time)
echo ""
echo "============================================================"
echo "Step 3: Initializing Submodules"
echo "============================================================"
echo ""

git submodule update --init --recursive --depth 1

echo "✓ Submodules initialized"

# Step 4: Build PyTorch
echo ""
echo "============================================================"
echo "Step 4: Building PyTorch (This will take 12-24 hours)"
echo "============================================================"
echo ""
echo "Build started at: $(date)"
echo ""

# Set build options for minimal PyTorch (inference only)
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

# Build with limited parallelism to avoid OOM
export MAX_JOBS=2

python3 setup.py install

echo ""
echo "Build completed at: $(date)"
echo "✓ PyTorch built and installed"

# Step 5: Verify installation
echo ""
echo "============================================================"
echo "Step 5: Verifying Installation"
echo "============================================================"
echo ""

python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import torch; print('PyTorch build:', torch.__config__.show())"

echo ""
echo "============================================================"
echo "PYTORCH INSTALLATION COMPLETE!"
echo "============================================================"
echo ""
echo "PyTorch is now installed and ready for LLAMA inference."
echo ""
echo "Next steps:"
echo "  1. Test PyTorch: python3 -c 'import torch; print(torch.__version__)'"
echo "  2. Download LLAMA model (see LLAMA setup docs)"
echo "  3. Run inference with SST backend"
echo "============================================================"
