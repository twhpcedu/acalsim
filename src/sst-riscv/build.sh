#!/bin/bash
set -e # Exit on error

echo "=== Building ACALSim SST Integration ==="
cd /home/user/projects/acalsim/src/sst-riscv

# Clean
echo "Cleaning..."
make clean

# Build
echo "Building..."
make

# Check if library was built
if [ ! -f "libacalsim_sst.so" ]; then
	echo "ERROR: libacalsim_sst.so was not built!"
	exit 1
fi
echo "Library built successfully: libacalsim_sst.so"

# Get SST element directory
SST_ELEM_DIR=$(sst-config --prefix)/lib/sst-elements-library
echo "SST element directory: $SST_ELEM_DIR"

# Create directory if needed
mkdir -p $SST_ELEM_DIR

# Install
echo "Installing to $SST_ELEM_DIR..."
cp -v libacalsim_sst.so $SST_ELEM_DIR/

# Verify
echo ""
echo "=== Verifying Installation ==="
ls -la $SST_ELEM_DIR/libacalsim_sst.so

echo ""
echo "=== Checking SST Registration ==="
sst-info acalsim

echo ""
echo "=== Installation Complete ==="
