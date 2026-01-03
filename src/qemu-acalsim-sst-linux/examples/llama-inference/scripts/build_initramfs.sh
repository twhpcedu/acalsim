#!/bin/bash
#
# Build Complete Initramfs with LLAMA Inference App
#
# This script:
# 1. Installs LLAMA app files to rootfs
# 2. Rebuilds initramfs with all files
# 3. Creates a bootable initramfs.cpio.gz
#
# Copyright 2023-2026 Playlab/ACAL
# Licensed under the Apache License, Version 2.0
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================================"
echo "Build Complete Initramfs with LLAMA App"
echo "============================================================"
echo ""

# Configuration
ROOTFS=${ROOTFS:-/home/user/rootfs}
OUTPUT=${OUTPUT:-/home/user/initramfs-llama.cpio.gz}

echo "Configuration:"
echo "  ROOTFS: $ROOTFS"
echo "  OUTPUT: $OUTPUT"
echo ""

# Validate rootfs
if [ ! -d "$ROOTFS" ]; then
	echo -e "${RED}✗${NC} ROOTFS directory not found: $ROOTFS"
	exit 1
fi

if [ ! -f "$ROOTFS/init" ]; then
	echo -e "${RED}✗${NC} ROOTFS missing /init script"
	echo "This doesn't appear to be a valid rootfs"
	exit 1
fi

echo -e "${GREEN}✓${NC} ROOTFS found"

# Install LLAMA app
echo ""
echo "Installing LLAMA inference app..."
INSTALL_DIR="$ROOTFS/apps/llama-inference"
mkdir -p "$INSTALL_DIR"

# Copy files
cp llama_inference.py "$INSTALL_DIR/"
cp llama_sst_backend.py "$INSTALL_DIR/"
cp test_prompts.txt "$INSTALL_DIR/"
cp README.md "$INSTALL_DIR/"

# Make scripts executable
chmod +x "$INSTALL_DIR/llama_inference.py"

echo -e "${GREEN}✓${NC} LLAMA app installed to $INSTALL_DIR"

# Show what's in the rootfs
echo ""
echo "ROOTFS contents:"
echo "  Base directories:"
ls -ld "$ROOTFS"/{bin,sbin,lib,etc} 2>/dev/null || echo "    (minimal rootfs)"
echo "  Applications:"
ls -ld "$ROOTFS/apps"/* 2>/dev/null || echo "    (none)"

# Build initramfs
echo ""
echo "Building initramfs..."
cd "$ROOTFS"

# Count files
FILE_COUNT=$(find . -type f | wc -l)
echo "  Total files: $FILE_COUNT"

# Create initramfs
find . | cpio -o -H newc 2>/dev/null | gzip >"$OUTPUT"

if [ ! -f "$OUTPUT" ]; then
	echo -e "${RED}✗${NC} Failed to create initramfs"
	exit 1
fi

OUTPUT_SIZE=$(du -h "$OUTPUT" | cut -f1)
echo -e "${GREEN}✓${NC} Initramfs created: $OUTPUT ($OUTPUT_SIZE)"

# Verify contents
echo ""
echo "Verifying initramfs contents..."
VERIFY_DIR=$(mktemp -d)
cd "$VERIFY_DIR"
gunzip -c "$OUTPUT" | cpio -idm 2>/dev/null

if [ -f "init" ] && [ -f "apps/llama-inference/llama_inference.py" ]; then
	echo -e "${GREEN}✓${NC} Initramfs verified"
	echo "  Contains:"
	echo "    - /init script"
	echo "    - LLAMA inference app"
	ls apps/llama-inference/ | sed 's/^/      /'
else
	echo -e "${YELLOW}⚠${NC}  Initramfs may be incomplete"
fi

# Cleanup
cd /
rm -rf "$VERIFY_DIR"

# Usage instructions
echo ""
echo "============================================================"
echo "BUILD COMPLETE"
echo "============================================================"
echo ""
echo "Created: $OUTPUT"
echo "Size: $OUTPUT_SIZE"
echo ""
echo "To use this initramfs:"
echo "  1. Terminal 1:"
echo "     cd examples/llama-inference"
echo "     ./run_sst.sh"
echo ""
echo "  2. Terminal 2:"
echo "     export INITRAMFS=$OUTPUT"
echo "     cd examples/llama-inference"
echo "     ./run_qemu.sh"
echo ""
echo "  3. In Linux:"
echo "     cd /apps/llama-inference"
echo "     ./llama_inference.py \"Your prompt\""
echo ""
echo "Note: This initramfs contains the LLAMA app but NOT PyTorch."
echo "For full PyTorch support, see PYTORCH_LLAMA_SETUP.md"
echo "============================================================"
