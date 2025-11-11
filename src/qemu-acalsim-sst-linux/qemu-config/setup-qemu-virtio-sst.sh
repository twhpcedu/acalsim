#!/bin/bash
#
# QEMU VirtIO SST Device Setup Script
#
# Copyright 2023-2025 Playlab/ACAL
#
# This script integrates the VirtIO SST device into QEMU source tree.
# Run this script from within the acalsim-workspace Docker container.
#
# Usage:
#   ./setup-qemu-virtio-sst.sh <qemu-source-dir>
#
# Example:
#   ./setup-qemu-virtio-sst.sh /home/user/qemu-build/qemu
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -ne 1 ]; then
    echo -e "${RED}Error: QEMU source directory not specified${NC}"
    echo "Usage: $0 <qemu-source-dir>"
    echo "Example: $0 /home/user/qemu-build/qemu"
    exit 1
fi

QEMU_SRC="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIRTIO_DEVICE_DIR="$(dirname "$SCRIPT_DIR")/virtio-device"

# Verify directories exist
if [ ! -d "$QEMU_SRC" ]; then
    echo -e "${RED}Error: QEMU source directory not found: $QEMU_SRC${NC}"
    exit 1
fi

if [ ! -d "$VIRTIO_DEVICE_DIR" ]; then
    echo -e "${RED}Error: VirtIO device directory not found: $VIRTIO_DEVICE_DIR${NC}"
    exit 1
fi

echo "============================================"
echo "  QEMU VirtIO SST Device Setup"
echo "============================================"
echo "QEMU Source:   $QEMU_SRC"
echo "VirtIO Device: $VIRTIO_DEVICE_DIR"
echo "============================================"
echo ""

# Step 1: Copy VirtIO SST device files
echo -e "${GREEN}[1/4] Copying VirtIO SST device files...${NC}"

cp -v "$VIRTIO_DEVICE_DIR/sst-protocol.h" "$QEMU_SRC/include/hw/virtio/" || exit 1
cp -v "$VIRTIO_DEVICE_DIR/virtio-sst.h" "$QEMU_SRC/include/hw/virtio/" || exit 1
cp -v "$VIRTIO_DEVICE_DIR/virtio-sst.c" "$QEMU_SRC/hw/virtio/" || exit 1

echo -e "${GREEN}✓ Device files copied${NC}"
echo ""

# Step 2: Update meson.build
echo -e "${GREEN}[2/4] Updating hw/virtio/meson.build...${NC}"

MESON_BUILD="$QEMU_SRC/hw/virtio/meson.build"

# Check if already added
if grep -q "CONFIG_VIRTIO_SST" "$MESON_BUILD"; then
    echo -e "${YELLOW}⚠ VirtIO SST already in meson.build, removing old entry...${NC}"
    sed -i '/CONFIG_VIRTIO_SST/d' "$MESON_BUILD"
fi

# Find the line with specific_ss.add_all and insert before it
LINE_NUM=$(grep -n "^specific_ss.add_all(when: 'CONFIG_VIRTIO', if_true: virtio_ss)" "$MESON_BUILD" | cut -d: -f1)

if [ -z "$LINE_NUM" ]; then
    echo -e "${RED}Error: Could not find specific_ss.add_all line in meson.build${NC}"
    exit 1
fi

# Insert the virtio-sst line before specific_ss.add_all
sed -i "${LINE_NUM}i virtio_ss.add(when: 'CONFIG_VIRTIO_SST', if_true: files('virtio-sst.c'))" "$MESON_BUILD"

echo -e "${GREEN}✓ meson.build updated (inserted at line $LINE_NUM)${NC}"
echo ""

# Step 3: Update Kconfig
echo -e "${GREEN}[3/4] Updating hw/virtio/Kconfig...${NC}"

KCONFIG="$QEMU_SRC/hw/virtio/Kconfig"

# Check if already added
if grep -q "CONFIG_VIRTIO_SST" "$KCONFIG"; then
    echo -e "${YELLOW}⚠ VirtIO SST already in Kconfig, skipping...${NC}"
else
    # Add Kconfig entry at the end
    cat >> "$KCONFIG" <<'EOF'

config VIRTIO_SST
    bool
    default y
    depends on VIRTIO
EOF
    echo -e "${GREEN}✓ Kconfig updated${NC}"
fi
echo ""

# Step 4: Verify files
echo -e "${GREEN}[4/4] Verifying installation...${NC}"

ERRORS=0

# Check header files
if [ ! -f "$QEMU_SRC/include/hw/virtio/sst-protocol.h" ]; then
    echo -e "${RED}✗ Missing: include/hw/virtio/sst-protocol.h${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ include/hw/virtio/sst-protocol.h${NC}"
fi

if [ ! -f "$QEMU_SRC/include/hw/virtio/virtio-sst.h" ]; then
    echo -e "${RED}✗ Missing: include/hw/virtio/virtio-sst.h${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ include/hw/virtio/virtio-sst.h${NC}"
fi

# Check source file
if [ ! -f "$QEMU_SRC/hw/virtio/virtio-sst.c" ]; then
    echo -e "${RED}✗ Missing: hw/virtio/virtio-sst.c${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ hw/virtio/virtio-sst.c${NC}"
fi

# Check meson.build
if ! grep -q "CONFIG_VIRTIO_SST.*virtio-sst.c" "$MESON_BUILD"; then
    echo -e "${RED}✗ VirtIO SST not properly added to meson.build${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ hw/virtio/meson.build configured${NC}"
fi

# Check Kconfig
if ! grep -q "CONFIG_VIRTIO_SST" "$KCONFIG"; then
    echo -e "${RED}✗ VirtIO SST not in Kconfig${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ hw/virtio/Kconfig configured${NC}"
fi

echo ""

if [ $ERRORS -eq 0 ]; then
    echo "============================================"
    echo -e "${GREEN}✓ VirtIO SST device successfully integrated!${NC}"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "  1. Configure QEMU:"
    echo "     cd $QEMU_SRC"
    echo "     mkdir -p build && cd build"
    echo "     ../configure --target-list=riscv64-softmmu --enable-virtfs"
    echo ""
    echo "  2. Build QEMU:"
    echo "     make -j\$(nproc)"
    echo ""
    echo "  3. Verify build:"
    echo "     ./qemu-system-riscv64 --version"
    echo ""
    exit 0
else
    echo "============================================"
    echo -e "${RED}✗ Setup completed with $ERRORS error(s)${NC}"
    echo "============================================"
    exit 1
fi
