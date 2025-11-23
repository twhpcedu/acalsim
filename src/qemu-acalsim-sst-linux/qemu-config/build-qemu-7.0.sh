#!/bin/bash
#
# QEMU 7.0.0 Build Script with virtio-sst Support
#
# This script builds QEMU 7.0.0 with virtio-sst device integration.
# QEMU 7.0.0 is the last version compatible with the current virtio-sst code.
#
# Usage:
#   ./build-qemu-7.0.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIRTIO_DEVICE_DIR="$(dirname "$SCRIPT_DIR")/virtio-device"
BUILD_DIR="/home/user/qemu-build"
QEMU_VERSION="v7.0.0"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  QEMU 7.0.0 Build with virtio-sst${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Step 1: Install dependencies
echo -e "${GREEN}[1/6] Installing dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y \
  ninja-build meson build-essential \
  libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev \
  pkg-config python3 git \
  libslirp-dev libslirp0 \
  libcap-ng-dev libattr1-dev

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Clone QEMU
echo -e "${GREEN}[2/6] Cloning QEMU ${QEMU_VERSION}...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ -d "qemu" ]; then
  echo -e "${YELLOW}⚠ QEMU directory exists, cleaning...${NC}"
  rm -rf qemu
fi

git clone https://github.com/qemu/qemu.git
cd qemu
git checkout "$QEMU_VERSION"

echo -e "${GREEN}✓ QEMU cloned and checked out${NC}"
echo ""

# Step 3: Integrate virtio-sst device
echo -e "${GREEN}[3/6] Integrating virtio-sst device...${NC}"

# Copy device files
cp "$VIRTIO_DEVICE_DIR/sst-protocol.h" include/hw/virtio/
cp "$VIRTIO_DEVICE_DIR/virtio-sst.h" include/hw/virtio/
cp "$VIRTIO_DEVICE_DIR/virtio-sst.c" hw/virtio/

# Fix include order (qemu/osdep.h must be first)
sed -i '/#include "qemu\/osdep.h"/d' include/hw/virtio/virtio-sst.h
sed -i '1i #include "qemu/osdep.h"' hw/virtio/virtio-sst.c

echo -e "${GREEN}✓ Device files copied${NC}"

# Step 4: Update build configuration
echo -e "${GREEN}[4/6] Updating build configuration...${NC}"

# Add to meson.build (QEMU 7.0 uses virtio_ss source set)
cd hw/virtio
if grep -q "CONFIG_VIRTIO_SST" meson.build; then
  echo -e "${YELLOW}⚠ virtio-sst already in meson.build${NC}"
else
  # Add to virtio_ss after VIRTIO_MEM
  sed -i "/^virtio_ss.add(when: 'CONFIG_VIRTIO_MEM'/a virtio_ss.add(when: 'CONFIG_VIRTIO_SST', if_true: files('virtio-sst.c'))" meson.build
  echo -e "${GREEN}✓ Updated meson.build${NC}"
fi

# Add to Kconfig
if grep -q "CONFIG_VIRTIO_SST" Kconfig; then
  echo -e "${YELLOW}⚠ virtio-sst already in Kconfig${NC}"
else
  cat >> Kconfig << 'EOF'

config VIRTIO_SST
    bool
    default y
    depends on VIRTIO
EOF
  echo -e "${GREEN}✓ Updated Kconfig${NC}"
fi

cd "$BUILD_DIR/qemu"
echo ""

# Step 5: Configure QEMU
echo -e "${GREEN}[5/6] Configuring QEMU...${NC}"
mkdir -p build
cd build
../configure --target-list=riscv64-softmmu --enable-virtfs

echo -e "${GREEN}✓ Configuration complete${NC}"
echo ""

# Step 6: Build QEMU
echo -e "${GREEN}[6/6] Building QEMU (this may take several minutes)...${NC}"
make -j$(nproc)

echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Verification
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Verification${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

VERSION=$(./qemu-system-riscv64 --version | head -1)
echo -e "${GREEN}Version: ${VERSION}${NC}"

if ./qemu-system-riscv64 -device help 2>&1 | grep -q "virtio-sst"; then
  echo -e "${GREEN}✓ virtio-sst-device: Available${NC}"
else
  echo -e "${RED}✗ virtio-sst-device: NOT FOUND${NC}"
fi

if ./qemu-system-riscv64 -netdev help 2>&1 | grep -q "user"; then
  echo -e "${GREEN}✓ user network backend: Available${NC}"
else
  echo -e "${RED}✗ user network backend: NOT FOUND${NC}"
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✓ QEMU build successful!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "QEMU binary: $BUILD_DIR/qemu/build/qemu-system-riscv64"
echo ""
echo "To add to PATH, run:"
echo "  export PATH=$BUILD_DIR/qemu/build:\$PATH"
echo "  echo 'export PATH=$BUILD_DIR/qemu/build:\$PATH' >> ~/.bashrc"
echo ""
echo "To boot with virtio-sst:"
echo "  cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference"
echo "  ./run_qemu_persistent.sh"
echo ""
