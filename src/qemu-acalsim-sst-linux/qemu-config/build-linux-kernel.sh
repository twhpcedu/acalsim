#!/bin/bash
#
# Linux Kernel Build Script for RISC-V
#
# This script builds a RISC-V Linux kernel compatible with QEMU and virtio-sst.
#
# Usage:
#   ./build-linux-kernel.sh [kernel-version]
#
# Example:
#   ./build-linux-kernel.sh v6.1
#   ./build-linux-kernel.sh  # Uses default v6.1
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

KERNEL_VERSION="${1:-v6.1}"
BUILD_DIR="/home/user"
LINUX_DIR="$BUILD_DIR/linux"
CROSS_COMPILE="riscv64-linux-gnu-"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Linux Kernel Build for RISC-V${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Kernel Version: $KERNEL_VERSION"
echo "Build Directory: $BUILD_DIR"
echo "Cross Compiler: ${CROSS_COMPILE}gcc"
echo ""

# Step 1: Install dependencies
echo -e "${GREEN}[1/5] Installing build dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y \
	build-essential \
	libncurses-dev \
	bison \
	flex \
	libssl-dev \
	libelf-dev \
	bc \
	gcc-riscv64-linux-gnu \
	git

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Clone Linux kernel
echo -e "${GREEN}[2/5] Cloning Linux kernel...${NC}"
cd "$BUILD_DIR"

if [ -d "$LINUX_DIR" ]; then
	echo -e "${YELLOW}⚠ Linux directory exists${NC}"
	read -p "Remove and re-clone? (y/N): " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		rm -rf "$LINUX_DIR"
	else
		echo "Using existing kernel source"
		cd "$LINUX_DIR"
		git fetch --all --tags
	fi
fi

if [ ! -d "$LINUX_DIR" ]; then
	git clone --depth 1 --branch "$KERNEL_VERSION" https://github.com/torvalds/linux.git
	cd "$LINUX_DIR"
fi

echo -e "${GREEN}✓ Kernel source ready${NC}"
echo ""

# Step 3: Configure kernel
echo -e "${GREEN}[3/5] Configuring kernel for RISC-V...${NC}"
cd "$LINUX_DIR"

# Start with defconfig
make ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" defconfig

# Enable virtio drivers
echo -e "${BLUE}Enabling VirtIO drivers...${NC}"
scripts/config --enable CONFIG_VIRTIO
scripts/config --enable CONFIG_VIRTIO_BLK
scripts/config --enable CONFIG_VIRTIO_NET
scripts/config --enable CONFIG_VIRTIO_CONSOLE
scripts/config --enable CONFIG_VIRTIO_MMIO
scripts/config --enable CONFIG_HW_RANDOM_VIRTIO
scripts/config --enable CONFIG_9P_FS
scripts/config --enable CONFIG_9P_FS_POSIX_ACL
scripts/config --enable CONFIG_NET_9P
scripts/config --enable CONFIG_NET_9P_VIRTIO

# Enable networking
scripts/config --enable CONFIG_INET
scripts/config --enable CONFIG_PACKET

# Enable initramfs
scripts/config --enable CONFIG_BLK_DEV_INITRD

# Regenerate config with dependencies
make ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" olddefconfig

echo -e "${GREEN}✓ Kernel configured${NC}"
echo ""

# Step 4: Build kernel
echo -e "${GREEN}[4/5] Building kernel (this may take 15-30 minutes)...${NC}"
echo -e "${YELLOW}Build started at: $(date)${NC}"

NUM_CORES=$(nproc)
echo "Using $NUM_CORES CPU cores"

make ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" -j"$NUM_CORES"

echo -e "${YELLOW}Build completed at: $(date)${NC}"
echo -e "${GREEN}✓ Kernel built successfully${NC}"
echo ""

# Step 5: Verify kernel image
echo -e "${GREEN}[5/5] Verifying kernel image...${NC}"

KERNEL_IMAGE="$LINUX_DIR/arch/riscv/boot/Image"

if [ -f "$KERNEL_IMAGE" ]; then
	KERNEL_SIZE=$(du -h "$KERNEL_IMAGE" | cut -f1)
	echo -e "${GREEN}✓ Kernel image found: $KERNEL_IMAGE${NC}"
	echo -e "${GREEN}  Size: $KERNEL_SIZE${NC}"
else
	echo -e "${RED}✗ Kernel image not found at expected location${NC}"
	exit 1
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✓ Kernel build successful!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Kernel image: $KERNEL_IMAGE"
echo "Version: $(strings "$KERNEL_IMAGE" | grep "Linux version" | head -1 || echo "Unable to determine")"
echo ""
echo "To boot with QEMU:"
echo "  export KERNEL=$KERNEL_IMAGE"
echo "  cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference"
echo "  ./run_qemu_persistent.sh"
echo ""
echo "Or use the default path:"
echo "  KERNEL=/home/user/linux/arch/riscv/boot/Image ./run_qemu_persistent.sh"
echo ""
