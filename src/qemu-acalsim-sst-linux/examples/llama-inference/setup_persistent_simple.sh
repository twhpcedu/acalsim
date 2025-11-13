#!/bin/bash
#
# Simple Persistent Root Filesystem Setup (Works in Docker!)
#
# Uses genext2fs to create filesystem without needing mount/loop devices
#
# Copyright 2023-2025 Playlab/ACAL
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
echo "Simple Persistent Root Filesystem Setup"
echo "============================================================"
echo ""

# Configuration
ROOTFS_SOURCE=${ROOTFS_SOURCE:-/home/user/rootfs}
ROOTFS_DISK=${ROOTFS_DISK:-/home/user/rootfs-persistent.qcow2}
ROOTFS_EXT2=${ROOTFS_EXT2:-/home/user/rootfs-persistent.ext2}
ROOTFS_SIZE_MB=${ROOTFS_SIZE_MB:-10240} # 10GB

echo "Configuration:"
echo "  Source rootfs: $ROOTFS_SOURCE"
echo "  Output disk: $ROOTFS_DISK"
echo "  Size: ${ROOTFS_SIZE_MB}MB (~$((ROOTFS_SIZE_MB / 1024))GB)"
echo ""

# Check if rootfs exists
if [ ! -d "$ROOTFS_SOURCE" ]; then
	echo -e "${RED}✗${NC} Source rootfs not found: $ROOTFS_SOURCE"
	exit 1
fi

echo -e "${GREEN}✓${NC} Source rootfs found"

# Check if disk already exists
if [ -f "$ROOTFS_DISK" ]; then
	echo -e "${YELLOW}⚠${NC}  Disk image already exists: $ROOTFS_DISK"
	read -p "Overwrite? (y/N) " -n 1 -r
	echo
	if [[ ! $REPLY =~ ^[Yy]$ ]]; then
		echo "Aborted"
		exit 1
	fi
	rm -f "$ROOTFS_DISK" "$ROOTFS_EXT2"
fi

echo ""
echo "Installing genext2fs..."

# Install genext2fs if not available
if ! command -v genext2fs &>/dev/null; then
	sudo apt-get update -qq
	sudo apt-get install -y -qq genext2fs >/dev/null 2>&1

	if ! command -v genext2fs &>/dev/null; then
		echo -e "${RED}✗${NC} Failed to install genext2fs"
		exit 1
	fi
fi

echo -e "${GREEN}✓${NC} genext2fs available"

# Create temporary staging directory with LLAMA app
echo ""
echo "Preparing filesystem contents..."

STAGING_DIR=$(mktemp -d)

# Copy rootfs, excluding /dev (device files need special privileges)
# We'll let devtmpfs populate /dev at boot time
cd "$ROOTFS_SOURCE"
for item in *; do
	if [ "$item" != "dev" ]; then
		cp -a "$item" "$STAGING_DIR/" 2>/dev/null || true
	fi
done

# Create empty dev directory
mkdir -p "$STAGING_DIR/dev"

# Add LLAMA app
mkdir -p "$STAGING_DIR/apps/llama-inference"
cp llama_inference.py "$STAGING_DIR/apps/llama-inference/" 2>/dev/null || true
cp llama_sst_backend.py "$STAGING_DIR/apps/llama-inference/" 2>/dev/null || true
cp test_prompts.txt "$STAGING_DIR/apps/llama-inference/" 2>/dev/null || true
cp README.md "$STAGING_DIR/apps/llama-inference/" 2>/dev/null || true
chmod +x "$STAGING_DIR/apps/llama-inference/llama_inference.py" 2>/dev/null || true

echo -e "${GREEN}✓${NC} Contents prepared"

# Calculate number of blocks needed
echo ""
echo "Calculating filesystem size..."

# Get size in KB
SIZE_KB=$(du -sk "$STAGING_DIR" | cut -f1)
# Add 20% overhead + 100MB for future growth
BLOCKS=$((SIZE_KB + SIZE_KB / 5 + 102400))

echo "  Source size: ${SIZE_KB}KB"
echo "  Filesystem blocks: ${BLOCKS}"

# Create ext2 filesystem from directory
echo ""
echo "Creating ext2 filesystem..."
genext2fs -b $BLOCKS -d "$STAGING_DIR" "$ROOTFS_EXT2"

if [ ! -f "$ROOTFS_EXT2" ]; then
	echo -e "${RED}✗${NC} Failed to create ext2 filesystem"
	rm -rf "$STAGING_DIR"
	exit 1
fi

echo -e "${GREEN}✓${NC} Ext2 filesystem created"

# Cleanup staging
rm -rf "$STAGING_DIR"

# Resize to desired size
echo ""
echo "Resizing to ${ROOTFS_SIZE_MB}MB..."
e2fsck -f -y "$ROOTFS_EXT2" >/dev/null 2>&1 || true
resize2fs "$ROOTFS_EXT2" "${ROOTFS_SIZE_MB}M" >/dev/null 2>&1

# Convert ext2 to ext4
echo "Converting to ext4..."
tune2fs -O extents,uninit_bg,dir_index "$ROOTFS_EXT2" >/dev/null 2>&1
e2fsck -f -y "$ROOTFS_EXT2" >/dev/null 2>&1 || true

echo -e "${GREEN}✓${NC} Filesystem ready"

# Convert to qcow2
echo ""
echo "Converting to qcow2..."
qemu-img convert -f raw -O qcow2 "$ROOTFS_EXT2" "$ROOTFS_DISK"

if [ ! -f "$ROOTFS_DISK" ]; then
	echo -e "${RED}✗${NC} Failed to convert to qcow2"
	exit 1
fi

# Cleanup ext2 file
rm -f "$ROOTFS_EXT2"

# Show result
DISK_SIZE=$(du -h "$ROOTFS_DISK" | cut -f1)
echo -e "${GREEN}✓${NC} Created: $ROOTFS_DISK ($DISK_SIZE)"

# Create boot script
echo ""
echo "Creating boot script..."

cat >"$(dirname "$0")/run_qemu_persistent.sh" <<'EOFSCRIPT'
#!/bin/bash
# Boot from persistent root disk

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
ROOTFS_DISK=${ROOTFS_DISK:-/home/user/rootfs-persistent.qcow2}
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

exec $QEMU_BIN \
    -M virt \
    -cpu rv64 \
    -smp 4 \
    -m 8G \
    -kernel "$KERNEL" \
    -append "console=ttyS0 earlycon=sbi root=/dev/vda rw" \
    -drive file="$ROOTFS_DISK",if=none,id=rootfs,format=qcow2 \
    -device virtio-blk-device,drive=rootfs \
    -device virtio-sst-device,socket=$SOCKET_PATH \
    -nographic
EOFSCRIPT

chmod +x "$(dirname "$0")/run_qemu_persistent.sh"

echo -e "${GREEN}✓${NC} Created run_qemu_persistent.sh"

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Created persistent root filesystem:"
echo "  Disk: $ROOTFS_DISK"
echo "  Size: $DISK_SIZE"
echo ""
echo "To use:"
echo "  1. Terminal 1: ./run_sst.sh"
echo "  2. Terminal 2: ./run_qemu_persistent.sh"
echo ""
echo "All changes made in Linux will persist across reboots!"
echo ""
echo "To install Python/PyTorch:"
echo "  - Boot the system with run_qemu_persistent.sh"
echo "  - Follow PYTORCH_LLAMA_SETUP.md Steps 2-3"
echo "  - Changes will be saved automatically to the disk"
echo "============================================================"
