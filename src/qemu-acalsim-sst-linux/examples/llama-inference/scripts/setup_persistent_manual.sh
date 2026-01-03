#!/bin/bash
#
# Manual Persistent Root Filesystem Setup
#
# This script creates an empty persistent disk and boots QEMU
# so you can manually format and populate it.
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
echo "Manual Persistent Root Filesystem Setup"
echo "============================================================"
echo ""

# Configuration
ROOTFS_SOURCE=${ROOTFS_SOURCE:-/home/user/rootfs}
ROOTFS_DISK=${ROOTFS_DISK:-/home/user/rootfs-persistent.qcow2}
ROOTFS_SIZE=${ROOTFS_SIZE:-10G}
QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
INITRAMFS=${INITRAMFS:-/home/user/initramfs.cpio.gz}
SOCKET_PATH="/tmp/qemu-sst-llama-setup.sock"

echo "Configuration:"
echo "  Source rootfs: $ROOTFS_SOURCE"
echo "  Disk image: $ROOTFS_DISK"
echo "  Size: $ROOTFS_SIZE"
echo ""

# Check if disk already exists
if [ -f "$ROOTFS_DISK" ]; then
	echo -e "${YELLOW}⚠${NC}  Disk image already exists: $ROOTFS_DISK"
	read -p "Overwrite? (y/N) " -n 1 -r
	echo
	if [[ ! $REPLY =~ ^[Yy]$ ]]; then
		echo "Aborted"
		exit 1
	fi
	rm -f "$ROOTFS_DISK"
fi

# Create empty disk image
echo "Creating empty disk image..."
qemu-img create -f qcow2 "$ROOTFS_DISK" "$ROOTFS_SIZE"

if [ ! -f "$ROOTFS_DISK" ]; then
	echo -e "${RED}✗${NC} Failed to create disk image"
	exit 1
fi

echo -e "${GREEN}✓${NC} Disk image created: $ROOTFS_DISK"

# Create rootfs tarball for manual installation
echo ""
echo "Creating rootfs tarball..."
cd "$ROOTFS_SOURCE"
tar czf /tmp/rootfs-install.tar.gz .

echo -e "${GREEN}✓${NC} Created /tmp/rootfs-install.tar.gz"

# Create run script for persistent disk
cat >"$ROOTFS_SOURCE/../run_qemu_persistent.sh" <<'EOFSCRIPT'
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

chmod +x "$ROOTFS_SOURCE/../run_qemu_persistent.sh"

echo -e "${GREEN}✓${NC} Created run_qemu_persistent.sh"

echo ""
echo "============================================================"
echo "READY FOR MANUAL SETUP"
echo "============================================================"
echo ""
echo "The empty disk has been created. Now you need to format"
echo "and populate it manually."
echo ""
echo "QEMU will now boot with:"
echo "  - The regular initramfs (Linux system)"
echo "  - The empty persistent disk attached as /dev/vda"
echo ""
echo "After Linux boots, run these commands:"
echo ""
echo -e "${BLUE}# 1. Format the disk${NC}"
echo "  mkfs.ext4 -F /dev/vda"
echo ""
echo -e "${BLUE}# 2. Mount it${NC}"
echo "  mkdir -p /mnt/rootfs"
echo "  mount /dev/vda /mnt/rootfs"
echo ""
echo -e "${BLUE}# 3. Extract the rootfs${NC}"
echo "  cd /mnt/rootfs"
echo "  tar xzf /tmp/rootfs-install.tar.gz"
echo ""
echo -e "${BLUE}# 4. Install LLAMA app${NC}"
echo "  mkdir -p /mnt/rootfs/apps/llama-inference"
echo "  # Copy files from /apps/llama-inference if they exist"
echo ""
echo -e "${BLUE}# 5. Sync and poweroff${NC}"
echo "  sync"
echo "  poweroff"
echo ""
echo "============================================================"
echo ""
read -p "Press Enter to boot QEMU for manual setup..."

# Boot QEMU with initramfs and empty disk
exec $QEMU_BIN \
	-M virt \
	-cpu rv64 \
	-smp 2 \
	-m 4G \
	-kernel "$KERNEL" \
	-initrd "$INITRAMFS" \
	-append "console=ttyS0 earlycon=sbi rdinit=/init" \
	-drive file="$ROOTFS_DISK",if=none,id=rootfs,format=qcow2 \
	-device virtio-blk-device,drive=rootfs \
	-nographic
