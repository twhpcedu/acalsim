#!/bin/bash
#
# Setup Persistent Root Filesystem for LLAMA Inference
#
# This script creates a persistent root disk that preserves changes across reboots.
# Much better than initramfs for development and testing.
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
echo "Setup Persistent Root Filesystem"
echo "============================================================"
echo ""

# Configuration
ROOTFS_SOURCE=${ROOTFS_SOURCE:-/home/user/rootfs}
ROOTFS_DISK=${ROOTFS_DISK:-/home/user/rootfs-persistent.qcow2}
ROOTFS_SIZE=${ROOTFS_SIZE:-10G}
TEMP_MOUNT=/tmp/rootfs-setup

echo "Configuration:"
echo "  Source rootfs: $ROOTFS_SOURCE"
echo "  Disk image: $ROOTFS_DISK"
echo "  Size: $ROOTFS_SIZE"
echo ""

# Check if rootfs source exists
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
    rm -f "$ROOTFS_DISK"
fi

# Create disk image
echo ""
echo "Creating disk image..."
qemu-img create -f qcow2 "$ROOTFS_DISK" "$ROOTFS_SIZE"

if [ ! -f "$ROOTFS_DISK" ]; then
    echo -e "${RED}✗${NC} Failed to create disk image"
    exit 1
fi

echo -e "${GREEN}✓${NC} Disk image created: $ROOTFS_DISK"

# Format the disk
echo ""
echo "Formatting disk..."
echo "This uses a helper script to format the disk via NBD"
echo ""

# Use qemu-nbd to format the disk without booting QEMU
sudo modprobe nbd max_part=8 2>/dev/null || true

if ! command -v qemu-nbd &> /dev/null; then
    echo -e "${YELLOW}⚠${NC}  qemu-nbd not available, using alternative method"
    echo ""
    echo "You'll need to format the disk manually:"
    echo "  1. Boot QEMU with both the initramfs and this disk"
    echo "  2. Run: mkfs.ext4 /dev/vdb"
    echo "  3. Run: mount /dev/vdb /mnt"
    echo "  4. Run: cd /mnt && tar -xzf /rootfs.tar.gz"
    echo ""
    echo "Creating rootfs tarball for manual installation..."

    cd "$ROOTFS_SOURCE"
    tar czf /home/user/rootfs.tar.gz .

    echo -e "${GREEN}✓${NC} Created /home/user/rootfs.tar.gz"
    echo ""
    echo "Disk image created but not populated."
    echo "Follow manual steps above to populate it."
    exit 0
fi

# Connect disk via NBD
sudo qemu-nbd --connect=/dev/nbd0 "$ROOTFS_DISK"

# Format
echo "Formatting as ext4..."
sudo mkfs.ext4 /dev/nbd0

# Mount
mkdir -p "$TEMP_MOUNT"
sudo mount /dev/nbd0 "$TEMP_MOUNT"

# Copy rootfs
echo ""
echo "Copying rootfs contents..."
sudo cp -a "$ROOTFS_SOURCE"/* "$TEMP_MOUNT/"

# Add LLAMA app
echo "Installing LLAMA app..."
sudo mkdir -p "$TEMP_MOUNT/apps/llama-inference"
sudo cp llama_inference.py "$TEMP_MOUNT/apps/llama-inference/"
sudo cp llama_sst_backend.py "$TEMP_MOUNT/apps/llama-inference/"
sudo cp test_prompts.txt "$TEMP_MOUNT/apps/llama-inference/"
sudo cp README.md "$TEMP_MOUNT/apps/llama-inference/"
sudo chmod +x "$TEMP_MOUNT/apps/llama-inference/llama_inference.py"

echo -e "${GREEN}✓${NC} LLAMA app installed"

# Unmount
sudo umount "$TEMP_MOUNT"
sudo qemu-nbd --disconnect /dev/nbd0

echo -e "${GREEN}✓${NC} Disk populated and unmounted"

# Create helper script to boot from this disk
cat > run_qemu_persistent.sh << 'EOFSCRIPT'
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

chmod +x run_qemu_persistent.sh

echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Created persistent root filesystem:"
echo "  Disk: $ROOTFS_DISK"
echo "  Size: $(du -h "$ROOTFS_DISK" | cut -f1)"
echo ""
echo "To use:"
echo "  1. Terminal 1: ./run_sst.sh"
echo "  2. Terminal 2: ./run_qemu_persistent.sh"
echo ""
echo "All changes made in Linux will persist across reboots!"
echo ""
echo "To install Python/PyTorch:"
echo "  - Boot the system"
echo "  - Follow PYTORCH_LLAMA_SETUP.md Steps 2-3"
echo "  - Changes will be saved automatically"
echo "============================================================"
