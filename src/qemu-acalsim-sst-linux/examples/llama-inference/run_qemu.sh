#!/bin/bash
"""
Launch QEMU Linux for LLAMA 2 Inference

This script launches QEMU with:
- RISC-V Linux with PyTorch and Transformers
- VirtIO SST device connected to SST simulation
- Virtual disk with LLAMA 2 7B model (models.qcow2)

Copyright 2023-2025 Playlab/ACAL
Licensed under the Apache License, Version 2.0
"""

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================"
echo "LLAMA 2 Inference - QEMU Linux Launcher"
echo "============================================================"
echo ""

# Check if running inside Docker container
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Running inside Docker container"
    IN_DOCKER=true
else
    echo -e "${YELLOW}⚠${NC}  Not running in Docker container"
    echo "This script is designed to run inside the acalsim-workspace container"
    echo ""
    echo "To run inside Docker:"
    echo "  docker exec -it acalsim-workspace bash"
    echo "  cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference"
    echo "  ./run_qemu.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    IN_DOCKER=false
fi

# Configuration
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

if [ "$IN_DOCKER" = true ]; then
    # Docker paths
    QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
    KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
    INITRAMFS=${INITRAMFS:-/home/user/initramfs-full.cpio.gz}
    MODEL_DISK=${MODEL_DISK:-/home/user/models.qcow2}
else
    # Local paths (adjust as needed)
    QEMU_BIN=${QEMU_BIN:-$HOME/qemu-build/qemu/build/qemu-system-riscv64}
    KERNEL=${KERNEL:-$HOME/linux/arch/riscv/boot/Image}
    INITRAMFS=${INITRAMFS:-$HOME/initramfs-full.cpio.gz}
    MODEL_DISK=${MODEL_DISK:-$HOME/models.qcow2}
fi

MEMORY=${MEMORY:-8G}
CPUS=${CPUS:-4}

echo "Configuration:"
echo "  QEMU: $QEMU_BIN"
echo "  Kernel: $KERNEL"
echo "  Initramfs: $INITRAMFS"
echo "  Model Disk: $MODEL_DISK"
echo "  Memory: $MEMORY"
echo "  CPUs: $CPUS"
echo "  Socket: $SOCKET_PATH"
echo ""

# Check if QEMU exists
if [ ! -f "$QEMU_BIN" ]; then
    echo -e "${RED}✗${NC} QEMU binary not found: $QEMU_BIN"
    echo ""
    echo "Build QEMU with:"
    echo "  cd /home/user/qemu-build/qemu"
    echo "  ./configure --target-list=riscv64-softmmu"
    echo "  make -j\$(nproc)"
    exit 1
fi

echo -e "${GREEN}✓${NC} QEMU binary found"

# Check if kernel exists
if [ ! -f "$KERNEL" ]; then
    echo -e "${RED}✗${NC} Kernel image not found: $KERNEL"
    echo ""
    echo "Build kernel with:"
    echo "  cd /home/user/linux"
    echo "  make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- defconfig"
    echo "  make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j\$(nproc)"
    exit 1
fi

echo -e "${GREEN}✓${NC} Kernel image found"

# Check if initramfs exists
if [ ! -f "$INITRAMFS" ]; then
    echo -e "${YELLOW}⚠${NC}  Initramfs not found: $INITRAMFS"
    echo ""
    echo "Using basic initramfs. For full PyTorch support, build with:"
    echo "  See PYTORCH_LLAMA_SETUP.md for instructions"
    echo ""

    # Try to find basic initramfs
    INITRAMFS="/home/user/initramfs.cpio.gz"
    if [ ! -f "$INITRAMFS" ]; then
        echo -e "${RED}✗${NC} No initramfs found"
        exit 1
    fi
    echo -e "${YELLOW}⚠${NC}  Using basic initramfs (PyTorch may not be available)"
fi

echo -e "${GREEN}✓${NC} Initramfs found"

# Check if model disk exists
if [ ! -f "$MODEL_DISK" ]; then
    echo -e "${YELLOW}⚠${NC}  Model disk not found: $MODEL_DISK"
    echo ""
    echo "The LLAMA 2 7B model disk is not present."
    echo "You can:"
    echo "  1. Create a virtual disk:"
    echo "     qemu-img create -f qcow2 $MODEL_DISK 20G"
    echo ""
    echo "  2. Download and install LLAMA 2 model:"
    echo "     See PYTORCH_LLAMA_SETUP.md Step 5"
    echo ""
    echo "QEMU will start, but the model will not be available."
    echo ""
    MODEL_DISK_ARG=""
else
    echo -e "${GREEN}✓${NC} Model disk found: $MODEL_DISK"
    MODEL_DISK_SIZE=$(du -h "$MODEL_DISK" | cut -f1)
    echo "  Size: $MODEL_DISK_SIZE"
    MODEL_DISK_ARG="-drive file=$MODEL_DISK,if=none,id=vda,format=qcow2 -device virtio-blk-device,drive=vda"
fi

# Check if SST socket exists (SST should be running)
if [ ! -S "$SOCKET_PATH" ]; then
    echo ""
    echo -e "${YELLOW}⚠${NC}  SST socket not found: $SOCKET_PATH"
    echo ""
    echo "Make sure SST simulation is running first!"
    echo ""
    echo "In another terminal:"
    echo "  cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference"
    echo "  ./run_sst.sh"
    echo ""
    echo "Wait for 'Waiting for QEMU to connect...' message, then start QEMU."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    echo -e "${YELLOW}⚠${NC}  Continuing without SST connection (device will not work)"
fi

echo ""
echo "============================================================"
echo "Starting QEMU Linux"
echo "============================================================"
echo ""
echo "Wait for Linux to boot, then login as 'root' (no password)"
echo ""
echo -e "${BLUE}Quick Start Commands (in Linux):${NC}"
echo ""
echo "  # Mount model disk"
echo "  mount /dev/vda /mnt/models"
echo ""
echo "  # Check SST device"
echo "  ls -l /dev/sst0"
echo ""
echo "  # Run inference"
echo "  cd /apps/llama-inference"
echo "  ./llama_inference.py \"Explain quantum computing\""
echo ""
echo "============================================================"
echo ""

# Build QEMU command
QEMU_CMD="$QEMU_BIN \
    -M virt \
    -cpu rv64 \
    -smp $CPUS \
    -m $MEMORY \
    -kernel $KERNEL \
    -initrd $INITRAMFS \
    -append \"console=ttyS0 root=/dev/ram rdinit=/sbin/init\" \
    -nographic \
    -device virtio-sst-device,socket=$SOCKET_PATH \
    $MODEL_DISK_ARG"

# Show command (for debugging)
if [ "${DEBUG:-0}" = "1" ]; then
    echo "QEMU Command:"
    echo "$QEMU_CMD"
    echo ""
fi

# Run QEMU
exec $QEMU_CMD
