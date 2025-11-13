#!/bin/bash
#
# QEMU Launch Script for Linux SST Integration
#
# Copyright 2023-2025 Playlab/ACAL
#
# This script launches QEMU with Linux kernel and VirtIO SST device.
#

# Configuration
QEMU="${QEMU:-qemu-system-riscv64}"
KERNEL="${KERNEL:-vmlinux}"
INITRD="${INITRD:-initramfs.cpio.gz}"
SOCKET="${SOCKET:-/tmp/qemu-sst-linux.sock}"
DEVICE_ID="${DEVICE_ID:-0}"
MEMORY="${MEMORY:-2G}"
CPUS="${CPUS:-4}"

# Display configuration
echo "============================================"
echo "  QEMU Linux SST Integration"
echo "============================================"
echo "QEMU:      $QEMU"
echo "Kernel:    $KERNEL"
echo "Initrd:    $INITRD"
echo "Socket:    $SOCKET"
echo "Device ID: $DEVICE_ID"
echo "Memory:    $MEMORY"
echo "CPUs:      $CPUS"
echo "============================================"
echo ""

# Check prerequisites
if [ ! -x "$(command -v $QEMU)" ]; then
	echo "Error: QEMU not found: $QEMU"
	echo "Set QEMU environment variable or install qemu-system-riscv64"
	exit 1
fi

if [ ! -f "$KERNEL" ]; then
	echo "Error: Kernel not found: $KERNEL"
	echo "Set KERNEL environment variable to point to vmlinux"
	exit 1
fi

# Build kernel command line
CMDLINE="console=ttyS0 earlycon=sbi"

# Add initrd to cmdline if present
if [ -f "$INITRD" ]; then
	echo "Using initrd: $INITRD"
	INITRD_ARG="-initrd $INITRD"
else
	echo "Warning: Initrd not found: $INITRD"
	echo "Booting without initrd"
	INITRD_ARG=""
fi

# Launch QEMU
echo ""
echo "Starting QEMU..."
echo "Note: Make sure SST is running and listening on $SOCKET"
echo ""

exec $QEMU \
	-machine virt \
	-cpu rv64 \
	-m $MEMORY \
	-smp $CPUS \
	-nographic \
	-kernel $KERNEL \
	$INITRD_ARG \
	-append "$CMDLINE" \
	-device virtio-sst-device,socket=$SOCKET,device-id=$DEVICE_ID \
	-serial mon:stdio \
	"$@"
