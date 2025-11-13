#!/bin/bash
#
# Create Pre-formatted Persistent Root Filesystem
#
# This script creates a persistent disk using host tools to format it.
# Works around Docker NBD and initramfs tool limitations.
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
echo "Create Pre-formatted Persistent Root Filesystem"
echo "============================================================"
echo ""

# Configuration
ROOTFS_SOURCE=${ROOTFS_SOURCE:-/home/user/rootfs}
ROOTFS_DISK=${ROOTFS_DISK:-/home/user/rootfs-persistent.qcow2}
ROOTFS_RAW=${ROOTFS_RAW:-/home/user/rootfs-persistent.raw}
ROOTFS_SIZE_MB=${ROOTFS_SIZE_MB:-10240} # 10GB in MB

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
	rm -f "$ROOTFS_DISK" "$ROOTFS_RAW"
fi

echo ""
echo "============================================================"
echo "METHOD: Using mke2fs with raw disk image"
echo "============================================================"
echo ""

# Check if mke2fs is available
if ! command -v mke2fs &>/dev/null; then
	echo -e "${RED}✗${NC} mke2fs not found"
	echo ""
	echo "Installing e2fsprogs..."
	apt-get update -qq && apt-get install -y -qq e2fsprogs >/dev/null 2>&1

	if ! command -v mke2fs &>/dev/null; then
		echo -e "${RED}✗${NC} Failed to install e2fsprogs"
		exit 1
	fi
fi

echo -e "${GREEN}✓${NC} mke2fs available"

# Create raw disk image
echo ""
echo "Creating raw disk image..."
dd if=/dev/zero of="$ROOTFS_RAW" bs=1M count="$ROOTFS_SIZE_MB" status=progress 2>&1 | tail -1

if [ ! -f "$ROOTFS_RAW" ]; then
	echo -e "${RED}✗${NC} Failed to create raw disk image"
	exit 1
fi

echo -e "${GREEN}✓${NC} Raw disk created: $ROOTFS_RAW"

# Format with ext4
echo ""
echo "Formatting as ext4..."
mke2fs -t ext4 -F "$ROOTFS_RAW" >/dev/null 2>&1

echo -e "${GREEN}✓${NC} Filesystem created"

# Mount and populate
echo ""
echo "Mounting and populating..."

MOUNT_POINT=$(mktemp -d)

# Try to mount
if mount -o loop "$ROOTFS_RAW" "$MOUNT_POINT" 2>/dev/null; then
	echo -e "${GREEN}✓${NC} Mounted at $MOUNT_POINT"

	# Copy rootfs
	echo "Copying rootfs contents..."
	cp -a "$ROOTFS_SOURCE"/* "$MOUNT_POINT/"

	# Install LLAMA app
	echo "Installing LLAMA app..."
	mkdir -p "$MOUNT_POINT/apps/llama-inference"
	cp llama_inference.py "$MOUNT_POINT/apps/llama-inference/" 2>/dev/null || true
	cp llama_sst_backend.py "$MOUNT_POINT/apps/llama-inference/" 2>/dev/null || true
	cp test_prompts.txt "$MOUNT_POINT/apps/llama-inference/" 2>/dev/null || true
	cp README.md "$MOUNT_POINT/apps/llama-inference/" 2>/dev/null || true
	chmod +x "$MOUNT_POINT/apps/llama-inference/llama_inference.py" 2>/dev/null || true

	echo -e "${GREEN}✓${NC} LLAMA app installed"

	# Sync
	sync

	# Unmount
	umount "$MOUNT_POINT"
	rmdir "$MOUNT_POINT"

	echo -e "${GREEN}✓${NC} Disk populated and unmounted"

else
	echo -e "${YELLOW}⚠${NC}  Cannot mount (needs privileges or loop device support)"
	echo ""
	echo "Using alternative: e2cp to copy files..."

	# Use e2tools if available, or debugfs
	if command -v e2cp &>/dev/null; then
		# Use e2tools
		echo "Using e2tools to populate filesystem..."
		cd "$ROOTFS_SOURCE"
		find . -type f | while read file; do
			e2cp "$file" "$ROOTFS_RAW:$file" 2>/dev/null || true
		done
		echo -e "${GREEN}✓${NC} Files copied"

	elif command -v debugfs &>/dev/null; then
		# Use debugfs
		echo "Using debugfs to populate filesystem..."

		# Create command file for debugfs
		DEBUG_CMD=$(mktemp)
		cat >"$DEBUG_CMD" <<EOF
cd /
mkdir apps
cd /apps
mkdir llama-inference
cd /
quit
EOF

		debugfs -w -f "$DEBUG_CMD" "$ROOTFS_RAW" >/dev/null 2>&1
		rm "$DEBUG_CMD"

		echo -e "${YELLOW}⚠${NC}  Partial population - filesystem created but empty"
		echo "    You'll need to populate it manually on first boot"

	else
		echo -e "${YELLOW}⚠${NC}  No copy tools available"
		echo "    Created empty filesystem - populate manually on first boot"
	fi

	rmdir "$MOUNT_POINT" 2>/dev/null || true
fi

# Convert to qcow2
echo ""
echo "Converting to qcow2 format..."
qemu-img convert -f raw -O qcow2 "$ROOTFS_RAW" "$ROOTFS_DISK"

if [ ! -f "$ROOTFS_DISK" ]; then
	echo -e "${RED}✗${NC} Failed to convert to qcow2"
	exit 1
fi

echo -e "${GREEN}✓${NC} Converted to qcow2"

# Clean up raw image
rm -f "$ROOTFS_RAW"

# Show result
DISK_SIZE=$(du -h "$ROOTFS_DISK" | cut -f1)
echo -e "${GREEN}✓${NC} Disk image ready: $ROOTFS_DISK ($DISK_SIZE)"

# Create run script
echo ""
echo "Creating boot script..."

cat >"$(dirname $0)/run_qemu_persistent.sh" <<'EOFSCRIPT'
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

chmod +x "$(dirname $0)/run_qemu_persistent.sh"

echo -e "${GREEN}✓${NC} Created run_qemu_persistent.sh"

echo ""
echo "============================================================"
echo "SETUP COMPLETE"
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
echo "  - Boot the system"
echo "  - Follow PYTORCH_LLAMA_SETUP.md Steps 2-3"
echo "  - Changes will be saved automatically"
echo "============================================================"
