#!/bin/bash
# Setup VirtIO-SST device in QEMU Debian
# Run this script inside QEMU to enable /dev/sst0

set -e

echo "=========================================="
echo "VirtIO-SST Device Setup"
echo "=========================================="
echo ""

# Check if running in QEMU
if ! grep -q "QEMU" /proc/cpuinfo 2>/dev/null && ! grep -q "riscv" /proc/cpuinfo; then
	echo "⚠ Warning: Not running in QEMU RISC-V"
	echo "This script should be run inside QEMU Debian"
	read -p "Continue anyway? [y/N]: " -n 1 -r
	echo
	if [[ ! $REPLY =~ ^[Yy]$ ]]; then
		exit 1
	fi
fi

# Copy kernel module from shared folder
MODULE_SRC="/mnt/shared/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko"
MODULE_DEST=""/lib/modules/$(uname -r)/extra"/virtio-sst.ko"

if [ ! -f "$MODULE_SRC" ]; then
	echo "✗ Kernel module not found at: $MODULE_SRC"
	echo ""
	echo "Looking for module in alternate locations..."
	MODULE_SRC=$(find /mnt/shared -name "virtio-sst.ko" 2>/dev/null | head -1)
	if [ -z "$MODULE_SRC" ]; then
		echo "✗ virtio-sst.ko not found in shared folder"
		echo ""
		echo "Please ensure:"
		echo "  1. Shared folder is mounted at /mnt/shared"
		echo "  2. virtio-sst.ko exists in the ACALSIM project"
		exit 1
	fi
	echo "✓ Found module at: $MODULE_SRC"
fi

# Create extra modules directory
echo "Creating modules directory..."
sudo mkdir -p "/lib/modules/$(uname -r)/extra"

# Copy module
echo "Copying kernel module..."
sudo cp "$MODULE_SRC" "$MODULE_DEST"

# Update module dependencies
echo "Updating module dependencies..."
sudo depmod -a

# Load module
echo "Loading virtio-sst kernel module..."
if lsmod | grep -q virtio_sst; then
	echo "✓ virtio-sst already loaded"
else
	sudo modprobe virtio-sst || sudo insmod "$MODULE_DEST"
	echo "✓ virtio-sst module loaded"
fi

# Check if device exists
echo ""
echo "Checking for /dev/sst* devices..."
if ls /dev/sst* >/dev/null 2>&1; then
	echo "✓ VirtIO-SST devices found:"
	ls -l /dev/sst*
else
	echo "⚠ No /dev/sst* devices found"
	echo ""
	echo "Possible reasons:"
	echo "  1. QEMU not started with -device virtio-sst-device"
	echo "  2. VirtIO SST device not detected by kernel"
	echo "  3. Module loaded but device not created"
	echo ""
	echo "Check kernel messages:"
	dmesg | tail -20 | grep -i virtio
fi

# Check module info
echo ""
echo "Module information:"
modinfo virtio-sst 2>/dev/null || modinfo "$MODULE_DEST"

# Display kernel messages
echo ""
echo "Recent kernel messages:"
dmesg | grep -i "virtio-sst" | tail -10

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify /dev/sst0 exists: ls -l /dev/sst0"
echo "  2. Test device: python3 /mnt/shared/device_gemm/test_virtio_sst.py"
echo ""
