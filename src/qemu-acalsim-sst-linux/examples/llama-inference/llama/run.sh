echo ""
echo "============================================================"
echo "Step 4: Extract Rootfs"
echo "============================================================"
echo ""
OUTPUT_DIR="/home/user/buildroot-llama/buildroot-2024.02/output"
ROOTFS_CPIO="$OUTPUT_DIR/images/rootfs.cpio.gz"
ROOTFS_TAR="$OUTPUT_DIR/images/rootfs.tar"
ROOTFS_EXTRACT="/home/user/rootfs-python"

if [ ! -f "$ROOTFS_CPIO" ]; then
	echo -e "${RED}✗${NC} Rootfs not found: $ROOTFS_CPIO"
	exit 1
fi

echo "Extracting rootfs..."
mkdir -p "$ROOTFS_EXTRACT"
cd "$ROOTFS_EXTRACT"

gunzip -c "$ROOTFS_CPIO" | cpio -idmv >/dev/null 2>&1

echo -e "${GREEN}✓${NC} Rootfs extracted to $ROOTFS_EXTRACT"

# Verify Python
if [ -f "$ROOTFS_EXTRACT/usr/bin/python3" ]; then
	echo -e "${GREEN}✓${NC} Python 3 found"
	PYTHON_VERSION=$("$ROOTFS_EXTRACT/usr/bin/python3" --version 2>&1 || echo "unknown")
	echo "  Version: $PYTHON_VERSION"
else
	echo -e "${RED}✗${NC} Python 3 not found in rootfs"
fi

# Check size
ROOTFS_SIZE=$(du -sh "$ROOTFS_EXTRACT" | cut -f1)
echo "  Rootfs size: $ROOTFS_SIZE"

echo ""
echo "============================================================"
echo "Step 5: Create Persistent Disk with Python Rootfs"
echo "============================================================"
echo ""

# Use existing setup_persistent_simple.sh but with new rootfs
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference

echo "Creating persistent disk with Python-enabled rootfs..."

ROOTFS_SOURCE="$ROOTFS_EXTRACT" \
	ROOTFS_DISK="/home/user/rootfs-python-persistent.qcow2" \
	./setup_persistent_simple.sh

echo ""
echo "============================================================"
echo "BUILDROOT SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Created:"
echo "  ✓ Full RISC-V Linux rootfs with Python 3"
echo "  ✓ Persistent disk: /home/user/rootfs-python-persistent.qcow2"
echo "  ✓ Boot script: run_qemu_persistent.sh"
echo ""
echo "Rootfs includes:"
echo "  - Python 3 with pip, setuptools, NumPy"
echo "  - Clang/LLVM compiler toolchain"
echo "  - CMake, Git, Vim, Make, Binutils"
echo "  - BusyBox utilities"
echo "  - SSH server (Dropbear)"
echo "  - Network support (DHCP, wget)"
echo ""
echo "Next steps:"
echo "  1. Boot Linux: ./run_qemu_persistent.sh"
echo "  2. In Linux, verify Python:"
echo "     python3 --version"
echo "     python3 -c 'import numpy; print(numpy.__version__)'"
echo "  3. Install PyTorch (see PYTORCH_LLAMA_SETUP.md Step 3)"
echo "  4. Download LLAMA 2 model (see PYTORCH_LLAMA_SETUP.md Step 5)"
echo ""
echo "Total build time: ~2-4 hours"
echo "============================================================"
