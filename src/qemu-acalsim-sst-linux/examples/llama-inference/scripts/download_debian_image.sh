#\!/bin/bash
#
# Download Pre-built Debian RISC-V Image
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================================"
echo "Download Pre-built Debian RISC-V Image"
echo "============================================================"
echo ""

WORK_DIR="/home/user/debian-riscv"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Searching for available Debian RISC-V images..."
echo ""

# Option 1: Debian official cloud images (if available)
DEBIAN_CLOUD_BASE="https://cloud.debian.org/images/cloud"

# Option 2: Debian Ports artifacts
DEBIAN_PORTS_BASE="https://people.debian.org/~gio/dqib"

# Option 3: Known working source
IMAGE_URL="https://gitlab.com/giomasce/dqib/-/jobs/artifacts/master/raw/artifacts/riscv64-virt/image.qcow2?job=convert_riscv64-virt"
IMAGE_FILE="debian-riscv64.qcow2"

echo "Downloading Debian RISC-V disk image..."
echo "Source: GitLab DQIB project"
echo "Size: ~500MB-1GB"
echo ""

if [ -f "$IMAGE_FILE" ]; then
    echo -e "${YELLOW}⚠${NC}  Image already exists: $IMAGE_FILE"
    read -p "Re-download? (y/N) " -n 1 -r
    echo
    if [[ \! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing image"
        exit 0
    fi
    rm -f "$IMAGE_FILE"
fi

wget -O "$IMAGE_FILE" "$IMAGE_URL" || {
    echo ""
    echo -e "${RED}✗${NC} Download failed from primary source"
    echo ""
    echo "Trying alternative: Create from rootfs tarball..."
    
    # Alternative: Download rootfs tarball
    ROOTFS_URL="https://people.debian.org/~gio/dqib/riscv64-virt/rootfs.tar.xz"
    ROOTFS_FILE="debian-rootfs.tar.xz"
    
    wget -O "$ROOTFS_FILE" "$ROOTFS_URL" || {
        echo -e "${RED}✗${NC} Alternative download also failed"
        echo ""
        echo "Please try one of these manual methods:"
        echo "1. Visit: https://gitlab.com/giomasce/dqib"
        echo "2. Or: https://people.debian.org/~gio/dqib/"
        echo "3. Download RISC-V artifacts manually"
        exit 1
    }
    
    echo -e "${GREEN}✓${NC} Downloaded rootfs tarball"
    echo ""
    echo "Creating qcow2 image from tarball..."
    
    # Create image
    qemu-img create -f qcow2 "$IMAGE_FILE" 10G
    
    # Note: Populating requires NBD or loop device
    echo -e "${YELLOW}⚠${NC}  Rootfs downloaded but needs to be converted to bootable image"
    echo "See DEBIAN_SETUP_GUIDE.md for manual setup"
    exit 1
}

if [ -f "$IMAGE_FILE" ]; then
    IMAGE_SIZE=$(du -h "$IMAGE_FILE" | cut -f1)
    echo ""
    echo -e "${GREEN}✓${NC} Downloaded Debian RISC-V image"
    echo "  Location: $WORK_DIR/$IMAGE_FILE"
    echo "  Size: $IMAGE_SIZE"
    echo ""
    
    # Expand image to give more space
    echo "Expanding image to 20GB..."
    qemu-img resize "$IMAGE_FILE" 20G
    
    echo -e "${GREEN}✓${NC} Image ready to boot"
    echo ""
    echo "Next steps:"
    echo "  1. Boot Debian:"
    echo "     cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference"
    echo "     ./run_qemu_debian_disk.sh"
    echo "  2. Default login (check documentation for credentials)"
    echo "  3. Expand root partition on first boot:"
    echo "     resize2fs /dev/vda1"
    echo ""
else
    echo -e "${RED}✗${NC} Failed to download image"
    exit 1
fi
