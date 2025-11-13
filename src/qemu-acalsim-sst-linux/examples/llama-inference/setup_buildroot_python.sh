#!/bin/bash
#
# Automated Buildroot Setup for LLAMA Inference
#
# This script:
# 1. Downloads Buildroot 2024.02
# 2. Configures it for RISC-V with Python 3
# 3. Builds complete Linux rootfs (2-4 hours)
# 4. Creates persistent disk with Python environment
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
echo "Automated Buildroot Setup for LLAMA Inference"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Download Buildroot 2024.02 (~4MB)"
echo "  2. Configure for RISC-V + Python 3 + NumPy"
echo "  3. Build complete Linux system (2-4 hours)"
echo "  4. Create 10GB persistent disk with rootfs"
echo ""
echo "Requirements:"
echo "  - 20GB free disk space"
echo "  - 2-4 hours build time"
echo "  - Internet connection"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	echo "Aborted"
	exit 1
fi

# Configuration
BUILDROOT_VERSION="2024.02"
BUILDROOT_URL="https://buildroot.org/downloads/buildroot-${BUILDROOT_VERSION}.tar.gz"
WORK_DIR="/home/user/buildroot-llama"
BUILDROOT_DIR="$WORK_DIR/buildroot-${BUILDROOT_VERSION}"
OUTPUT_DIR="$BUILDROOT_DIR/output"

echo ""
echo "============================================================"
echo "Step 1: Download Buildroot"
echo "============================================================"
echo ""

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ -f "buildroot-${BUILDROOT_VERSION}.tar.gz" ]; then
	echo -e "${YELLOW}⚠${NC}  Buildroot tarball already exists, skipping download"
else
	echo "Downloading Buildroot ${BUILDROOT_VERSION}..."
	wget -q --show-progress "$BUILDROOT_URL"
	echo -e "${GREEN}✓${NC} Downloaded"
fi

if [ -d "$BUILDROOT_DIR" ]; then
	echo -e "${YELLOW}⚠${NC}  Buildroot directory exists"
	read -p "Clean and rebuild with new config? (y/N) " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		cd "$BUILDROOT_DIR"
		echo "Cleaning previous build..."
		make clean
		cd "$WORK_DIR"
	else
		echo "Keeping existing build, will update config only"
	fi
else
	echo "Extracting..."
	tar -xzf "buildroot-${BUILDROOT_VERSION}.tar.gz"
	echo -e "${GREEN}✓${NC} Extracted"
fi

cd "$BUILDROOT_DIR"

echo ""
echo "============================================================"
echo "Step 2: Configure Buildroot"
echo "============================================================"
echo ""

# Create custom defconfig
cat >configs/acalsim_riscv64_defconfig <<'EOF'
# ACAL Simulator RISC-V 64-bit Configuration with Python 3

# Target Architecture
BR2_riscv=y
BR2_RISCV_64=y
BR2_riscv_g=y
BR2_RISCV_ABI_LP64D=y

# Toolchain
BR2_TOOLCHAIN_BUILDROOT_GLIBC=y
BR2_PACKAGE_HOST_LINUX_HEADERS_CUSTOM_6_1=y
BR2_TOOLCHAIN_BUILDROOT_CXX=y
BR2_GCC_ENABLE_OPENMP=y
BR2_TOOLCHAIN_BUILDROOT_FORTRAN=y

# System Configuration
BR2_TARGET_GENERIC_HOSTNAME="acalsim-riscv"
BR2_TARGET_GENERIC_ISSUE="Welcome to ACAL Simulator RISC-V Linux"
BR2_ROOTFS_DEVICE_CREATION_DYNAMIC_EUDEV=y
BR2_TARGET_GENERIC_ROOT_PASSWD=""
BR2_SYSTEM_DHCP="eth0"

# Kernel Headers
BR2_PACKAGE_HOST_LINUX_HEADERS_CUSTOM_6_1=y

# Init System
BR2_INIT_BUSYBOX=y

# BusyBox Configuration
BR2_PACKAGE_BUSYBOX_CONFIG="package/busybox/busybox.config"
BR2_PACKAGE_BUSYBOX_SHOW_OTHERS=y

# Python 3 and packages
BR2_PACKAGE_PYTHON3=y
BR2_PACKAGE_PYTHON3_PY_ONLY=y
BR2_PACKAGE_PYTHON3_PYC_ONLY=y
BR2_PACKAGE_PYTHON_PIP=y
BR2_PACKAGE_PYTHON_SETUPTOOLS=y
BR2_PACKAGE_PYTHON_NUMPY=y

# Python modules (core)
BR2_PACKAGE_PYTHON3_BZIP2=y
BR2_PACKAGE_PYTHON3_READLINE=y
BR2_PACKAGE_PYTHON3_SSL=y
BR2_PACKAGE_PYTHON3_SQLITE=y
BR2_PACKAGE_PYTHON3_ZLIB=y
BR2_PACKAGE_PYTHON3_PYEXPAT=y
BR2_PACKAGE_PYTHON3_CURSES=y

# Development tools (for building PyTorch from source)
# Use Clang/LLVM instead of GCC for better RISC-V support
BR2_PACKAGE_LLVM=y
BR2_PACKAGE_CLANG=y
BR2_PACKAGE_BINUTILS=y
BR2_PACKAGE_MAKE=y
BR2_PACKAGE_CMAKE=y
BR2_PACKAGE_GIT=y
BR2_PACKAGE_PATCH=y
BR2_PACKAGE_DIFFUTILS=y

# Text editors
BR2_PACKAGE_VIM=y

# Compression and archiving
BR2_PACKAGE_BZIP2=y
BR2_PACKAGE_GZIP=y
BR2_PACKAGE_TAR=y
BR2_PACKAGE_XZ=y
BR2_PACKAGE_ZLIB=y

# Networking
BR2_PACKAGE_DHCPCD=y
BR2_PACKAGE_DROPBEAR=y
BR2_PACKAGE_WGET=y
BR2_PACKAGE_CA_CERTIFICATES=y

# System tools
BR2_PACKAGE_HTOP=y
BR2_PACKAGE_PROCPS_NG=y
BR2_PACKAGE_UTIL_LINUX=y
BR2_PACKAGE_UTIL_LINUX_BINARIES=y
BR2_PACKAGE_UTIL_LINUX_MOUNT=y

# Development tools
BR2_PACKAGE_GDB=y
BR2_PACKAGE_STRACE=y

# Filesystem tools
BR2_PACKAGE_E2FSPROGS=y
BR2_PACKAGE_E2FSPROGS_FSCK=y

# Libraries
BR2_PACKAGE_LIBFFI=y
BR2_PACKAGE_OPENSSL=y
BR2_PACKAGE_READLINE=y
BR2_PACKAGE_NCURSES=y

# Filesystem images
BR2_TARGET_ROOTFS_CPIO=y
BR2_TARGET_ROOTFS_CPIO_GZIP=y
BR2_TARGET_ROOTFS_TAR=y
EOF

echo "Loading custom configuration..."
make acalsim_riscv64_defconfig

echo -e "${GREEN}✓${NC} Configuration loaded"
echo ""
echo "Configuration summary:"
echo "  Architecture: RISC-V 64-bit (RV64GC)"
echo "  ABI: lp64d"
echo "  C Library: glibc"
echo "  Python: 3.x with pip, setuptools, numpy"
echo "  Compiler: Clang/LLVM (better RISC-V support than GCC)"
echo "  Dev Tools: cmake, git, vim, make, binutils"
echo "  Init: BusyBox"
echo "  Networking: DHCP, Dropbear SSH, wget"
echo "  Tools: htop, tar, gzip, e2fsprogs"
echo ""

echo ""
echo "============================================================"
echo "Step 3: Build Buildroot"
echo "============================================================"
echo ""
echo -e "${BLUE}This will take 2-4 hours depending on your system.${NC}"
echo "You can monitor progress in real-time."
echo ""
echo "Build will use $(nproc) parallel jobs"
echo ""
read -p "Start build now? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	echo ""
	echo "Build not started. To build manually later:"
	echo "  cd $BUILDROOT_DIR"
	echo "  make -j\$(nproc)"
	exit 0
fi

# Start build
echo ""
echo "Starting build at $(date)..."
echo ""

# Install build dependencies
echo "Checking build dependencies..."
if ! command -v rsync &>/dev/null; then
	echo "Installing rsync..."
	sudo apt-get update -qq && sudo apt-get install -y -qq rsync
fi
echo -e "${GREEN}✓${NC} Build dependencies ready"
echo ""

# Clean environment for Buildroot (it doesn't like LD_LIBRARY_PATH pollution)
echo "Cleaning environment variables..."
unset LD_LIBRARY_PATH
unset PKG_CONFIG_PATH

# Use PIPESTATUS to capture make exit code, not tee
set -o pipefail
env -i \
	HOME="$HOME" \
	PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
	TERM="$TERM" \
	make -j$(nproc) 2>&1 | tee buildroot_build.log
BUILD_RESULT=$?
set +o pipefail

if [ $BUILD_RESULT -eq 0 ]; then
	echo ""
	echo -e "${GREEN}✓${NC} Build completed successfully at $(date)"
else
	echo ""
	echo -e "${RED}✗${NC} Build failed with exit code $BUILD_RESULT"
	echo "Check log: $BUILDROOT_DIR/buildroot_build.log"
	exit 1
fi

echo ""
echo "============================================================"
echo "Step 4: Extract Rootfs"
echo "============================================================"
echo ""

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
