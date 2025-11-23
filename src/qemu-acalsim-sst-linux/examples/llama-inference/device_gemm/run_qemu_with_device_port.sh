#!/bin/bash
# Boot DQIB Debian RISC-V image

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
DQIB_DIR="/home/user/debian-riscv/dqib_riscv64-virt"
KERNEL="$DQIB_DIR/kernel"
INITRD="$DQIB_DIR/initrd"
DEBIAN_DISK="$DQIB_DIR/image.qcow2"
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

# Shared folder configuration
SHARED_HOST_DIR="/home/user/projects"
SHARED_MOUNT_TAG="hostshare"

if [ ! -f "$DEBIAN_DISK" ]; then
	echo "Error: DQIB Debian disk not found at: $DEBIAN_DISK"
	echo ""
	echo "Please download it first - see documentation"
	exit 1
fi

echo "======================================"
echo "Booting DQIB Debian RISC-V Linux"
echo "======================================"
echo ""
echo "Login credentials:"
echo "  root / root"
echo "  debian / debian"
echo ""
echo "SSH access: ssh -p 2222 debian@localhost"
echo ""
echo "Shared Folder:"
echo "  Host (Docker): $SHARED_HOST_DIR"
echo "  Guest mount point: /mnt/shared"
echo "  Already configured in /etc/fstab"
echo ""
echo "Press Ctrl-A then X to exit QEMU"
echo "======================================"
echo ""

exec $QEMU_BIN \
	-M virt \
	-cpu rv64 \
	-smp 4 \
	-m 32G \
	-kernel "$KERNEL" \
	-initrd "$INITRD" \
	-append "root=/dev/vda1 rootwait console=ttyS0" \
	-drive file="$DEBIAN_DISK",if=none,id=hd,format=qcow2 \
	-device virtio-blk-device,drive=hd \
	-netdev user,id=net0,hostfwd=tcp:127.0.0.1:2222-:22,hostfwd=tcp:127.0.0.1:9999-:9999 \
	-device virtio-net-device,netdev=net0 \
	-device virtio-sst-device,socket=$SOCKET_PATH \
	-object rng-random,filename=/dev/urandom,id=rng \
	-device virtio-rng-device,rng=rng \
	-virtfs local,path=$SHARED_HOST_DIR,mount_tag=$SHARED_MOUNT_TAG,security_model=passthrough,id=hostshare \
	-nographic
