#!/bin/bash
# Boot Debian RISC-V from disk image

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
DEBIAN_DISK=${DEBIAN_DISK:-/home/user/debian-riscv/debian-riscv64.qcow2}
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

if [ ! -f "$DEBIAN_DISK" ]; then
    echo "Error: Debian disk image not found at: $DEBIAN_DISK"
    echo ""
    echo "Please run the download script first:"
    echo "  ./download_debian_image.sh"
    exit 1
fi

echo "Booting Debian RISC-V Linux from disk..."
echo "Press Ctrl-A then X to exit QEMU"
echo ""
echo "Image: $DEBIAN_DISK"
echo ""

exec $QEMU_BIN \
    -M virt \
    -cpu rv64 \
    -smp 4 \
    -m 8G \
    -kernel "$KERNEL" \
    -append "console=ttyS0 root=/dev/vda1 rw rootwait" \
    -drive file="$DEBIAN_DISK",if=none,id=rootfs,format=qcow2 \
    -device virtio-blk-device,drive=rootfs \
    -netdev user,id=net0 \
    -device virtio-net-device,netdev=net0 \
    -device virtio-sst-device,socket=$SOCKET_PATH \
    -nographic
