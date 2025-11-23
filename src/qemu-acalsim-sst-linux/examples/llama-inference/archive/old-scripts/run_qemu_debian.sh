#!/bin/bash
# Boot Debian RISC-V in QEMU with initramfs

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
DEBIAN_INITRAMFS=${DEBIAN_INITRAMFS:-/home/user/debian-initramfs.cpio.gz}
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

echo "Booting Debian RISC-V Linux..."
echo "Press Ctrl-A then X to exit QEMU"
echo ""
echo "Login credentials:"
echo "  Username: root"
echo "  Password: root"
echo ""

exec $QEMU_BIN \
    -M virt \
    -cpu rv64 \
    -smp 4 \
    -m 8G \
    -kernel "$KERNEL" \
    -initrd "$DEBIAN_INITRAMFS" \
    -append "console=ttyS0" \
    -netdev user,id=net0 \
    -device virtio-net-device,netdev=net0 \
    -device virtio-sst-device,socket=$SOCKET_PATH \
    -nographic
