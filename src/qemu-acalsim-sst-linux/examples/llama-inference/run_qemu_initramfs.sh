#!/bin/bash
# Boot with initramfs to get a working shell

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
INITRAMFS=${INITRAMFS:-/home/user/initramfs-buildroot.cpio.gz}
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

echo "Booting QEMU with initramfs..."
echo "Press Ctrl-A then X to exit QEMU"
echo ""

exec $QEMU_BIN \
	-M virt \
	-cpu rv64 \
	-smp 4 \
	-m 8G \
	-kernel "$KERNEL" \
	-initrd "$INITRAMFS" \
	-append "console=ttyS0" \
	-netdev user,id=net0 \
	-device virtio-net-device,netdev=net0 \
	-device virtio-sst-device,socket=$SOCKET_PATH \
	-nographic
