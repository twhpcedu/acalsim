#!/bin/bash
# Quick boot script for Debian RISC-V

QEMU_BIN=/home/user/qemu-build/qemu/build/qemu-system-riscv64
KERNEL=/home/user/debian-riscv/dqib_riscv64-virt/kernel
INITRD=/home/user/debian-riscv/dqib_riscv64-virt/initrd
DEBIAN_DISK=/home/user/debian-riscv64.qcow2

exec $QEMU_BIN \
	-M virt \
	-cpu rv64 \
	-smp 4 \
	-m 8G \
	-kernel "$KERNEL" \
	-initrd "$INITRD" \
	-append "console=ttyS0 root=/dev/vda1 rootwait rw" \
	-drive file="$DEBIAN_DISK",if=none,id=hd,format=qcow2 \
	-device virtio-blk-device,drive=hd \
	-netdev user,id=net0,hostfwd=tcp::2222-:22 \
	-device virtio-net-device,netdev=net0 \
	-object rng-random,filename=/dev/urandom,id=rng \
	-device virtio-rng-device,rng=rng \
	-nographic
