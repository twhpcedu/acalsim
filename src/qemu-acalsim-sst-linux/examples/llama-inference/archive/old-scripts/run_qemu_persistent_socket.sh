#!/bin/bash
# Boot from persistent root disk with socket networking
# This connects QEMU network to a socket that can be bridged to host network

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
ROOTFS_DISK=${ROOTFS_DISK:-/home/user/rootfs-persistent.qcow2}
SOCKET_PATH="/tmp/qemu-sst-llama.sock"
NETWORK_SOCKET="/tmp/qemu-network.sock"

exec $QEMU_BIN \
	-M virt \
	-cpu rv64 \
	-smp 4 \
	-m 8G \
	-kernel "$KERNEL" \
	-append "console=ttyS0 earlycon=sbi root=/dev/vda rw" \
	-drive file="$ROOTFS_DISK",if=none,id=rootfs,format=qcow2 \
	-device virtio-blk-device,drive=rootfs \
	-netdev socket,id=net0,listen=:1234 \
	-device virtio-net-device,netdev=net0 \
	-device virtio-sst-device,socket=$SOCKET_PATH \
	-nographic
