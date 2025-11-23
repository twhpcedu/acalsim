#!/bin/bash
# Boot from persistent root disk

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
KERNEL=${KERNEL:-/home/user/linux/arch/riscv/boot/Image}
ROOTFS_DISK=${ROOTFS_DISK:-/home/user/rootfs-persistent.qcow2}
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

exec $QEMU_BIN \
    -M virt \
    -cpu rv64 \
    -smp 4 \
    -m 8G \
    -kernel "$KERNEL" \
    -append "console=ttyS0 earlycon=sbi root=/dev/vda rw rootfstype=ext4 rootwait" \
    -drive file="$ROOTFS_DISK",if=none,id=rootfs,format=qcow2 \
    -device virtio-blk-device,drive=rootfs \
    -netdev user,id=net0 \
    -device virtio-net-device,netdev=net0 \
    -device virtio-sst-device,socket=$SOCKET_PATH \
    -nographic
