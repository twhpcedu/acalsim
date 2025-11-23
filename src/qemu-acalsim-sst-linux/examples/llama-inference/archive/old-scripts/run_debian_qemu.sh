#!/bin/bash
# Boot Debian RISC-V with SST support

QEMU_BIN=${QEMU_BIN:-/home/user/qemu-build/qemu/build/qemu-system-riscv64}
# Use Debian-provided kernel and initrd (has proper virtio drivers)
KERNEL=${KERNEL:-/home/user/debian-riscv/dqib_riscv64-virt/kernel}
INITRD=${INITRD:-/home/user/debian-riscv/dqib_riscv64-virt/initrd}
DEBIAN_DISK=${DEBIAN_DISK:-/home/user/debian-riscv64.qcow2}
SOCKET_PATH="/tmp/qemu-sst-llama.sock"

# Remove old socket
rm -f "$SOCKET_PATH"

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
	-device virtio-sst-device,socket=$SOCKET_PATH \
	-object rng-random,filename=/dev/urandom,id=rng \
	-device virtio-rng-device,rng=rng \
	-nographic
