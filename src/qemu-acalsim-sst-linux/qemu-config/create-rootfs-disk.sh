#!/bin/bash
# Create rootfs disk from buildroot output using QEMU

set -e

BUILDROOT_TAR="/home/user/buildroot-llama/buildroot-2024.02/output/images/rootfs.tar"
OUTPUT_DISK="/home/user/rootfs-python-persistent.qcow2"
TEMP_DISK="/tmp/rootfs-temp.qcow2"
KERNEL="/home/user/linux/arch/riscv/boot/Image"
QEMU="/home/user/qemu-build/qemu/build/qemu-system-riscv64"

echo "Creating rootfs disk from buildroot..."

# Check if buildroot output exists
if [ ! -f "$BUILDROOT_TAR" ]; then
    echo "ERROR: Buildroot output not found at $BUILDROOT_TAR"
    echo "Please run setup_buildroot_python.sh first"
    exit 1
fi

# Create empty 10GB disk
echo "Creating 10GB qcow2 disk..."
qemu-img create -f qcow2 "$TEMP_DISK" 10G

# Create init script to format and populate disk
cat > /tmp/populate-disk.sh << 'INITEOF'
#!/bin/sh
echo "=== Formatting disk ==="
mkfs.ext4 -F /dev/vda
mount /dev/vda /mnt

echo "=== Extracting rootfs ==="
cd /mnt
tar xf /rootfs.tar

echo "=== Setting permissions ==="
chmod 755 /mnt
chmod 4755 /mnt/bin/busybox 2>/dev/null || true
chmod 1777 /mnt/tmp

echo "=== Syncing ==="
sync
umount /mnt

echo "=== Done ==="
poweroff -f
INITEOF

chmod +x /tmp/populate-disk.sh

# Create initramfs with the script and rootfs.tar
echo "Creating temporary initramfs..."
INITRAMFS_DIR="/tmp/initramfs-populate"
rm -rf "$INITRAMFS_DIR"
mkdir -p "$INITRAMFS_DIR"/{bin,dev,mnt,proc,sys,tmp}

# Copy busybox
cp /bin/busybox "$INITRAMFS_DIR/bin/"
cd "$INITRAMFS_DIR/bin"
for cmd in sh mount umount mkfs.ext4 poweroff sync tar cd chmod; do
    ln -sf busybox $cmd 2>/dev/null || true
done
cd -

# Copy rootfs.tar
cp "$BUILDROOT_TAR" "$INITRAMFS_DIR/rootfs.tar"

# Copy populate script as init
cp /tmp/populate-disk.sh "$INITRAMFS_DIR/init"
chmod +x "$INITRAMFS_DIR/init"

# Create initramfs
cd "$INITRAMFS_DIR"
find . | cpio -o -H newc | gzip > /tmp/populate.cpio.gz

echo "Booting QEMU to populate disk..."
timeout 120 "$QEMU" \
    -M virt \
    -cpu rv64 \
    -smp 2 \
    -m 2G \
    -kernel "$KERNEL" \
    -initrd /tmp/populate.cpio.gz \
    -drive file="$TEMP_DISK",if=none,id=hd,format=qcow2 \
    -device virtio-blk-device,drive=hd \
    -append "console=ttyS0" \
    -nographic || true

echo ""
echo "Moving disk to final location..."
mv "$TEMP_DISK" "$OUTPUT_DISK"

echo ""
echo "✓ Rootfs disk created successfully!"
echo "  Location: $OUTPUT_DISK"
echo "  Size: $(du -h "$OUTPUT_DISK" | cut -f1)"
echo ""
echo "To boot:"
echo "  cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference"
echo "  ./run_qemu_persistent.sh"
