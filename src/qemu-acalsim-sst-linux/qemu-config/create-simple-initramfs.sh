#!/bin/bash
# Create a simple initramfs with busybox for testing

set -e

INITRAMFS_DIR="/tmp/initramfs"
OUTPUT="/home/user/initramfs.cpio.gz"

echo "Creating simple initramfs..."

# Clean and create directory
rm -rf "$INITRAMFS_DIR"
mkdir -p "$INITRAMFS_DIR"

# Create directory structure
cd "$INITRAMFS_DIR"
mkdir -p bin sbin etc proc sys dev tmp root

# Create init script
cat > init << 'EOF'
#!/bin/sh

# Mount essential filesystems
mount -t proc none /proc
mount -t sysfs none /sys
mount -t devtmpfs none /dev

echo ""
echo "================================================"
echo "  Minimal Linux System - QEMU RISC-V"
echo "================================================"
echo ""
echo "System booted successfully!"
echo ""
echo "Available commands:"
echo "  - ls, cat, echo, mount, ps, etc."
echo "  - exit: to shutdown"
echo ""

# Start shell
exec /bin/sh
EOF

chmod +x init

# Check if busybox is available
if command -v busybox >/dev/null 2>&1; then
    echo "Using system busybox..."
    cp $(which busybox) bin/busybox
else
    echo "ERROR: busybox not found. Installing..."
    sudo apt-get update -qq
    sudo apt-get install -y busybox-static
    cp /bin/busybox bin/busybox
fi

# Create busybox symlinks
cd bin
for cmd in sh ls cat echo mount umount ps kill reboot poweroff; do
    ln -sf busybox $cmd
done
cd ..

# Create basic device nodes (if running as root)
if [ "$(id -u)" = "0" ]; then
    mknod -m 666 dev/null c 1 3
    mknod -m 666 dev/zero c 1 5
    mknod -m 666 dev/tty c 5 0
    mknod -m 666 dev/console c 5 1
    mknod -m 666 dev/ttyS0 c 4 64
fi

# Create initramfs
echo "Creating initramfs archive..."
find . -print0 | cpio --null -ov --format=newc | gzip -9 > "$OUTPUT"

echo ""
echo "Initramfs created: $OUTPUT"
ls -lh "$OUTPUT"
echo ""
echo "To boot with this initramfs:"
echo "  Add to QEMU: -initrd $OUTPUT"
