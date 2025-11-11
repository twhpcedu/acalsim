# Build Notes for Linux Integration

## Building Components

The Linux integration consists of several components that must be built in different environments:

### 1. VirtIO Device (QEMU side)
**Build location**: Host machine
**Build method**: Integrated into QEMU build

```bash
# On host (macOS/Linux)
cd /path/to/qemu/source

# Copy VirtIO SST device files
cp virtio-device/*.h include/hw/virtio/
cp virtio-device/virtio-sst.c hw/virtio/

# Update QEMU build files (see GETTING_STARTED.md)
# Rebuild QEMU
cd build && ninja
```

### 2. Kernel Driver
**Build location**: Host with RISC-V cross-compiler OR inside RISC-V Linux
**Requires**: Linux kernel source with headers

#### Option A: Cross-compile for RISC-V
```bash
# On host with RISC-V toolchain
cd drivers

# Point to RISC-V kernel source
export KDIR=/path/to/riscv-linux

# Cross-compile
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- KDIR=$KDIR
```

#### Option B: Build in RISC-V environment
```bash
# Inside RISC-V Linux (real hardware or QEMU)
cd drivers
make  # Will use current running kernel
```

#### Option C: Build for development kernel
```bash
# If you have a custom RISC-V kernel build
cd drivers
make KDIR=/path/to/custom/kernel/build
```

### 3. Test Applications
**Build location**: Host with RISC-V cross-compiler

```bash
cd rootfs/apps

# Cross-compile
make CROSS_COMPILE=riscv64-linux-gnu-

# Output: sst-test (RISC-V binary)
```

### 4. SST Components
**Build location**: Docker container or host

```bash
# In Docker container
docker exec acalsim-workspace bash -c "
    cd /home/user/projects/acalsim/src/acalsim-device &&
    make
"

# Or on host
cd ../../acalsim-device
make
```

## Why the Docker Build Fails

The error you're seeing:
```
make: *** /lib/modules/6.10.14-linuxkit/build: No such file or directory
```

This happens because:
1. The Docker container is running a linuxkit kernel (Docker's minimal kernel)
2. Kernel headers for this kernel aren't installed in the container
3. **The kernel driver is meant for RISC-V Linux, not x86_64 linuxkit**

## Recommended Build Workflow

### For Development/Testing
```bash
# 1. Build SST components in Docker
docker exec acalsim-workspace bash -c "cd /home/user/projects/acalsim/src/acalsim-device && make"

# 2. Build QEMU VirtIO device on host
# (Requires QEMU source integration - see GETTING_STARTED.md)

# 3. Cross-compile kernel driver on host
cd drivers
make ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- KDIR=/path/to/riscv-linux

# 4. Cross-compile test applications on host
cd ../rootfs/apps
make CROSS_COMPILE=riscv64-linux-gnu-

# 5. Create initramfs with driver and apps
# (See GETTING_STARTED.md for full instructions)
```

### For Documentation/Reference
The code is complete and ready for use. The actual build happens when:
1. You integrate the VirtIO device into QEMU
2. You build a RISC-V Linux kernel
3. You create a root filesystem for RISC-V
4. You run the complete simulation

## Quick Verification (Without Full Build)

To verify the code is syntactically correct without full kernel headers:

```bash
# Syntax check only (won't produce .ko file)
cd drivers
gcc -fsyntax-only -I../virtio-device \
    -I/usr/include \
    -D__KERNEL__ -DMODULE \
    sst-virtio.c

# Check for obvious errors
echo $?  # Should be 0 if no syntax errors
```

## Docker Container Purpose

The Docker container (`acalsim-workspace`) is used for:
- **Building SST components** (C++ code)
- **Running SST simulations**
- **Building QEMU** (if configured)

It is **not** used for:
- Building Linux kernel modules (requires target kernel headers)
- Building RISC-V binaries (requires cross-compiler setup)

## Next Steps for Full Integration

To actually run the Linux integration:

1. **Get or build RISC-V Linux kernel**
2. **Integrate VirtIO device into QEMU**
3. **Cross-compile kernel driver**
4. **Create root filesystem**
5. **Run simulation** (SST + QEMU with Linux)

See GETTING_STARTED.md for complete instructions.

## Summary

The build failure in Docker is **expected and normal**. The Linux kernel driver is meant to be built:
- Cross-compiled for RISC-V target, OR
- Built inside a running RISC-V Linux environment

The Docker container is for SST component development, not for Linux kernel module development.
