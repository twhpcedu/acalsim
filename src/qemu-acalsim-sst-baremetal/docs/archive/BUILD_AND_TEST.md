<!--
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Phase 2C: Build and Test Guide

Complete instructions for building and testing the QEMU-SST Binary MMIO integration.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Building Components](#building-components)
4. [Building Custom QEMU](#building-custom-qemu)
5. [Running Tests](#running-tests)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools
- **RISC-V Toolchain**: `riscv64-unknown-elf-gcc`
- **SST-Core**: Installed and in PATH
- **QEMU Build Dependencies**: ninja, pkg-config, libglib2.0-dev, libpixman-1-dev
- **C++ Compiler**: g++ or clang with C++17 support

### Environment Setup

```bash
# Set SST environment
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

# Verify SST installation
sst --version
sst-config --prefix
```

### Docker Environment (Recommended)

If using the acalsim-workspace Docker container:

```bash
docker exec -it acalsim-workspace bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal
```

---

## Quick Start

For a complete build and test from scratch:

```bash
# Navigate to project directory
cd /path/to/acalsim/src/qemu-acalsim-sst-baremetal

# 1. Build RISC-V test program
cd riscv-programs
make clean
make mmio_test.elf
cd ..

# 2. Build SST components
cd qemu-binary
make clean && make && make install
cd ..

cd acalsim-device
make clean && make && make install
cd ..

# 3. Verify installation
sst-info qemubinary
sst-info acalsim

# 4. Run Phase 2C test
sst qemu_binary_test.py
```

Expected output: All 4 tests should PASS.

---

## Building Components

### 1. Build RISC-V Test Program

The MMIO test program runs on QEMU and communicates with SST via binary MMIO protocol.

```bash
cd riscv-programs

# Build all test programs
make

# Or build just the Phase 2C test
make mmio_test.elf

# Verify the binary
ls -lh mmio_test.elf
# Should show ~8KB ELF binary
```

**Output files:**
- `mmio_test.elf` - RISC-V executable
- `mmio_test.bin` - Raw binary
- `mmio_test.dump` - Disassembly listing

**What it does:**
- Tests MMIO read/write operations at 0x10200000
- Performs echo tests (writes data, reads it back)
- Tests STATUS and CONTROL registers
- Reports results via UART console

### 2. Build QEMUBinary Component

SST component that launches QEMU and handles binary MMIO protocol.

```bash
cd qemu-binary

# Clean previous build
make clean

# Build component
make

# Install to SST
make install

# Verify installation
sst-info qemubinary
```

**Expected output:**
```
Component: QEMUBinary
  Description: QEMU subprocess with binary MMIO protocol (Phase 2C)
  Parameters:
    - clock
    - verbose
    - binary_path
    - qemu_path
    - socket_path
    - device_base
```

### 3. Build ACALSim Device Component

SST device component that responds to MMIO transactions.

```bash
cd acalsim-device

# Build and install
make clean
make
make install

# Verify installation
sst-info acalsim
```

**Expected output:**
```
Component: QEMUDevice
  Description: ACALSim device for QEMU integration
  Parameters:
    - clock
    - base_addr
    - size
    - verbose
    - echo_latency
```

### 4. Verify All Components

```bash
# Check component libraries exist
ls -lh $SST_CORE_HOME/lib/sstcore/libqemubinary.so
ls -lh $SST_CORE_HOME/lib/sstcore/libacalsim.so

# List all installed components
sst-info qemubinary
sst-info acalsim
```

---

## Building Custom QEMU

Phase 2C requires a custom QEMU build with the SST device integrated. If you already completed Phase 2C.3, this is done. Otherwise:

### Option 1: Use Pre-built Custom QEMU

If you completed Phase 2C.3 integration:

```bash
# Custom QEMU should be at:
/home/user/qemu-build/install/bin/qemu-system-riscv32

# Verify it has SST device
/home/user/qemu-build/install/bin/qemu-system-riscv32 -M virt,help 2>&1 | grep -i sst
```

Set environment variable:

```bash
export QEMU_PATH=/home/user/qemu-build/install/bin/qemu-system-riscv32
```

### Option 2: Build Custom QEMU from Scratch

See `PHASE2C_INTEGRATION.md` for complete integration instructions.

**Summary:**
1. Clone QEMU v6.2.0
2. Copy `qemu-sst-device/sst-device.c` to `qemu/hw/misc/`
3. Update `hw/misc/meson.build` and `hw/misc/Kconfig`
4. Modify `hw/riscv/virt.c` to add SST device at 0x10200000
5. Configure and build QEMU
6. Install custom QEMU binary

**Quick rebuild (if already configured):**

```bash
cd /home/user/qemu-build/qemu/build
ninja
ninja install
```

---

## Running Tests

### Test 1: SST Component Test (Phase 2C Integration)

**Full end-to-end test with QEMU and SST:**

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal

# Ensure custom QEMU path is set
export QEMU_PATH=/home/user/qemu-build/install/bin/qemu-system-riscv32

# Run SST simulation
sst qemu_binary_test.py
```

**Expected output:**

```
====================================================================
QEMU-Binary SST Integration Test - Phase 2C
====================================================================
Binary:      /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal/riscv-programs/mmio_test.elf
QEMU:        /home/user/qemu-build/install/bin/qemu-system-riscv32
Socket:      /tmp/qemu-sst-mmio.sock
Device base: 0x10200000
Sim time:    1000 us
====================================================================

[QEMUBinaryComponent] Creating socket server...
[QEMUBinaryComponent] Socket server ready at /tmp/qemu-sst-mmio.sock
[QEMUBinaryComponent] Launching QEMU subprocess...
[QEMUBinaryComponent] QEMU PID: 12345

===========================================
  QEMU-SST Phase 2C: Binary MMIO Protocol
===========================================

Device base address: 0x10200000

[TEST 1] Simple write/read
  Writing 0xDEADBEEF to SST_DATA_IN
  Triggering operation
  Waiting for completion
  Read result: 0xDEADBEEF
  [PASS] Echo test passed

[TEST 2] Multiple transactions
  Transaction 1: 0x12345678 ... PASS
  Transaction 2: 0xCAFEBABE ... PASS
  Transaction 3: 0xDEADC0DE ... PASS
  Transaction 4: 0x0BADF00D ... PASS
  Transaction 5: 0x1337BEEF ... PASS
  Result: 5/5 passed
  [PASS] All transactions passed

[TEST 3] Status register
  Reading initial status
  Initial status: 0x00000000
  Status bits:
    BUSY:       0
    DATA_READY: 0
    ERROR:      0
  [PASS] Status register readable

[TEST 4] Control register
  Testing RESET bit
  Reset issued
  [PASS] Control register writable

===========================================
  All tests complete!
===========================================

Simulation is complete, simulated time: 1 ms

Device Statistics:
  Total Loads:  6
  Total Stores: 6
  Total Echos:  6
```

**Result:** ✅ All 4 tests should PASS

### Test 2: QEMU Standalone Test

**Test RISC-V program in QEMU without SST (for debugging):**

```bash
cd riscv-programs

# Run with standard QEMU (will timeout waiting for SST)
make test-mmio

# Or manually:
qemu-system-riscv32 -M virt -bios none -nographic -kernel mmio_test.elf
```

Press `Ctrl-A` then `X` to exit QEMU.

**Expected behavior:**
- Program starts and prints header
- Tests timeout (no SST device to respond)
- This confirms RISC-V program works independently

### Test 3: Custom QEMU Device Test

**Verify SST device is in custom QEMU:**

```bash
# Check if device exists
$QEMU_PATH -device help 2>&1 | grep -i sst

# Expected output:
# name "sst-device", bus System

# Check memory map
$QEMU_PATH -M virt -monitor stdio -nographic
(qemu) info mtree
# Should show sst-device at 0x10200000
```

---

## Test Configuration

### qemu_binary_test.py Parameters

Key configuration parameters in the SST test script:

```python
# QEMU binary (use custom build)
qemu_path = os.environ.get("QEMU_PATH",
    "/home/user/qemu-build/install/bin/qemu-system-riscv32")

# RISC-V test program
binary_path = os.environ.get("RISCV_BINARY",
    "/home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal/riscv-programs/mmio_test.elf")

# Unix socket for communication
socket_path = "/tmp/qemu-sst-mmio.sock"

# SST device address (must match QEMU virt.c and mmio_test.c)
device_base = 0x10200000

# Simulation time
sim_time_us = 1000  # 1ms
```

### Environment Variables

```bash
# Override QEMU path
export QEMU_PATH=/path/to/custom/qemu-system-riscv32

# Override test binary
export RISCV_BINARY=/path/to/custom/test.elf

# SST verbosity
export SST_VERBOSE=1
```

---

## Troubleshooting

### Issue: sst-info shows component not found

**Symptom:**
```
sst-info: No element library 'qemubinary' found
```

**Solution:**
```bash
# Rebuild and reinstall
cd qemu-binary
make clean
make
make install

# Check installation
ls -l $SST_CORE_HOME/lib/sstcore/libqemubinary.so
```

### Issue: RISC-V toolchain not found

**Symptom:**
```
make: riscv64-unknown-elf-gcc: Command not found
```

**Solution:**
```bash
# Install RISC-V toolchain
# Ubuntu/Debian:
sudo apt-get install gcc-riscv64-unknown-elf

# Or build from source:
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv32imac --with-abi=ilp32
make
export PATH=/opt/riscv/bin:$PATH
```

### Issue: QEMU device not found

**Symptom:**
```
qemu-system-riscv32: Device 'sst-device' not found
```

**Solution:**
1. Verify you're using custom QEMU build:
   ```bash
   which qemu-system-riscv32
   # Should be /home/user/qemu-build/install/bin/qemu-system-riscv32
   ```

2. Rebuild QEMU with SST device:
   ```bash
   cd /home/user/qemu-build/qemu/build
   rm -rf *
   ../configure --target-list=riscv32-softmmu --enable-debug
   make -j$(nproc)
   make install
   ```

### Issue: Socket connection timeout

**Symptom:**
```
[QEMUBinaryComponent] Waiting for QEMU to connect...
[QEMUBinaryComponent] Timeout waiting for connection
```

**Solution:**
1. Check socket path permissions:
   ```bash
   ls -la /tmp/qemu-sst-mmio.sock
   ```

2. Verify QEMU has SST device:
   ```bash
   $QEMU_PATH -device help | grep sst
   ```

3. Check device is at correct address:
   ```bash
   grep SST_DEVICE_BASE riscv-programs/mmio_test.c
   # Should be 0x10200000

   grep device_base qemu_binary_test.py
   # Should be 0x10200000
   ```

### Issue: Test hangs or times out

**Symptom:**
```
[TEST 1] Simple write/read
  Writing 0xDEADBEEF to SST_DATA_IN
  Triggering operation
  Waiting for completion
  [TIMEOUT] Device did not respond
```

**Solution:**
1. Check all addresses match:
   - `mmio_test.c`: `SST_DEVICE_BASE = 0x10200000`
   - `qemu_binary_test.py`: `device_base = 0x10200000`
   - `virt.c`: `memmap[VIRT_SST_DEVICE] = { 0x10200000, 0x1000 }`

2. Check QEMU device is properly initialized:
   ```bash
   # Add debug output to virt.c and rebuild
   # Or run with QEMU monitor to inspect device state
   ```

3. Increase simulation time:
   ```python
   # In qemu_binary_test.py:
   sim_time_us = 10000  # 10ms instead of 1ms
   ```

### Issue: Build errors in SST components

**Symptom:**
```
error: SST/Component.h: No such file or directory
```

**Solution:**
```bash
# Verify SST environment
sst-config --prefix
sst-config --ELEMENT_CXXFLAGS

# Ensure SST_CORE_HOME is set
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
```

---

## Performance Validation

### Expected Performance

Phase 2C binary MMIO protocol should provide:

| Metric | Phase 2B (Serial) | Phase 2C (MMIO) | Improvement |
|--------|------------------|----------------|-------------|
| Throughput | ~1,000 tx/sec | ~10,000 tx/sec | 10x |
| Latency | ~1ms/tx | ~100μs/tx | 10x |
| CPU Usage | High (parsing) | Low (binary) | ~9x reduction |
| Protocol Overhead | ~80% | ~8% | 10x reduction |

### Measuring Performance

Add timing instrumentation to test:

```bash
# Run with increased verbosity
cd qemu-binary
# Edit QEMUBinaryComponent.cc to add timing measurements
make clean && make && make install

# Run test
time sst qemu_binary_test.py
```

Check device statistics in output:
```
Device Statistics:
  Total Loads:  6    # Memory read operations
  Total Stores: 6    # Memory write operations
  Total Echos:  6    # Successful echo transactions
```

---

## Next Steps

After successful Phase 2C testing:

1. **Performance Benchmarking**: Run extended tests with more transactions
2. **Multi-Transaction Tests**: Test with hundreds/thousands of MMIO operations
3. **Stress Testing**: Test with concurrent access, larger data transfers
4. **Phase 2D**: Multi-core support (multiple QEMU instances)
5. **Phase 3**: Cycle-accurate timing model
6. **Phase 4**: Full system simulation with Linux

---

## Summary

**Complete build and test flow:**

```bash
# 1. Environment setup
export SST_CORE_HOME=/home/user/projects/acalsim/sst-core/sst-core-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH
export QEMU_PATH=/home/user/qemu-build/install/bin/qemu-system-riscv32

# 2. Build all components
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-baremetal
cd riscv-programs && make clean && make mmio_test.elf && cd ..
cd qemu-binary && make clean && make && make install && cd ..
cd acalsim-device && make clean && make && make install && cd ..

# 3. Verify installation
sst-info qemubinary
sst-info acalsim

# 4. Run test
sst qemu_binary_test.py

# Expected: All 4 tests PASS
```

**Success criteria:**
- ✅ All components build without errors
- ✅ Components install to SST successfully
- ✅ sst-info shows both components
- ✅ All 4 integration tests PASS
- ✅ Device statistics show correct load/store/echo counts

For detailed integration instructions, see `PHASE2C_INTEGRATION.md`.
