<!--
Copyright 2023-2026 Playlab/ACAL

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

# SST Device for QEMU - Phase 2C

## Overview

This directory contains the QEMU device implementation that provides binary MMIO interface for SST integration.

## Architecture

```
RISC-V Program        QEMU SST Device       SST Component
     |                      |                      |
     |-- MMIO Write ------->|                      |
     |  (0x20000000)        |                      |
     |                      |-- Binary Request --->|
     |                      |   {type, addr,       |
     |                      |    data, size}       |
     |                      |                      |
     |                      |<-- Binary Response --|
     |<-- MMIO Complete ----|   {success, data}    |
```

## Device Registers

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x00 | DATA_IN | R/W | Data to send to SST |
| 0x04 | DATA_OUT | R | Data received from SST |
| 0x08 | STATUS | R | Device status |
| 0x0C | CONTROL | W | Control register |

### STATUS Register Bits

- Bit 0: BUSY - Transaction in progress
- Bit 1: DATA_READY - Response data available
- Bit 2: ERROR - Error occurred

### CONTROL Register Bits

- Bit 0: START - Start transaction (send DATA_IN to SST)
- Bit 1: RESET - Reset device

## Building

### Prerequisites

1. QEMU source code (version 7.0+)
2. QEMU development headers
3. C compiler (gcc/clang)

### Option 1: Build as QEMU Device Plugin (Recommended)

This approach builds the device as a loadable module that QEMU can load at runtime.

```bash
# Set QEMU source directory
export QEMU_SRC=/path/to/qemu/source

# Build the device plugin
make
```

### Option 2: Build into QEMU

For better integration, you can build the device directly into QEMU:

1. Copy `sst-device.c` to `qemu/hw/misc/`
2. Add to `qemu/hw/misc/meson.build`:
   ```
   softmmu_ss.add(when: 'CONFIG_SST_DEVICE', if_true: files('sst-device.c'))
   ```
3. Add to `qemu/hw/misc/Kconfig`:
   ```
   config SST_DEVICE
       bool
       default y if RISCV
   ```
4. Rebuild QEMU:
   ```bash
   cd qemu
   ./configure --target-list=riscv32-softmmu
   make
   ```

## Usage

### With QEMU

Launch QEMU with the SST device:

```bash
qemu-system-riscv32 \
    -M virt \
    -bios none \
    -nographic \
    -kernel test.elf \
    -device sst-device,socket=/tmp/qemu-sst-mmio.sock
```

The device will be mapped at address 0x20000000 on the virt machine.

### With SST

Start SST first so the socket server is ready:

```bash
# Terminal 1: Start SST simulation
sst qemu_binary_test.py

# Terminal 2: QEMU will connect when launched
# (QEMU command from above)
```

## Programming Interface

### C Example

```c
#define SST_DEVICE_BASE  0x20000000
#define SST_DATA_IN      (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x00))
#define SST_DATA_OUT     (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x04))
#define SST_STATUS       (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x08))
#define SST_CONTROL      (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x0C))

// Write data to SST
SST_DATA_IN = 0xDEADBEEF;
SST_CONTROL = 1;  // START

// Wait for completion
while (SST_STATUS & 1);  // BUSY

// Read response
uint32_t result = SST_DATA_OUT;
```

## Protocol

The device communicates with SST using binary protocol over Unix socket:

### MMIORequest (QEMU → SST)
```c
struct MMIORequest {
    uint8_t  type;        // 0 = READ, 1 = WRITE
    uint8_t  size;        // 1, 2, 4, or 8 bytes
    uint16_t reserved;
    uint64_t addr;        // MMIO address
    uint64_t data;        // Write data
} __attribute__((packed));
```

### MMIOResponse (SST → QEMU)
```c
struct MMIOResponse {
    uint8_t  success;     // 0 = error, 1 = success
    uint8_t  reserved[7];
    uint64_t data;        // Read data
} __attribute__((packed));
```

## Testing

### Standalone Test (QEMU only)

Test the device without SST connection:

```bash
# Create a simple test socket server
nc -lU /tmp/qemu-sst-mmio.sock &

# Launch QEMU
qemu-system-riscv32 -M virt -device sst-device,socket=/tmp/qemu-sst-mmio.sock ...
```

### Full Integration Test

```bash
# Build components
cd ../qemu-binary && make && make install
cd ../riscv-programs && make mmio_test.elf

# Run SST simulation
cd ..
sst qemu_binary_test.py
```

## Troubleshooting

### Device Not Found

If QEMU reports "Device 'sst-device' not found":
- Check that the device module is built correctly
- Verify QEMU can find the device library
- Try building the device into QEMU directly

### Socket Connection Failed

If device logs "Failed to connect to SST socket":
- Ensure SST is running first and listening on the socket
- Check socket path matches between QEMU and SST config
- Verify socket file permissions

### MMIO Reads Return 0

If all MMIO reads return 0:
- Check that device is mapped at correct address (0x20000000)
- Verify socket connection is established
- Enable QEMU logging: `-d guest_errors`

## Performance

Expected throughput with binary MMIO protocol:
- ~10,000 transactions/second
- ~100μs latency per transaction
- ~10x improvement over Phase 2B serial protocol

## References

- [QEMU Device Model](https://qemu.readthedocs.io/en/latest/devel/qom.html)
- [QEMU Memory API](https://qemu.readthedocs.io/en/latest/devel/memory.html)
- [RISC-V QEMU](https://qemu.readthedocs.io/en/latest/system/target-riscv.html)
