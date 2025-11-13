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

# Getting Started with QEMU-ACALSim-SST Baremetal

**Version**: 2.0 (N-Device Support)
**Last Updated**: 2025-11-10
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Single Device Setup](#single-device-setup)
4. [Two-Device Setup](#two-device-setup)
5. [N-Device Setup](#n-device-setup)
6. [Writing Baremetal Code](#writing-baremetal-code)
7. [Writing Assembly Functions](#writing-assembly-functions)
8. [Device Memory-Mapped I/O](#device-memory-mapped-io)
9. [Address Region Configuration](#address-region-configuration)
10. [Compiling and Running](#compiling-and-running)
11. [Multi-Server Deployment](#multi-server-deployment)
12. [Troubleshooting](#troubleshooting)

---

## Overview

QEMU-ACALSim-SST provides a co-simulation environment that integrates:
- **QEMU**: RISC-V processor emulator
- **SST**: Structural Simulation Toolkit for device modeling
- **Baremetal**: Direct hardware programming without OS

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ RISC-V Baremetal Program (C/C++/Assembly)              │
│   ├─ main.c: Application logic                          │
│   ├─ device_driver.c: MMIO operations                   │
│   └─ startup.S: Boot code                               │
└──────────────┬──────────────────────────────────────────┘
               │ MMIO Read/Write
┌──────────────▼──────────────────────────────────────────┐
│ QEMU (qemu-system-riscv32)                              │
│   ├─ sst-device #0 @ 0x10200000                        │
│   ├─ sst-device #1 @ 0x10300000                        │
│   └─ sst-device #N @ 0x10200000 + N*0x100000           │
└──────────────┬──────────────────────────────────────────┘
               │ Unix Sockets
┌──────────────▼──────────────────────────────────────────┐
│ SST QEMUBinaryComponent                                 │
│   ├─ N Socket Servers                                   │
│   ├─ Address-based Router                               │
│   └─ Device Links                                       │
└──────────────┬──────────────────────────────────────────┘
               │ SST Events (local or MPI)
┌──────────────▼──────────────────────────────────────────┐
│ SST Device Components                                   │
│   ├─ Echo Device: Simple data echo                      │
│   ├─ Compute Device: Arithmetic operations              │
│   ├─ Memory Device: Storage simulation                  │
│   └─ Custom Devices: Your implementations               │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

```bash
# Check installations
which qemu-system-riscv32  # QEMU RISC-V
which sst                   # SST simulator
riscv32-unknown-elf-gcc --version  # RISC-V toolchain

# If missing, see Installation section
```

### 30-Second Test

```bash
cd qemu-binary

# Build components
make clean && make && make install

# Compile test program
cd ../riscv-programs
make hello.elf

# Run simulation
cd ../qemu-binary
sst test_single_device.py
```

**Expected Output:**
```
QEMU-SST Single Device Test
Device initialization complete
Hello from RISC-V baremetal!
Simulation complete
```

---

## Single Device Setup

### Step 1: Create SST Configuration

**File**: `test_single_device.py`

```python
#!/usr/bin/env python3
import sst

# QEMU Component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "1GHz",
    "verbose": 2,
    "binary_path": "/path/to/your_program.elf",
    "qemu_path": "qemu-system-riscv32",
    "num_devices": 1,                    # Single device mode
    "device0_base": "0x10200000",        # Device address
    "device0_size": 4096,                # 4KB region
    "device0_name": "my_device"
})

# Device Component
device = sst.Component("my_device", "acalsim.QEMUDevice")
device.addParams({
    "clock": "1GHz",
    "verbose": 2,
    "echo_latency": 10  # 10 cycle latency
})

# Connect via SST Link
link = sst.Link("qemu_device_link")
link.connect(
    (qemu, "device_port_0", "1ns"),
    (device, "cpu_port", "1ns")
)

# Run for 10 seconds simulation time
sst.setProgramOption("stop-at", "10s")
```

### Step 2: Write Baremetal Program

**File**: `riscv-programs/hello_device.c`

```c
#include <stdint.h>

// Device memory-mapped registers
#define DEVICE_BASE     0x10200000
#define DEVICE_DATA     (*(volatile uint32_t*)(DEVICE_BASE + 0x00))
#define DEVICE_STATUS   (*(volatile uint32_t*)(DEVICE_BASE + 0x04))

int main(void) {
    // Write to device
    DEVICE_DATA = 0x12345678;

    // Wait for response
    while ((DEVICE_STATUS & 0x1) == 0);  // Wait for ready bit

    // Read response
    uint32_t response = DEVICE_DATA;

    return 0;
}
```

### Step 3: Compile and Run

```bash
# Compile
cd riscv-programs
make hello_device.elf

# Run
cd ../qemu-binary
sst test_single_device.py
```

---

## Two-Device Setup

### SST Configuration

**File**: `test_2device.py`

```python
#!/usr/bin/env python3
import sst

# QEMU Component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu.addParams({
    "clock": "1GHz",
    "verbose": 2,
    "binary_path": "/path/to/two_device_test.elf",
    "qemu_path": "qemu-system-riscv32",
    "num_devices": 2,                    # Two devices!
    "device0_base": "0x10200000",
    "device0_size": 4096,
    "device0_name": "echo_device",
    "device1_base": "0x10300000",        # Different address!
    "device1_size": 4096,
    "device1_name": "compute_device"
})

# Device 0: Echo Device
echo_dev = sst.Component("echo_device", "acalsim.QEMUDevice")
echo_dev.addParams({
    "clock": "1GHz",
    "verbose": 2,
    "echo_latency": 10
})

# Device 1: Compute Device
compute_dev = sst.Component("compute_device", "acalsim.ComputeDevice")
compute_dev.addParams({
    "clock": "1GHz",
    "verbose": 2,
    "compute_latency": 100
})

# Links
link0 = sst.Link("link_0")
link0.connect((qemu, "device_port_0", "1ns"), (echo_dev, "cpu_port", "1ns"))

link1 = sst.Link("link_1")
link1.connect((qemu, "device_port_1", "1ns"), (compute_dev, "cpu_port", "1ns"))

sst.setProgramOption("stop-at", "10s")
```

### Baremetal Program for Two Devices

**File**: `riscv-programs/two_device_test.c`

```c
#include <stdint.h>

// Device 0: Echo device
#define ECHO_BASE       0x10200000
#define ECHO_DATA       (*(volatile uint32_t*)(ECHO_BASE + 0x00))
#define ECHO_STATUS     (*(volatile uint32_t*)(ECHO_BASE + 0x04))

// Device 1: Compute device
#define COMPUTE_BASE    0x10300000
#define COMPUTE_OP_A    (*(volatile uint32_t*)(COMPUTE_BASE + 0x00))
#define COMPUTE_OP_B    (*(volatile uint32_t*)(COMPUTE_BASE + 0x04))
#define COMPUTE_RESULT  (*(volatile uint32_t*)(COMPUTE_BASE + 0x08))
#define COMPUTE_CMD     (*(volatile uint32_t*)(COMPUTE_BASE + 0x0C))

#define CMD_ADD         0x1
#define CMD_MUL         0x2

int main(void) {
    // Test echo device
    ECHO_DATA = 0xABCD;
    while ((ECHO_STATUS & 0x1) == 0);
    uint32_t echo_result = ECHO_DATA;

    // Test compute device
    COMPUTE_OP_A = 10;
    COMPUTE_OP_B = 20;
    COMPUTE_CMD = CMD_ADD;
    while ((COMPUTE_CMD & 0x80000000) == 0);  // Wait for done
    uint32_t sum = COMPUTE_RESULT;            // Should be 30

    COMPUTE_OP_A = 5;
    COMPUTE_OP_B = 6;
    COMPUTE_CMD = CMD_MUL;
    while ((COMPUTE_CMD & 0x80000000) == 0);
    uint32_t product = COMPUTE_RESULT;        // Should be 30

    return 0;
}
```

---

## N-Device Setup

### Scalable Configuration

**File**: `test_N_device.py`

```python
#!/usr/bin/env python3
import sst

# Configuration
NUM_DEVICES = 8
BASE_ADDR = 0x10200000
ADDR_STRIDE = 0x100000  # 1MB spacing

# QEMU Component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
params = {
    "clock": "1GHz",
    "verbose": 2,
    "binary_path": "/path/to/n_device_test.elf",
    "qemu_path": "qemu-system-riscv32",
    "num_devices": NUM_DEVICES
}

# Add device parameters
for i in range(NUM_DEVICES):
    addr = BASE_ADDR + (i * ADDR_STRIDE)
    params[f"device{i}_base"] = f"0x{addr:x}"
    params[f"device{i}_size"] = 4096
    params[f"device{i}_name"] = f"device_{i}"

qemu.addParams(params)

# Create N devices
devices = []
for i in range(NUM_DEVICES):
    dev = sst.Component(f"device_{i}", "acalsim.QEMUDevice")
    dev.addParams({
        "clock": "1GHz",
        "verbose": 2,
        "echo_latency": 10 * (i + 1)
    })
    devices.append(dev)

    # Link to QEMU
    link = sst.Link(f"link_{i}")
    link.connect(
        (qemu, f"device_port_{i}", "1ns"),
        (dev, "cpu_port", "1ns")
    )

sst.setProgramOption("stop-at", "10s")
```

### N-Device Baremetal Program

**File**: `riscv-programs/n_device_test.c`

```c
#include <stdint.h>

#define NUM_DEVICES     8
#define BASE_ADDR       0x10200000
#define ADDR_STRIDE     0x100000
#define REG_DATA        0x00
#define REG_STATUS      0x04

// Helper function to access device
static inline void device_write(int dev_id, uint32_t reg, uint32_t value) {
    volatile uint32_t *addr = (volatile uint32_t*)(BASE_ADDR + dev_id * ADDR_STRIDE + reg);
    *addr = value;
}

static inline uint32_t device_read(int dev_id, uint32_t reg) {
    volatile uint32_t *addr = (volatile uint32_t*)(BASE_ADDR + dev_id * ADDR_STRIDE + reg);
    return *addr;
}

static inline void device_wait_ready(int dev_id) {
    while ((device_read(dev_id, REG_STATUS) & 0x1) == 0);
}

int main(void) {
    // Test all devices in sequence
    for (int i = 0; i < NUM_DEVICES; i++) {
        // Write unique data to each device
        device_write(i, REG_DATA, 0x1000 + i);

        // Wait for response
        device_wait_ready(i);

        // Read back
        uint32_t response = device_read(i, REG_DATA);
    }

    // Test all devices in parallel (write then read)
    for (int i = 0; i < NUM_DEVICES; i++) {
        device_write(i, REG_DATA, 0x2000 + i);
    }

    for (int i = 0; i < NUM_DEVICES; i++) {
        device_wait_ready(i);
        uint32_t response = device_read(i, REG_DATA);
    }

    return 0;
}
```

---

## Writing Baremetal Code

### Project Structure

```
riscv-programs/
├── Makefile              # Build system
├── linker.ld            # Memory layout
├── startup.S            # Boot code
├── main.c               # Your application
├── device_driver.c      # Device abstraction
├── device_driver.h      # Device API
└── assembly_funcs.S     # Assembly functions
```

### Main Application Template

**File**: `main.c`

```c
#include <stdint.h>
#include "device_driver.h"

// Test function (can be called from tests)
int test_device_echo(void) {
    uint32_t test_data = 0x12345678;

    device_write(0, REG_DATA, test_data);
    device_wait(0);
    uint32_t result = device_read(0, REG_DATA);

    return (result == test_data) ? 0 : -1;  // 0 = pass
}

int main(void) {
    // Initialize devices
    for (int i = 0; i < NUM_DEVICES; i++) {
        device_init(i);
    }

    // Run tests
    int result = test_device_echo();

    // Your application logic here
    while (1) {
        // Main loop
    }

    return 0;
}
```

### Device Driver Abstraction

**File**: `device_driver.h`

```c
#ifndef DEVICE_DRIVER_H
#define DEVICE_DRIVER_H

#include <stdint.h>

// Configuration
#define NUM_DEVICES     4
#define BASE_ADDR       0x10200000
#define ADDR_STRIDE     0x100000

// Register offsets
#define REG_DATA        0x00
#define REG_STATUS      0x04
#define REG_CONTROL     0x08
#define REG_COMMAND     0x0C

// Status bits
#define STATUS_READY    0x1
#define STATUS_ERROR    0x2
#define STATUS_DONE     0x80000000

// API
void device_init(int dev_id);
void device_write(int dev_id, uint32_t reg, uint32_t value);
uint32_t device_read(int dev_id, uint32_t reg);
void device_wait(int dev_id);
int device_error(int dev_id);

#endif
```

**File**: `device_driver.c`

```c
#include "device_driver.h"

static volatile uint32_t* device_base(int dev_id) {
    return (volatile uint32_t*)(BASE_ADDR + dev_id * ADDR_STRIDE);
}

void device_init(int dev_id) {
    volatile uint32_t *base = device_base(dev_id);
    base[REG_CONTROL / 4] = 0x1;  // Enable device
}

void device_write(int dev_id, uint32_t reg, uint32_t value) {
    volatile uint32_t *base = device_base(dev_id);
    base[reg / 4] = value;
}

uint32_t device_read(int dev_id, uint32_t reg) {
    volatile uint32_t *base = device_base(dev_id);
    return base[reg / 4];
}

void device_wait(int dev_id) {
    while ((device_read(dev_id, REG_STATUS) & STATUS_READY) == 0);
}

int device_error(int dev_id) {
    return (device_read(dev_id, REG_STATUS) & STATUS_ERROR) != 0;
}
```

---

## Writing Assembly Functions

### Standalone Assembly File

**File**: `assembly_funcs.S`

```assembly
.section .text
.global asm_add
.global asm_multiply
.global asm_device_write
.global asm_device_read

# int asm_add(int a, int b)
asm_add:
    add a0, a0, a1      # a0 = a0 + a1
    ret

# int asm_multiply(int a, int b)
asm_multiply:
    mul a0, a0, a1      # a0 = a0 * a1
    ret

# void asm_device_write(uint32_t addr, uint32_t value)
# a0 = address, a1 = value
asm_device_write:
    sw a1, 0(a0)        # Store word: *addr = value
    ret

# uint32_t asm_device_read(uint32_t addr)
# a0 = address, returns value
asm_device_read:
    lw a0, 0(a0)        # Load word: return *addr
    ret

# Advanced: Atomic operations
.global asm_atomic_swap
# uint32_t asm_atomic_swap(uint32_t *addr, uint32_t new_val)
asm_atomic_swap:
    amoswap.w a0, a1, (a0)  # Atomic swap
    ret

# Memory barrier
.global asm_memory_fence
asm_memory_fence:
    fence rw, rw
    ret
```

### Calling Assembly from C

**File**: `main.c`

```c
#include <stdint.h>

// Declare assembly functions
extern int asm_add(int a, int b);
extern int asm_multiply(int a, int b);
extern void asm_device_write(uint32_t addr, uint32_t value);
extern uint32_t asm_device_read(uint32_t addr);
extern uint32_t asm_atomic_swap(uint32_t *addr, uint32_t new_val);
extern void asm_memory_fence(void);

#define DEVICE0_DATA    0x10200000

int main(void) {
    // Use assembly functions
    int sum = asm_add(10, 20);              // sum = 30
    int product = asm_multiply(5, 6);        // product = 30

    // Direct device access via assembly
    asm_device_write(DEVICE0_DATA, 0x1234);
    uint32_t result = asm_device_read(DEVICE0_DATA);

    // Atomic operations
    uint32_t device_reg = DEVICE0_DATA;
    uint32_t old_val = asm_atomic_swap((uint32_t*)device_reg, 0xABCD);

    // Ensure memory ordering
    asm_memory_fence();

    return 0;
}
```

### Makefile Integration

```makefile
# In riscv-programs/Makefile

# Source files
C_SRCS = main.c device_driver.c
ASM_SRCS = startup.S assembly_funcs.S

# Object files
C_OBJS = $(C_SRCS:.c=.o)
ASM_OBJS = $(ASM_SRCS:.S=.o)

# Compile assembly
%.o: %.S
	$(CC) $(CFLAGS) -c $< -o $@

# Link
program.elf: $(C_OBJS) $(ASM_OBJS)
	$(CC) $(LDFLAGS) -T linker.ld $^ -o $@
```

---

## Device Memory-Mapped I/O

### Memory Map Overview

```
Address Range          | Device | Purpose
-----------------------|--------|------------------
0x10200000-0x102FFFFF  | Dev 0  | Echo device
0x10300000-0x103FFFFF  | Dev 1  | Compute device
0x10400000-0x104FFFFF  | Dev 2  | Memory device
...
0x10200000+N*0x100000  | Dev N  | Custom device
```

### Register Layout (Per Device)

```
Offset | Register      | Access | Description
-------|---------------|--------|-------------------------
0x00   | DATA_IN       | W      | Write data to device
0x04   | DATA_OUT      | R      | Read data from device
0x08   | STATUS        | R      | Device status flags
0x0C   | CONTROL       | R/W    | Control register
0x10   | COMMAND       | W      | Command register
0x14   | ERROR         | R      | Error code
0x18   | CONFIG        | R/W    | Configuration
0x1C   | VERSION       | R      | Device version
```

### MMIO Examples

```c
#include <stdint.h>

// Define device base addresses
#define DEVICE0_BASE    0x10200000
#define DEVICE1_BASE    0x10300000

// Define registers for device 0
#define DEV0_DATA_IN    (*(volatile uint32_t*)(DEVICE0_BASE + 0x00))
#define DEV0_DATA_OUT   (*(volatile uint32_t*)(DEVICE0_BASE + 0x04))
#define DEV0_STATUS     (*(volatile uint32_t*)(DEVICE0_BASE + 0x08))
#define DEV0_CONTROL    (*(volatile uint32_t*)(DEVICE0_BASE + 0x0C))

// Example 1: Simple write/read
void example_basic_io(void) {
    DEV0_DATA_IN = 0x12345678;          // Write to device
    while (!(DEV0_STATUS & 0x1));       // Wait for ready
    uint32_t result = DEV0_DATA_OUT;    // Read result
}

// Example 2: Burst write
void example_burst_write(uint32_t *data, int count) {
    for (int i = 0; i < count; i++) {
        DEV0_DATA_IN = data[i];
        while (!(DEV0_STATUS & 0x1));   // Wait between writes
    }
}

// Example 3: DMA-style operation
void example_dma_transfer(uint32_t src_addr, uint32_t dst_addr, int size) {
    // Configure DMA (device-specific)
    volatile uint32_t *dev = (volatile uint32_t*)DEVICE1_BASE;
    dev[0] = src_addr;    // Source
    dev[1] = dst_addr;    // Destination
    dev[2] = size;        // Size
    dev[3] = 0x1;         // Start DMA

    // Wait for completion
    while (!(dev[4] & 0x80000000));  // Check done bit
}

// Example 4: Interrupt-style (polling)
void example_wait_interrupt(void) {
    DEV0_CONTROL = 0x100;  // Enable interrupt flag

    // Do other work...

    // Poll interrupt status
    while (!(DEV0_STATUS & 0x100));  // Wait for interrupt flag

    // Handle "interrupt"
    uint32_t result = DEV0_DATA_OUT;

    // Clear flag
    DEV0_STATUS = 0x100;
}
```

---

## Address Region Configuration

### Configuring Device Addresses

**In SST Python Config:**

```python
# Method 1: Sequential addressing
BASE = 0x10200000
STRIDE = 0x100000

for i in range(num_devices):
    addr = BASE + i * STRIDE
    qemu.addParam(f"device{i}_base", f"0x{addr:x}")
    qemu.addParam(f"device{i}_size", 4096)
```

**In Linker Script (`linker.ld`):**

```ld
MEMORY
{
    /* RAM for code and data */
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M

    /* Device regions (reserved, not allocated) */
    DEVICE0 : ORIGIN = 0x10200000, LENGTH = 1M
    DEVICE1 : ORIGIN = 0x10300000, LENGTH = 1M
    DEVICE2 : ORIGIN = 0x10400000, LENGTH = 1M
}

SECTIONS
{
    .text : { *(.text*) } > RAM
    .data : { *(.data*) } > RAM
    .bss  : { *(.bss*) } > RAM

    /* Device regions are accessed via pointers, not allocated */
}
```

### Address Space Layout

```
0x00000000 - 0x0FFFFFFF : Reserved
0x10000000 - 0x1FFFFFFF : Device MMIO space
  ├─ 0x10200000 : Device 0 (1MB)
  ├─ 0x10300000 : Device 1 (1MB)
  ├─ 0x10400000 : Device 2 (1MB)
  └─ ...
0x80000000 - 0x8FFFFFFF : Main RAM (code + data)
0xC0000000 - 0xFFFFFFFF : Reserved
```

### Validation in C

```c
#include <stdint.h>

// Compile-time address validation
#define DEVICE0_BASE    0x10200000
#define DEVICE1_BASE    0x10300000

// Ensure no overlap (compile-time assertion)
_Static_assert(DEVICE1_BASE >= DEVICE0_BASE + 0x100000,
               "Device address regions overlap!");

// Runtime validation
int validate_device_address(uint32_t addr, int dev_id) {
    uint32_t expected = 0x10200000 + dev_id * 0x100000;
    return (addr == expected);
}
```

---

## Compiling and Running

### Single Server Execution

#### Step 1: Build Components

```bash
# Build QEMU Binary component
cd qemu-binary
export SST_CORE_HOME=/path/to/sst-install
export PATH=$SST_CORE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SST_CORE_HOME/lib/sstcore:$LD_LIBRARY_PATH

make clean && make && make install
```

#### Step 2: Compile RISC-V Program

```bash
cd riscv-programs

# Edit Makefile if needed
make clean
make your_program.elf

# Verify ELF
riscv32-unknown-elf-readelf -h your_program.elf
```

#### Step 3: Run Simulation

```bash
cd ../qemu-binary

# Run with verbose output
sst --verbose test_config.py

# Run in background
sst test_config.py > output.log 2>&1 &

# With timeout
timeout 60s sst test_config.py
```

---

## Multi-Server Deployment

### Prerequisites for Distributed Execution

```bash
# On all servers:
# 1. Install MPI
sudo apt-get install -y openmpi-bin libopenmpi-dev

# 2. Verify connectivity
ping other_server_ip

# 3. Setup SSH keys
ssh-keygen
ssh-copy-id user@other_server

# 4. Sync project files
rsync -avz project/ user@other_server:~/project/
```

### Hostfile Configuration

**File**: `mpi_hosts`

```
# Format: hostname slots=N
192.168.100.178 slots=4 max_slots=8
192.168.100.69  slots=4 max_slots=8
```

### Launch Distributed Simulation

```bash
# 2 servers, 5 ranks total
mpirun -np 5 \
  --hostfile mpi_hosts \
  -x SST_CORE_HOME \
  -x PATH \
  -x LD_LIBRARY_PATH \
  --mca btl tcp,self \
  sst test_distributed.py

# Explicit host specification
mpirun -np 5 \
  -H 192.168.100.178:2,192.168.100.69:3 \
  sst test_config.py

# With rank display
mpirun -np 5 \
  --hostfile mpi_hosts \
  --display-map \
  sst test_config.py
```

### Monitoring Distributed Execution

```bash
# Watch processes on each server
# On server 1:
watch -n 1 "ps aux | grep sst"

# On server 2:
ssh user@192.168.100.69 "watch -n 1 'ps aux | grep sst'"

# Network traffic
iftop -i eth0
```

---

## Troubleshooting

### Issue: QEMU Not Found

```bash
# Check QEMU path
which qemu-system-riscv32

# Update path in config:
qemu.addParam("qemu_path", "/usr/local/bin/qemu-system-riscv32")
```

### Issue: Device Not Responding

```c
// Add timeout to device wait
int device_wait_timeout(int dev_id, int timeout_ms) {
    int count = 0;
    while ((device_read(dev_id, REG_STATUS) & STATUS_READY) == 0) {
        if (count++ > timeout_ms * 1000) {
            return -1;  // Timeout
        }
    }
    return 0;  // Success
}
```

### Issue: MPI Connection Failed

```bash
# Test MPI connectivity
mpirun -np 2 -H server1,server2 hostname

# Check firewall
sudo ufw status
sudo ufw allow from 192.168.100.0/24
```

### Issue: SST Component Not Found

```bash
# Verify installation
sst-info qemubinary
sst-info acalsim

# Check library path
ls $SST_CORE_HOME/lib/sstcore/libqemubinary.so
```

### Debug Output

```python
# In SST config, increase verbosity:
qemu.addParam("verbose", 3)  # 0-3, higher = more verbose
device.addParam("verbose", 3)
```

---

## Next Steps

- Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for architecture details
- See [DEMO_EXAMPLE.md](DEMO_EXAMPLE.md) for complete working example
- Check [API_REFERENCE.md](API_REFERENCE.md) for device APIs
- Browse `examples/` directory for more examples

---

**Questions?** Check the FAQ or file an issue on GitHub.
