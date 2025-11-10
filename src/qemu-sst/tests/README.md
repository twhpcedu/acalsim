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

# QEMU-ACALSim Test Programs

This directory contains test programs for the QEMU-ACALSim distributed simulation.

## Overview

These are reference implementations showing how RISC-V test programs would interact with the memory-mapped ACALSim device. In the current implementation, the QEMU SST component simulates this test behavior directly.

**For full QEMU integration**, these programs would be:
1. Compiled for RISC-V target architecture
2. Loaded into QEMU memory
3. Executed by QEMU RISC-V CPU
4. Device accesses forwarded to ACALSim via SST

## Test Programs

### echo_test.c

Simple echo device test that:
- Writes test patterns to device DATA_IN register
- Waits for device to complete echo operation
- Reads echoed data from DATA_OUT register
- Verifies correctness
- Runs multiple iterations

**Register Map**:
```
0x10000000  DATA_IN   (W)   Write data to trigger echo
0x10000004  DATA_OUT  (R)   Read echoed data
0x10000008  STATUS    (R)   Device status (busy/ready)
0x1000000C  CONTROL   (RW)  Control register (reset)
```

## Current Implementation Status

### Phase 1: SST Simulation (Current)

The QEMU component simulates the test program behavior:
- State machine implements test logic in `QEMUComponent::runTestProgram()`
- Generates load/store transactions
- Verifies responses
- Reports success/failure

### Phase 2: Full QEMU Integration (Future)

For real QEMU integration:
1. Compile test programs for RISC-V
2. Load binary into QEMU memory
3. QEMU executes instructions
4. Memory-mapped device accesses trapped
5. Converted to SST events
6. Sent to ACALSim device

## Building (Future)

When RISC-V toolchain is available:

```bash
# Install RISC-V toolchain
sudo apt-get install gcc-riscv64-unknown-elf

# Build test programs
make build

# This will create:
#   echo_test.elf  - ELF executable
#   echo_test.bin  - Raw binary
#   echo_test.asm  - Disassembly listing
```

## Usage

### Current Usage (SST Simulation)

The test behavior is already implemented in the QEMU component:

```bash
# Build and install components
cd ../qemu-component && make && make install
cd ../acalsim-device && make && make install

# Run simulation
cd ../config
mpirun -n 2 sst echo_device.py
```

### Future Usage (Real QEMU)

When QEMU integration is complete:

```bash
# Build test program
cd tests
make build

# Create SST config that loads test into QEMU
# Update qemu component parameters:
#   "binary": "tests/echo_test.bin"
#   "entry_point": "0x80000000"

# Run simulation with real QEMU
mpirun -n 2 sst echo_device.py
```

## Test Output

### Expected Behavior

```
==============================================
Echo Device Test Program
==============================================

Resetting device...
Iteration 1: Testing pattern 0xDEADBEEF... PASSED
Iteration 2: Testing pattern 0xDEADBEF0... PASSED
Iteration 3: Testing pattern 0xDEADBEF1... PASSED
Iteration 4: Testing pattern 0xDEADBEF2... PASSED
Iteration 5: Testing pattern 0xDEADBEF3... PASSED

==============================================
Test Summary:
  Total:     5
  Successes: 5
  Failures:  0
==============================================

*** ALL TESTS PASSED ***
```

## Test Program Structure

### Main Functions

**`main()`**: Test driver
- Initializes device
- Runs multiple test iterations
- Reports results

**`test_echo(uint32_t pattern)`**: Single echo test
- Writes pattern to DATA_IN
- Waits for device ready
- Reads from DATA_OUT
- Verifies result

**`wait_device_ready()`**: Polling function
- Reads STATUS register
- Checks for DATA_READY bit
- Implements timeout

**`reset_device()`**: Reset function
- Writes CONTROL register
- Clears device state

### Register Access

```c
// Write to register
volatile uint32_t* reg = (volatile uint32_t*)REG_DATA_IN;
*reg = 0xDEADBEEF;

// Read from register
volatile uint32_t* reg = (volatile uint32_t*)REG_STATUS;
uint32_t value = *reg;
```

## Adding New Tests

To add a new test program:

1. Create new `.c` file in this directory
2. Follow the same structure as `echo_test.c`
3. Use device register definitions
4. Add to `TESTS` variable in Makefile
5. Update this README

Example:

```c
// my_test.c
#include <stdint.h>

#define DEVICE_BASE 0x10000000
#define REG_DATA_IN (DEVICE_BASE + 0x00)

int main(void) {
    volatile uint32_t* data_in = (volatile uint32_t*)REG_DATA_IN;
    *data_in = 0x12345678;
    // ... test logic ...
    return 0;
}
```

## Debugging

### Compile-time Debugging

```bash
# Compile with debug info
make CFLAGS+="-g -O0" build

# Generate assembly listing
riscv64-unknown-elf-objdump -d echo_test.elf > echo_test.asm
```

### Runtime Debugging

For SST simulation (current):
- Increase verbosity in QEMU component
- Add logging in `QEMUComponent::runTestProgram()`

For real QEMU (future):
- Use QEMU GDB stub: `qemu-system-riscv32 -s -S`
- Attach GDB: `riscv64-unknown-elf-gdb echo_test.elf`
- Set breakpoints and step through

## Test Parameters

Modify these constants in `echo_test.c`:

```c
#define DEVICE_BASE     0x10000000  // Device base address
#define TEST_PATTERN    0xDEADBEEF  // Initial test pattern
#define NUM_ITERATIONS  5           // Number of iterations
#define MAX_WAIT_CYCLES 1000        // Timeout for device polling
```

## Performance Testing

For performance evaluation:

```c
// Add cycle counter (RISC-V performance counters)
uint64_t start_cycles = read_cycle_counter();
test_echo(pattern);
uint64_t end_cycles = read_cycle_counter();
uint64_t latency = end_cycles - start_cycles;
```

## See Also

- [QEMU Component](../qemu-component/README.md) - Simulates test behavior
- [ACALSim Device](../acalsim-device/README.md) - Device implementation
- [SST Configuration](../config/README.md) - Simulation setup
- [Architecture](../ARCHITECTURE.md) - System design

## References

- [RISC-V Specifications](https://riscv.org/technical/specifications/)
- [QEMU RISC-V](https://www.qemu.org/docs/master/system/target-riscv.html)
- [Memory-Mapped I/O](https://en.wikipedia.org/wiki/Memory-mapped_I/O)
