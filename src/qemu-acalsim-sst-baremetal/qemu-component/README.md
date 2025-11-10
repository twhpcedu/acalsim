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

# QEMU SST Component

This directory contains the QEMU wrapper component for the QEMU-ACALSim distributed SST simulation.

## Overview

The QEMUComponent is an SST component that simulates a test program running on QEMU RISC-V. It sends load/store transaction requests to the ACALSim device component via SST Links.

**Note**: This is a simplified simulation of QEMU behavior for demonstration purposes. In a full implementation, this would integrate with actual QEMU RISC-V emulator.

## Component Features

- **Test Program Simulation**: Implements a simple test that exercises the echo device
- **Memory Transaction Generation**: Sends load/store requests to device
- **Response Handling**: Receives and processes device responses
- **Request/Response Matching**: Tracks pending transactions by request ID
- **Statistics Tracking**: Counts loads, stores, successes, and failures

## Test Program Behavior

The component simulates a test program that:

1. **Write Phase**: Writes test pattern to device DATA_IN register (0x10000000)
2. **Poll Phase**: Polls device STATUS register (0x10000008) until ready
3. **Read Phase**: Reads echoed data from DATA_OUT register (0x10000004)
4. **Verify Phase**: Compares read data with written pattern
5. **Repeat**: Runs multiple iterations with varying patterns

## Building

### Prerequisites

- SST-Core installed with `sst-config` in PATH
- C++17 compatible compiler
- Make build system

### Build Commands

```bash
# Build the component
make

# Install to SST element library
make install

# Clean build artifacts
make clean

# Remove from SST
make uninstall
```

### Verification

After installation, verify the component is available:

```bash
sst-info qemu
```

You should see output showing the `RISCV` component.

## Usage in SST Configuration

```python
import sst

# Create QEMU component
qemu = sst.Component("qemu0", "qemu.RISCV")
qemu.addParams({
    "clock": "1GHz",                # QEMU clock frequency
    "device_base": "0x10000000",    # Device base address
    "device_size": "4096",          # Device size (4KB)
    "verbose": "1",                 # Verbosity level (0-3)
    "test_pattern": "0xDEADBEEF",   # Test data pattern
    "num_iterations": "5"           # Number of test iterations
})

# Connect to device component
link = sst.Link("qemu_device_link")
link.connect((qemu, "device_port", "1ns"),
             (device, "cpu_port", "1ns"))
```

## SST Event Interface

### Output Events (to Device)

**MemoryTransactionEvent**:
- `type`: LOAD or STORE
- `address`: Memory address (0x10000000 - 0x10000FFF)
- `data`: Data value (for STORE)
- `size`: Transaction size (1, 2, or 4 bytes)
- `req_id`: Unique request ID

### Input Events (from Device)

**MemoryResponseEvent**:
- `req_id`: Request ID (matches transaction)
- `data`: Response data (for LOAD)
- `success`: Transaction success status

## State Machine

The test program uses a state machine with the following states:

```
IDLE → WRITE_DATA → WAIT_BUSY → READ_STATUS → READ_DATA → VERIFY
   ↑                                                           ↓
   └───────────────────────────────────────────────────────────┘
                    (repeat for num_iterations)
                              ↓
                            DONE
```

## Example Output

```
QEMU[0:0]: Initializing QEMU Component
QEMU[0:0]: Configuration:
QEMU[0:0]:   Clock: 1GHz
QEMU[0:0]:   Device Base: 0x10000000
QEMU[0:0]:   Test Pattern: 0xDEADBEEF
QEMU[0:0]:   Iterations: 5

QEMU[0:0]: === Starting Test Iteration 1 ===
QEMU[0:0]: Writing pattern 0xDEADBEEF to DATA_IN
QEMU[0:0]: Issuing STORE: addr=0x10000000 data=0xDEADBEEF
QEMU[0:0]: Reading STATUS register
QEMU[0:0]: Device ready, reading DATA_OUT
QEMU[0:0]: ✓ Test iteration 1 PASSED (read=0xDEADBEEF)

...

QEMU[0:0]: === All Test Iterations Complete ===
QEMU[0:0]: Test Results:
QEMU[0:0]:   Iterations:    5
QEMU[0:0]:   Successes:     5
QEMU[0:0]:   Failures:      0
QEMU[0:0]:   Total Loads:   15
QEMU[0:0]:   Total Stores:  5

QEMU[0:0]: *** TEST PASSED ***
```

## Implementation Details

### Transaction Tracking

Each transaction is assigned a unique request ID and tracked in `pending_transactions_` map until the response is received. This allows proper matching of responses to requests.

### Response Processing

Responses are queued in `response_queue_` and processed at the beginning of each clock cycle. This ensures deterministic behavior and proper synchronization.

### Test Pattern Variation

Each iteration uses a slightly different test pattern:
- Iteration 1: `test_pattern + 0`
- Iteration 2: `test_pattern + 1`
- Iteration 3: `test_pattern + 2`
- etc.

This helps verify that the device correctly handles different data values.

## Files

- **QEMUComponent.hh**: Component header with event definitions
- **QEMUComponent.cc**: Component implementation
- **Makefile**: Build system
- **README.md**: This file

## Testing

See `../config/echo_device.py` for the SST configuration that connects this component to the ACALSim device.

## Future Extensions

For full QEMU integration, this component would need to:
- Integrate with QEMU's memory API
- Handle actual RISC-V instruction execution
- Support interrupts and exceptions
- Implement DMA support
- Add checkpoint/restore capability

## References

- [QEMU Documentation](https://qemu.readthedocs.io/)
- [SST-Core Documentation](http://sst-simulator.org/)
- [QEMU-ACALSim Architecture](../ARCHITECTURE.md)
