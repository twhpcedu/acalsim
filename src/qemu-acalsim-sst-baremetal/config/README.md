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

# QEMU-ACALSim SST Configuration Files

This directory contains SST Python configuration files for running distributed QEMU-ACALSim simulations.

## Files

### echo_device.py

Multi-process distributed simulation configuration for the echo device example.

**Components**:
- **Rank 0**: QEMU component (simulates test program)
- **Rank 1**: ACALSim device component (echo device)

**Communication**: Components communicate via SST Link with IPC (MPI)

## Prerequisites

Before running the simulation, ensure:

1. **SST-Core is installed** with MPI support:
   ```bash
   sst --version
   mpirun --version
   ```

2. **Components are built and installed**:
   ```bash
   # Build and install QEMU component
   cd ../qemu-component
   make && make install

   # Build and install ACALSim device
   cd ../acalsim-device
   make && make install
   ```

3. **Verify components are registered**:
   ```bash
   sst-info qemu
   sst-info acalsim
   ```

## Running the Simulation

### Single-Node Multi-Process

Run both processes on the same machine:

```bash
mpirun -n 2 sst echo_device.py
```

### Multi-Node Distributed

Run processes on different nodes (requires MPI cluster):

```bash
# Start QEMU on node0, device on node1
mpirun -n 1 -host node0 sst echo_device.py : \
       -n 1 -host node1 sst echo_device.py
```

### With Verbose Output

Increase verbosity for debugging:

```bash
# Modify QEMU_VERBOSE and DEVICE_VERBOSE in echo_device.py
# Then run:
mpirun -n 2 sst echo_device.py
```

Verbosity levels:
- `0`: Quiet (errors only)
- `1`: Normal (important events)
- `2`: Verbose (detailed transactions)
- `3`: Debug (everything)

## Configuration Parameters

Edit `echo_device.py` to customize simulation parameters:

### Clock and Timing

```python
CLOCK_FREQ = "1GHz"        # Component clock frequency
LINK_LATENCY = "1ns"       # Inter-process communication latency
ECHO_LATENCY = "10"        # Device echo operation latency (cycles)
```

### Device Memory Map

```python
DEVICE_BASE_ADDR = "0x10000000"  # Base address
DEVICE_SIZE = "4096"              # Size (4KB)
```

### Test Parameters

```python
TEST_PATTERN = "0xDEADBEEF"  # Initial test pattern
NUM_ITERATIONS = "5"         # Number of test iterations
```

### Verbosity

```python
QEMU_VERBOSE = "1"    # QEMU component verbosity
DEVICE_VERBOSE = "1"  # Device component verbosity
```

## Expected Output

### Successful Run

```
SST Configuration: Rank 0 of 2
Rank 0: Creating QEMU component

======================================================================
QEMU-ACALSim Distributed Simulation Configuration
======================================================================
Clock Frequency:      1GHz
Device Base Address:  0x10000000
Device Size:          4096 bytes
Link Latency:         1ns
Test Pattern:         0xDEADBEEF
Test Iterations:      5

Process Distribution:
  Rank 0: QEMU component (test program)
  Rank 1: ACALSim device (echo device)
======================================================================

SST Configuration: Rank 1 of 2
Rank 1: Creating ACALSim device component

QEMU[0:0]: Initializing QEMU Component
ACALSimDevice[1:0]: Initializing ACALSim Device Component

QEMU[0:0]: === Starting Test Iteration 1 ===
QEMU[0:0]: Writing pattern 0xDEADBEEF to DATA_IN
ACALSimDevice[1:0]: Received STORE request: addr=0x10000000 data=0xDEADBEEF
ACALSimDevice[1:0]: Starting echo operation (will complete at cycle 10)
QEMU[0:0]: Device ready, reading DATA_OUT
QEMU[0:0]: ✓ Test iteration 1 PASSED (read=0xDEADBEEF)

...

QEMU[0:0]: === All Test Iterations Complete ===
QEMU[0:0]: Test Results:
QEMU[0:0]:   Successes:     5
QEMU[0:0]:   Failures:      0

QEMU[0:0]: *** TEST PASSED ***
```

## Troubleshooting

### Component Not Found

```
Error: Cannot find component "qemu.RISCV"
```

**Solution**: Ensure components are installed:
```bash
cd ../qemu-component && make install
cd ../acalsim-device && make install
sst-info qemu
sst-info acalsim
```

### MPI Errors

```
Error: This simulation requires exactly 2 MPI ranks
```

**Solution**: Always run with exactly 2 MPI ranks:
```bash
mpirun -n 2 sst echo_device.py
```

### Link Connection Errors

```
Error: Failed to connect link
```

**Solution**: Check that both components define compatible ports:
- QEMU: `device_port`
- Device: `cpu_port`

### No Output

If simulation runs but produces no output:
1. Increase verbosity in configuration
2. Check SST is built with `--enable-debug`
3. Run with SST verbose flag: `sst --verbose echo_device.py`

## Advanced Usage

### Statistics Collection

Enable detailed statistics:

```python
# In echo_device.py
sst.enableAllStatisticsForAllComponents()
sst.setStatisticOutput("sst.statOutputCSV")
sst.setStatisticOutputOptions({"filepath": "./stats.csv"})
```

### Custom Test Patterns

Modify test patterns to stress different scenarios:

```python
# Test with sequential patterns
TEST_PATTERN = "0x00000000"

# Test with all 1s
TEST_PATTERN = "0xFFFFFFFF"

# Test with alternating bits
TEST_PATTERN = "0xAAAAAAAA"
```

### Extended Iterations

Run longer tests:

```python
NUM_ITERATIONS = "1000"
```

## Performance Tuning

### Reduce Link Latency

For faster simulation (less realistic):
```python
LINK_LATENCY = "0ns"
```

### Increase Clock Frequency

Run at higher frequency:
```python
CLOCK_FREQ = "2GHz"
```

## See Also

- [QEMU Component README](../qemu-component/README.md)
- [ACALSim Device README](../acalsim-device/README.md)
- [Architecture Documentation](../ARCHITECTURE.md)
- [Quick Start Guide](../QUICKSTART.md)
