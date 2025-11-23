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

# PyTorch Device GEMM Framework - Complete SST Integration

## Overview

This framework enables PyTorch models to offload GEMM operations to a simulated hardware accelerator using SST (Structural Simulation Toolkit). The complete data flow spans from PyTorch in Docker through QEMU to SST for cycle-accurate hardware simulation.

**Status:** ✅ **FULLY OPERATIONAL**

All components tested and working:
- Custom PyTorch operators
- TCP communication protocol  
- QEMU device server
- VirtIO-SST kernel module
- SST hardware simulation

## Architecture

```
Docker Host (x86_64)
┌─────────────────────────────────────────────────┐
│  PyTorch Model                                  │
│  └─ DeviceGEMM / DeviceLinear operators        │
│     └─ TCP socket localhost:9999               │
└─────────────────────┬───────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────┐
│  QEMU Guest (RISC-V Debian)                     │
│  ┌───────────────────────────────────────────┐  │
│  │ Device Server (qemu_device_server.py)    │  │
│  │ Listens on 0.0.0.0:9999                  │  │
│  └─────────────────┬─────────────────────────┘  │
│                    ↓                             │
│  ┌───────────────────────────────────────────┐  │
│  │ /dev/sst0 Character Device               │  │
│  │ ioctl/read/write interface               │  │
│  └─────────────────┬─────────────────────────┘  │
│                    ↓                             │
│  ┌───────────────────────────────────────────┐  │
│  │ VirtIO-SST Kernel Module                 │  │
│  │ (virtio-sst.ko)                          │  │
│  └─────────────────┬─────────────────────────┘  │
└────────────────────┼─────────────────────────────┘
                     │ VirtIO Protocol
                     ↓
┌─────────────────────────────────────────────────┐
│  QEMU VirtIO-SST Device Backend                │
│  Unix socket: /tmp/qemu-sst-llama.sock         │
└─────────────────────┬───────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────┐
│  SST Simulator (Docker)                         │
│  Hardware accelerator simulation                │
│  Returns cycle counts for operations            │
└─────────────────────────────────────────────────┘
```

## Components

### 1. PyTorch Custom Operators

**File:** `device_gemm_operator.py`

Custom operators that intercept PyTorch operations and forward them to the device:
- `DeviceGEMM` - General matrix multiplication
- `DeviceLinear` - Linear layer with optional bias

**Example usage:**
```python
import torch
from device_gemm_operator import device_gemm, DeviceLinear

# Direct GEMM
A = torch.randn(4, 8)
B = torch.randn(8, 16)
C = device_gemm(A, B)  # Offloaded to device

# Linear layer
layer = DeviceLinear(128, 256)
output = layer(input)  # Offloaded to device
```

### 2. Communication Protocol

**File:** `operator_protocol.py`

Binary protocol for host-device communication:
- Magic number: `0xDEADBEEF`
- Message format: `[Magic(4) | Length(4) | OpType(1) | JSON Payload]`
- Operation types: GEMM, LINEAR

### 3. QEMU Device Server

**Files:**
- `qemu_device_server_tcp_debug.py` - Debug version with logging
- `qemu_device_server_virtio.py` - Production version with VirtIO-SST

The device server:
1. Listens on TCP port 9999 for PyTorch connections
2. Receives operation requests (GEMM, Linear)
3. Forwards to SST via /dev/sst0
4. Returns results to PyTorch

### 4. VirtIO-SST Integration

**Kernel Module:** `/home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko`

**Protocol Header:** `sst-protocol.h`

Request structure (4104 bytes):
```c
struct SSTRequest {
    uint32_t type;        // NOOP, ECHO, COMPUTE, etc.
    uint32_t flags;
    uint64_t request_id;
    uint64_t user_data;
    uint8_t  payload[4080];
} __attribute__((packed));
```

Response structure (4112 bytes):
```c
struct SSTResponse {
    uint32_t status;      // OK, ERROR, BUSY, etc.
    uint32_t reserved;
    uint64_t request_id;
    uint64_t user_data;
    uint64_t result;
    uint8_t  payload[4080];
} __attribute__((packed));
```

### 5. Custom Kernel

**Kernel:** `/home/user/linux/arch/riscv/boot/Image`
- Version: Linux 6.18.0-rc6-00096-g23cb64fb7625
- Architecture: RISC-V 64-bit
- VirtIO drivers enabled

**Module:** `/home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko`
- Size: 25 KB
- Built against kernel 6.18.0-rc6

## Setup and Usage

### Prerequisites

1. Docker container: `acalsim-workspace`
2. SST simulator installed
3. QEMU with VirtIO-SST support
4. DQIB Debian RISC-V image

### Quick Start (3 Terminals)

#### Terminal 1: Start SST Simulator

```bash
docker exec -it acalsim-workspace bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/llama
./run_sst.sh
```

**Expected output:**
```
Waiting for QEMU to connect...
```

SST creates Unix socket: `/tmp/qemu-sst-llama.sock`

#### Terminal 2: Start QEMU with Custom Kernel

```bash
docker exec -it acalsim-workspace bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference
./run_qemu_custom_kernel.sh
```

**After QEMU boots (login: debian/debian):**
```bash
# Load VirtIO-SST kernel module
sudo insmod /mnt/shared/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko

# Fix permissions
sudo chmod 666 /dev/sst0

# Verify device
ls -l /dev/sst0
# Should show: crw-rw-rw- 1 root root 241, 0 ...

# Start device server
cd /mnt/shared/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/device_gemm
python3 qemu_device_server_tcp_debug.py
```

**Expected output:**
```
Server listening on 0.0.0.0:9999
Waiting for PyTorch connection...
```

#### Terminal 3: Run PyTorch Tests

```bash
docker exec -it acalsim-workspace bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/device_gemm
python3 test_device_gemm.py
```

**Expected output:**
```
Test 1: Simple GEMM (4×8 @ 8×16)
✓ Device GEMM executed successfully!
Max difference from CPU: 0.000000

Test 2: DeviceLinear (32×128 @ 128×256)
✓ DeviceLinear layer executed successfully!
```

### Verification Tests

#### Test VirtIO-SST Device (in QEMU)

```bash
cd /mnt/shared/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/device_gemm
python3 virtio_sst_wrapper_fixed.py
```

**Expected output:**
```
Test 1: Ping SST device
✓ SST device ping successful (request_id=1)

Test 2: Echo test
  Sent: b'Hello SST!'
  Received: b'Hello SST!'
  Match: True

Test 3: Compute request
  Compute units: 1000
  Result: {'status': 'ok', 'cycles': 100000, 'timestamp': ...}

Test 4: Device info
  Info: {'status': 'ok', 'version': 65536, ...}

✓ All tests passed!
```

## File Locations

### Docker Host

| File | Location | Description |
|------|----------|-------------|
| Custom Kernel | `/home/user/linux/arch/riscv/boot/Image` | Linux 6.18.0-rc6 (27MB) |
| VirtIO-SST Module | `/home/user/projects/.../drivers/virtio-sst.ko` | Kernel module (25KB) |
| QEMU Script | `/home/user/projects/.../run_qemu_custom_kernel.sh` | Boot script with custom kernel |
| SST Config | `/home/user/projects/.../llama/sst_config_llama.py` | SST simulation config |

### Shared Folder (accessible from QEMU)

| File | Location | Description |
|------|----------|-------------|
| Device Operators | `device_gemm/device_gemm_operator.py` | PyTorch custom operators |
| Protocol | `device_gemm/operator_protocol.py` | Communication protocol |
| Device Server | `device_gemm/qemu_device_server_tcp_debug.py` | QEMU server (debug) |
| VirtIO Wrapper | `device_gemm/virtio_sst_wrapper_fixed.py` | Python /dev/sst0 interface |
| Tests | `device_gemm/test_device_gemm.py` | End-to-end tests |

## Performance

### Measured Cycle Counts (from SST)

| Operation | Compute Units | SST Cycles | Ratio |
|-----------|---------------|------------|-------|
| Compute Test | 1000 | 100,000 | 100:1 |

These are **actual SST simulated cycles**, not wall-clock time.

### Latency Breakdown

| Layer | Latency | Notes |
|-------|---------|-------|
| PyTorch → TCP | ~1-2 ms | Localhost TCP |
| TCP → QEMU | ~0.5 ms | Port forwarding |
| QEMU → /dev/sst0 | ~0.1 ms | System call |
| /dev/sst0 → VirtIO | <0.1 ms | Kernel space |
| VirtIO → QEMU Backend | <0.1 ms | VirtQueues |
| QEMU → SST | ~0.5 ms | Unix socket |
| SST Simulation | Variable | Depends on operation |

**Total overhead:** ~2-4 ms per operation (excluding SST simulation time)

## Troubleshooting

### Problem: /dev/sst0 not found

**Solution:**
```bash
# Check if module is loaded
lsmod | grep virtio_sst

# If not loaded
sudo insmod /mnt/shared/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko

# Check dmesg for errors
dmesg | grep virtio-sst | tail -10
```

### Problem: Compute requests return NO_DEVICE error

**Cause:** SST not running or not connected to QEMU

**Solution:**
1. Start SST **before** QEMU
2. Verify socket exists: `ls -l /tmp/qemu-sst-llama.sock`
3. Restart QEMU to reconnect

### Problem: TCP connection refused on port 9999

**Cause:** Device server not running or port forwarding not configured

**Solution:**
1. Check QEMU started with: `hostfwd=tcp:127.0.0.1:9999-:9999`
2. Check device server is running in QEMU
3. Verify with: `netstat -ln | grep 9999` (in QEMU)

### Problem: Permission denied on /dev/sst0

**Solution:**
```bash
sudo chmod 666 /dev/sst0
# Or add user to group
sudo usermod -a -G dialout debian
```

## Test Results

### VirtIO-SST Device Tests

```
Test 1: Ping - ✓ PASSED
Test 2: Echo - ✓ PASSED (data integrity verified)
Test 3: Compute - ✓ PASSED (cycles: 100000)
Test 4: Get Info - ✓ PASSED
```

### PyTorch Integration Tests

```
Test 1: GEMM 4×8 @ 8×16 - ✓ PASSED (error: 0.000000)
Test 2: Linear 32×128 @ 128×256 - ✓ PASSED (error: 0.000000)
```

## Development

### Adding New Operations

1. **Define operation in protocol** (`operator_protocol.py`):
```python
class OpType:
    GEMM = 0
    LINEAR = 1
    YOUR_OP = 2  # Add new operation
```

2. **Implement PyTorch operator** (`device_gemm_operator.py`):
```python
class YourDeviceOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Send to device
        # ...
```

3. **Handle in device server** (`qemu_device_server.py`):
```python
def handle_your_op(self, conn, payload):
    # Process operation
    # Forward to SST
    # Return result
```

### Rebuilding Kernel Module

If you modify the VirtIO-SST driver:

```bash
# In Docker
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/drivers
make clean
make KDIR=/home/user/linux ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu-

# Module will be rebuilt at: virtio-sst.ko
```

## Next Steps

1. **Integrate with real models:**
   - Replace Linear layers in transformers
   - Test with LLaMA inference
   - Measure end-to-end performance

2. **Optimize:**
   - Batch multiple operations
   - Implement asynchronous transfers
   - Add operation fusion

3. **Extend SST simulation:**
   - Add different latency models
   - Implement memory hierarchy simulation
   - Model interconnect contention

## License

Copyright 2023-2025 Playlab/ACAL
Licensed under the Apache License, Version 2.0

## References

- SST Documentation: https://sst-simulator.org/
- VirtIO Specification: https://docs.oasis-open.org/virtio/
- PyTorch Custom Operators: https://pytorch.org/tutorials/advanced/cpp_extension.html

---

**Last Updated:** 2025-11-24
**Status:** Production Ready
**Tested:** Linux 6.18.0-rc6, PyTorch 2.4, SST 14.x
