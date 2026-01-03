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

# Tutorial: How to Modify SST Component to Handle Custom GEMM Operator

**Author**: ACALSim Team
**Date**: 2025-11-24
**Version**: 1.0

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Testing](#testing)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This tutorial shows you how to add custom operator support to the SST component for cycle-accurate performance simulation of PyTorch operations.

**What you'll learn:**
- How the PyTorch → QEMU → SST data flow works
- How to extend the SST protocol with new request types
- How to implement custom performance models in SST
- How to integrate everything end-to-end

**Example use case:** Adding GEMM (General Matrix Multiply) operator to SST for simulating AI accelerator performance.

---

## Architecture

### Complete Data Flow

```
Layer 1: PyTorch Application (Docker)
         Custom operators: DeviceGEMM, DeviceLinear
         Location: device_gemm/device_gemm_operator.py
                         |
                         | TCP Socket (localhost:9999)
                         | Protocol: Magic + Length + OpType + JSON
                         v
Layer 2: QEMU Device Server (QEMU Debian RISC-V)
         Server: qemu_device_server_virtio.py
         Location: /mnt/shared/device_gemm/
                         |
                         | /dev/sst0 (VirtIO-SST character device)
                         | Protocol: struct SSTRequest/SSTResponse
                         v
Layer 3: VirtIO-SST Kernel Module (QEMU Linux)
         Module: virtio-sst.ko
         Location: drivers/sst-virtio.c
                         |
                         | VirtIO Protocol (virtqueues)
                         v
Layer 4: QEMU VirtIO Device (QEMU Process)
         Device: virtio-sst-device
         Location: virtio-device/virtio-sst.c
                         |
                         | Unix Domain Socket
                         | Path: /tmp/qemu-sst-llama.sock
                         v
Layer 5: SST Component (SST Simulator)
         Component: ACALSimVirtIODeviceComponent
         Location: acalsim-device/*.cc
         <- THIS IS WHERE YOU IMPLEMENT CUSTOM LOGIC
```

**Key Insight:** The SST component at Layer 5 is where you implement your custom performance models. Everything above is just plumbing to get data there.

---

## Prerequisites

Before starting, ensure you have:

1. **Working PyTorch Device GEMM framework**
   - Completed setup from `docs/sst-integration/pytorch-device-gemm.md`
   - Tests passing with CPU fallback

2. **Build environment**
   - SST-Core installed
   - C++ compiler with C++14 support
   - Python 3.x

3. **Files you'll modify**
   ```
   src/qemu-acalsim-sst-linux/
   ├── virtio-device/
   │   └── sst-protocol.h                    # Protocol definitions
   ├── acalsim-device/
   │   ├── ACALSimVirtIODeviceComponent.hh   # Component header
   │   ├── ACALSimVirtIODeviceComponent.cc   # Component implementation
   │   └── Makefile                          # Build configuration
   └── examples/llama-inference/device_gemm/
       └── qemu_device_server_virtio.py      # QEMU-side server
   ```

---

## Step-by-Step Implementation

### Step 1: Define the Protocol

First, we need to add GEMM to the SST protocol.

**File:** `src/qemu-acalsim-sst-linux/virtio-device/sst-protocol.h`

#### 1.1: Add Request Type

```c
/*
 * Request Types
 */
enum SSTRequestType {
    SST_REQ_NOOP      = 0,  // No operation (test connectivity)
    SST_REQ_ECHO      = 1,  // Echo request (returns same data)
    SST_REQ_COMPUTE   = 2,  // Compute request (SST simulation)
    SST_REQ_READ      = 3,  // Read from SST device memory
    SST_REQ_WRITE     = 4,  // Write to SST device memory
    SST_REQ_RESET     = 5,  // Reset device state
    SST_REQ_GET_INFO  = 6,  // Get device information
    SST_REQ_CONFIGURE = 7,  // Configure device parameters
    SST_REQ_GEMM      = 8,  // GEMM operation (NEW!)
};
```

#### 1.2: Add GEMM Payload to Request Structure

```c
struct SSTRequest {
    uint32_t type;        // Request type (enum SSTRequestType)
    uint32_t flags;       // Request flags (reserved)
    uint64_t request_id;  // Unique request identifier
    uint64_t user_data;   // User-defined data (opaque to device)

    union {
        // Existing payloads...
        struct {
            uint64_t compute_units;
            uint32_t latency_model;
            uint32_t reserved;
        } compute;

        struct {
            uint64_t addr;
            uint32_t size;
            uint32_t reserved;
        } memory;

        // NEW: GEMM request payload
        struct {
            uint32_t M;           // Matrix A rows
            uint32_t N;           // Matrix B cols
            uint32_t K;           // Matrix A cols / B rows
            uint32_t dtype_size;  // Size of data type (4 for float32)
            uint64_t A_offset;    // Offset to matrix A data in payload
            uint64_t B_offset;    // Offset to matrix B data in payload
        } gemm;

        // Generic data buffer
        uint8_t data[SST_MAX_DATA_SIZE];
    } payload;
} __attribute__((packed));
```

**Explanation:**
- `M, N, K`: Standard GEMM dimensions (C[M,N] = A[M,K] × B[K,N])
- `dtype_size`: Allows supporting different data types (FP32, FP16, INT8)
- `A_offset, B_offset`: Where matrix data starts in the `data[]` array

#### 1.3: Add GEMM Response Payload

```c
struct SSTResponse {
    uint32_t status;      // Response status
    uint32_t reserved;
    uint64_t request_id;  // Matching request_id
    uint64_t user_data;   // Echo of user_data
    uint64_t result;      // Operation result (cycles for GEMM)

    union {
        // Existing payloads...
        struct {
            uint64_t cycles;
            uint64_t timestamp;
        } compute;

        struct {
            uint32_t version;
            uint32_t capabilities;
            uint64_t max_compute_units;
            uint64_t memory_size;
        } info;

        // NEW: GEMM response
        struct {
            uint64_t cycles;         // Total cycles for GEMM
            uint64_t compute_cycles; // Compute-bound cycles
            uint64_t memory_cycles;  // Memory-bound cycles
            uint64_t memory_bytes;   // Total memory traffic
            uint32_t result_offset;  // Offset to result matrix (optional)
            uint32_t reserved;
        } gemm;

        uint8_t data[SST_MAX_DATA_SIZE];
    } payload;
} __attribute__((packed));
```

**Explanation:**
- `cycles`: Total simulated cycles (max of compute/memory)
- `compute_cycles`: Cycles if compute-bound
- `memory_cycles`: Cycles if memory-bound
- `memory_bytes`: For analyzing memory bottlenecks

#### 1.4: Add Helper Function (Optional)

```c
static inline const char* sst_request_type_str(uint32_t type) {
    switch (type) {
        case SST_REQ_NOOP: return "NOOP";
        case SST_REQ_ECHO: return "ECHO";
        case SST_REQ_COMPUTE: return "COMPUTE";
        case SST_REQ_READ: return "READ";
        case SST_REQ_WRITE: return "WRITE";
        case SST_REQ_RESET: return "RESET";
        case SST_REQ_GET_INFO: return "GET_INFO";
        case SST_REQ_CONFIGURE: return "CONFIGURE";
        case SST_REQ_GEMM: return "GEMM";  // NEW!
        default: return "UNKNOWN";
    }
}
```

---

### Step 2: Implement SST Component Logic

Now we implement the actual performance model in the SST component.

**File:** `src/qemu-acalsim-sst-linux/acalsim-device/ACALSimVirtIODeviceComponent.cc`

#### 2.1: Add GEMM Case to processRequest()

Find the `processRequest()` function (around line 200) and add:

```cpp
void ACALSimVirtIODeviceComponent::processRequest(const uint8_t* data, size_t len) {
    if (len < sizeof(struct SSTRequest)) {
        out_.verbose(CALL_INFO, 2, 0, "Invalid request size: %zu\n", len);
        return;
    }

    const struct SSTRequest* req = (const struct SSTRequest*)data;
    struct SSTResponse       resp;
    memset(&resp, 0, sizeof(resp));

    // Echo request metadata
    resp.request_id = req->request_id;
    resp.user_data  = req->user_data;

    total_requests_++;

    out_.verbose(CALL_INFO, 2, 0, "Processing request type %u (%s)\n",
                req->type, sst_request_type_str(req->type));

    switch (req->type) {
        case SST_REQ_NOOP:
            // ... existing code ...
            break;

        case SST_REQ_ECHO:
            // ... existing code ...
            break;

        case SST_REQ_COMPUTE:
            // ... existing code ...
            break;

        // NEW: GEMM operation
        case SST_REQ_GEMM: {
            gemm_requests_++;  // Update statistics

            // Extract GEMM parameters
            uint32_t M = req->payload.gemm.M;
            uint32_t N = req->payload.gemm.N;
            uint32_t K = req->payload.gemm.K;
            uint32_t dtype_size = req->payload.gemm.dtype_size;

            out_.verbose(CALL_INFO, 2, 0,
                        "GEMM request: C[%u,%u] = A[%u,%u] × B[%u,%u], dtype_size=%u\n",
                        M, N, M, K, K, N, dtype_size);

            // Calculate FLOPs and memory traffic
            uint64_t flops = 2ULL * M * N * K;  // Multiply-add = 2 ops
            uint64_t memory_bytes = (M * K + K * N + M * N) * dtype_size;

            // Performance model parameters (example: A100-like accelerator)
            // Adjust these for your target hardware!
            const double PEAK_FLOPS_PER_CYCLE = 64.0;   // 64 FP32 ops/cycle
            const double MEMORY_BW_GB_S = 900.0;        // HBM2e bandwidth
            const double FREQ_GHZ = 1.41;               // Clock frequency
            const double BYTES_PER_CYCLE = (MEMORY_BW_GB_S / FREQ_GHZ);

            // Roofline model: max(compute_bound, memory_bound)
            uint64_t compute_cycles = (uint64_t)(flops / PEAK_FLOPS_PER_CYCLE);
            uint64_t memory_cycles = (uint64_t)(memory_bytes / BYTES_PER_CYCLE);
            uint64_t total_cycles = std::max(compute_cycles, memory_cycles);

            // Populate response
            resp.status = SST_STATUS_OK;
            resp.result = total_cycles;
            resp.payload.gemm.cycles = total_cycles;
            resp.payload.gemm.compute_cycles = compute_cycles;
            resp.payload.gemm.memory_cycles = memory_cycles;
            resp.payload.gemm.memory_bytes = memory_bytes;
            resp.payload.gemm.result_offset = 0;  // Not returning actual matrix

            // Log performance
            out_.verbose(CALL_INFO, 1, 0,
                        "GEMM simulated: %lu FLOPs, %lu total cycles "
                        "(compute: %lu, memory: %lu, %lu bytes)\n",
                        flops, total_cycles, compute_cycles, memory_cycles, memory_bytes);

            // Determine bottleneck
            const char* bottleneck = (compute_cycles > memory_cycles) ? "COMPUTE" : "MEMORY";
            out_.verbose(CALL_INFO, 1, 0, "Bottleneck: %s\n", bottleneck);

            break;
        }

        default:
            out_.verbose(CALL_INFO, 1, 0, "Unknown request type: %u\n", req->type);
            resp.status = SST_STATUS_INVALID;
            resp.result = 0;
            break;
    }

    // Send response
    sendResponse((const uint8_t*)&resp, sizeof(resp));
}
```

#### 2.2: Add Statistics Tracking

**File:** `src/qemu-acalsim-sst-linux/acalsim-device/ACALSimVirtIODeviceComponent.hh`

Add to private members:

```cpp
private:
    // Statistics
    SST::Cycle_t current_cycle_;
    uint64_t     total_requests_;
    uint64_t     noop_requests_;
    uint64_t     echo_requests_;
    uint64_t     compute_requests_;
    uint64_t     gemm_requests_;      // NEW!
```

Update constructor and `finish()` method to track GEMM stats (see full code in component).

---

### Step 3: Update VirtIO Wrapper (Python)

**File:** `examples/llama-inference/device_gemm/virtio_sst_wrapper.py`

Add GEMM request type constant:

```python
class SSTRequestType:
    NOOP = 0
    ECHO = 1
    COMPUTE = 2
    READ = 3
    WRITE = 4
    RESET = 5
    GET_INFO = 6
    CONFIGURE = 7
    GEMM = 8  # NEW!
```

Add GEMM helper method to `VirtIOSST` class:

```python
def gemm(self, M, N, K, A_data, B_data, dtype_size=4):
    """
    Send GEMM request to SST

    Args:
        M, N, K: GEMM dimensions
        A_data: Matrix A as bytes (M × K elements)
        B_data: Matrix B as bytes (K × N elements)
        dtype_size: Size of each element (4 for float32)

    Returns:
        dict with 'status', 'cycles', 'memory_bytes', etc.
    """
    req = SSTRequest(SSTRequestType.GEMM)

    # Pack GEMM metadata into payload
    gemm_metadata = struct.pack('<IIIIQQ',
        M, N, K, dtype_size,
        24,  # A_offset: after metadata (6 fields × 4 bytes = 24 bytes)
        24 + len(A_data)  # B_offset: after metadata + A
    )

    # Combine metadata + matrices
    payload = gemm_metadata + A_data + B_data
    req.payload = payload.ljust(SST_MAX_DATA_SIZE, b'\x00')

    # Send request
    resp = self.send_request(req)

    if resp.status == SSTStatus.OK:
        # Parse GEMM response from payload
        if len(resp.payload) >= 32:
            cycles, compute_cycles, memory_cycles, memory_bytes = \
                struct.unpack('<QQQQ', resp.payload[:32])

            return {
                'status': 'ok',
                'cycles': cycles,
                'compute_cycles': compute_cycles,
                'memory_cycles': memory_cycles,
                'memory_bytes': memory_bytes
            }
        else:
            return {'status': 'ok', 'cycles': resp.result}
    else:
        return {'status': 'error', 'code': resp.status}
```

---

### Step 4: Build and Install

```bash
# In Docker container
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/acalsim-device

# Clean previous build
make clean

# Build the component
make

# Install to SST
sudo make install

# Verify installation
sst-info acalsim.VirtIODevice
```

Expected output should show the component with GEMM support.

---

## Testing

### Test 1: Verify SST Component

```bash
# Terminal 1: Start SST
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/llama
./run_sst.sh
```

### Test 2: Test VirtIO Wrapper in QEMU

Create test script:

```python
#!/usr/bin/env python3
import numpy as np
import sys
sys.path.insert(0, '/mnt/shared/device_gemm')

from virtio_sst_wrapper import VirtIOSST

M, N, K = 4, 8, 4
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

with VirtIOSST() as sst:
    result = sst.gemm(M, N, K, A.tobytes(), B.tobytes())
    print(f"Cycles: {result['cycles']}")
    print(f"Memory bytes: {result['memory_bytes']}")
```

### Test 3: End-to-End PyTorch Test

```bash
cd /home/user/projects/acalsim/src/qemu-acalsim-sst-linux/examples/llama-inference/device_gemm
python3 test_device_gemm.py
```

---

## Advanced Topics

### Custom Performance Models

The simple roofline model can be replaced with more sophisticated models:

**Option 1: Tiled GEMM Model**
```cpp
const uint32_t TILE_SIZE = 256;
uint32_t M_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;
uint64_t total_tiles = M_tiles * N_tiles * K_tiles;
uint64_t total_cycles = total_tiles * cycles_per_tile;
```

**Option 2: Systolic Array Model**
```cpp
const uint32_t SYSTOLIC_DIM = 128;
uint64_t passes = ((M + SYSTOLIC_DIM - 1) / SYSTOLIC_DIM) *
                  ((N + SYSTOLIC_DIM - 1) / SYSTOLIC_DIM) * K;
uint64_t compute_cycles = passes * SYSTOLIC_DIM;
```

### Adding More Operators

Follow the same pattern for Conv2D, Attention, etc.:
1. Add request type to enum
2. Add payload structure
3. Implement performance model in SST component
4. Add Python wrapper method

---

## Troubleshooting

### Issue 1: SST Component Not Found
```bash
sst-config --prefix  # Check install location
sudo make install    # Reinstall
ls $(sst-config --prefix)/lib/sstcore-*/elemlib/  # Verify libacalsim.so
```

### Issue 2: Protocol Mismatch
- Ensure sst-protocol.h is identical everywhere
- Rebuild kernel module and SST component
- Check VirtIO wrapper uses correct constants

### Issue 3: /dev/sst0 Not Found
```bash
lsmod | grep virtio_sst
sudo modprobe virtio-sst
ls -l /dev/sst0
sudo chmod 666 /dev/sst0
```

---

## Summary

**Key files modified:**
- `sst-protocol.h`: Protocol definitions
- `ACALSimVirtIODeviceComponent.cc`: Performance model implementation
- `virtio_sst_wrapper.py`: Python interface
- `qemu_device_server_virtio.py`: QEMU integration

**Next steps:**
- Implement more sophisticated performance models
- Add support for other operators
- Profile real workloads and tune models

---

**Copyright 2023-2026 Playlab/ACAL**
**Licensed under Apache License 2.0**
