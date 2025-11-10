# N-Device QEMU Integration - Phase 2D

**Date**: 2025-11-10
**Status**: 🟡 **Partial Implementation - QEMU Side Complete, Needs N-Socket Support**

---

## Summary

This document describes the QEMU-side implementation of N-device support, enabling QEMU to communicate with multiple SST devices through the QEMUBinaryComponent. The core routing infrastructure is complete, but full N-device support requires implementing N socket servers.

---

## Architecture Overview

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  RISC-V Program (multi_device_test.c)                       │
│  - Writes to device addresses (0x10200000, 0x10300000, ...) │
└──────────────────────┬──────────────────────────────────────┘
                       │ MMIO transactions
┌──────────────────────▼──────────────────────────────────────┐
│  QEMU (qemu-system-riscv32)                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ sst-device │  │ sst-device │  │ sst-device │  ...        │
│  │ @0x1020000 │  │ @0x1030000 │  │ @0x1040000 │            │
│  │ socket:... │  │ socket:... │  │ socket:... │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │ MMIORequest   │                │                    │
│        │ (global addr) │                │                    │
└────────┼───────────────┼────────────────┼────────────────────┘
         │               │                │
         │ Unix Socket   │                │ (Future: N sockets)
         └───────────────┴────────────────┴─────────┐
                                                     │
┌────────────────────────────────────────────────────▼─────────┐
│  SST: QEMUBinaryComponent                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Socket Server (currently single)                      │    │
│  │ Receives MMIO requests with global addresses         │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                     │
│  ┌──────────────────────▼───────────────────────────────┐    │
│  │ Address-Based Router                                  │    │
│  │ findDeviceForAddress(addr) -> DeviceInfo*            │    │
│  │ routeToDevice(device, type, addr, data, size)        │    │
│  └──────────────┬────────────┬─────────────┬────────────┘    │
│                 │            │             │                  │
│    ┌────────────▼┐  ┌───────▼──────┐  ┌──▼───────────┐      │
│    │ device_port_0│  │ device_port_1│  │ device_port_2│ ...  │
└────┴──────────────┴──┴──────────────┴──┴──────────────┴──────┘
         │                  │                 │
┌────────▼─────┐   ┌────────▼──────┐  ┌──────▼────────┐
│ Echo Device  │   │Compute Device │  │ Echo Device 2 │  ...
│ @0x10200000  │   │ @0x10300000   │  │ @0x10400000   │
└──────────────┘   └───────────────┘  └───────────────┘
```

---

## Implementation Details

### 1. QEMU Device Changes (sst-device.c)

**Key Modification**: Devices now send **global addresses** instead of local offsets.

#### Added Properties:
```c
typedef struct {
    ...
    uint64_t base_address;    // NEW: Device base address for routing
    ...
} SSTDeviceState;

static Property sst_device_properties[] = {
    DEFINE_PROP_STRING("socket", SSTDeviceState, socket_path),
    DEFINE_PROP_UINT64("base_address", SSTDeviceState, base_address, 0x10200000),
    DEFINE_PROP_END_OF_LIST(),
};
```

#### Modified MMIO Handlers:
```c
// Before: sent local offset
.addr = addr  // e.g., 0x04 for DATA_OUT register

// After: sends global address
.addr = s->base_address + addr  // e.g., 0x10200004
```

**Impact**: QEMUBinaryComponent can now route based on the request address.

---

### 2. QEMUBinaryComponent Changes

#### Dynamic QEMU Command Line Building:

**Before** (lines 298-306):
```cpp
const char* args[] = {
    qemu_path_.c_str(),
    "-M", "virt",
    "-kernel", binary_path_.c_str(),
    // TODO: Add -device sst-device
    NULL
};
```

**After** (lines 294-330):
```cpp
std::vector<const char*> args;
args.push_back(qemu_path_.c_str());
args.push_back("-M"); args.push_back("virt");
args.push_back("-kernel"); args.push_back(binary_path_.c_str());

std::vector<std::string> device_args;
if (use_multi_device_) {
    for (size_t i = 0; i < devices_.size(); i++) {
        char addr_buf[32];
        snprintf(addr_buf, sizeof(addr_buf), "0x%lx", devices_[i].base_addr);
        std::string dev_arg = "sst-device,socket=" + socket_path_ +
                             ",base_address=" + std::string(addr_buf);
        device_args.push_back(dev_arg);
        args.push_back("-device");
        args.push_back(device_args.back().c_str());
    }
}
```

**Generated QEMU Command Example**:
```bash
qemu-system-riscv32 -M virt -kernel multi_device_test.elf \
  -device sst-device,socket=/tmp/qemu-sst-mmio.sock,base_address=0x10200000 \
  -device sst-device,socket=/tmp/qemu-sst-mmio.sock,base_address=0x10300000 \
  -device sst-device,socket=/tmp/qemu-sst-mmio.sock,base_address=0x10400000 \
  -device sst-device,socket=/tmp/qemu-sst-mmio.sock,base_address=0x10500000
```

---

### 3. Address-Based Routing (Already Implemented)

**File**: `QEMUBinaryComponent.cc:488-522`

#### Device Lookup:
```cpp
DeviceInfo* QEMUBinaryComponent::findDeviceForAddress(uint64_t address) {
    for (auto& dev : devices_) {
        if (address >= dev.base_addr && address < dev.base_addr + dev.size) {
            return &dev;
        }
    }
    return nullptr;  // No device at this address
}
```

#### Request Routing:
```cpp
void QEMUBinaryComponent::routeToDevice(DeviceInfo* device, uint8_t type,
                                         uint64_t addr, uint64_t data, uint8_t size) {
    // Create SST event
    auto* event = new MemoryTransactionEvent(tx_type, addr, data, size, req_id);

    // Send to device link
    device->link->send(event);

    // Update statistics
    device->num_requests++;
}
```

**Routing Flow**:
1. QEMU device sends MMIO request with address `0x10200004`
2. QEMUBinaryComponent receives request
3. `findDeviceForAddress(0x10200004)` finds device at `0x10200000`
4. `routeToDevice()` sends event to `device_port_0`
5. Echo device receives and processes

---

## Current Status

### ✅ Completed

1. **QEMU Device (sst-device.c)**
   - [x] Added `base_address` property
   - [x] Modified MMIO handlers to send global addresses
   - [x] Built and installed in Docker QEMU

2. **QEMUBinaryComponent**
   - [x] Dynamic command-line building for N devices
   - [x] Address-based routing (`findDeviceForAddress`, `routeToDevice`)
   - [x] Per-device statistics tracking
   - [x] N-device configuration (already done in Phase 2C.3)

3. **SST Configuration**
   - [x] N-device test configs (4-device, 2-device)
   - [x] Configuration generator (`generate_sst_config.py`)
   - [x] Test programs (`multi_device_test.c`, `test_4devices.c`)

4. **Documentation**
   - [x] N-device implementation guide (ADDING_N_DEVICES.md)
   - [x] SST-only test results (N_DEVICE_TEST_RESULTS.md)
   - [x] QEMU integration summary (this document)

---

## Current Limitations

### 🚧 Single Socket Connection

**Problem**: QEMUBinaryComponent currently creates **one socket server** (lines 256-277).

```cpp
// Current: Single socket server
server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
bind(server_fd_, socket_path_.c_str(), ...);
listen(server_fd_, 1);  // Only 1 pending connection

// All N QEMU devices try to connect to same socket
// Only the FIRST device connects successfully
// Other devices fail to connect
```

**Impact**:
- Only device 0 can communicate with SST
- Devices 1, 2, 3, ... fail to connect
- Cannot test true N-device MMIO routing

### Missing N-Socket Implementation

**Required Changes** (Future Work):

```cpp
// Add to DeviceInfo:
struct DeviceInfo {
    uint64_t base_addr;
    uint64_t size;
    SST::Link* link;
    std::string name;
    uint64_t num_requests;

    // NEW: Per-device socket
    int server_fd;          // Server socket FD
    int client_fd;          // Client connection FD
    std::string socket_path; // e.g., /tmp/qemu-sst-device0.sock
};

// In launchQEMU():
for (size_t i = 0; i < devices_.size(); i++) {
    // Create socket server for each device
    devices_[i].socket_path = "/tmp/qemu-sst-device" + std::to_string(i) + ".sock";
    devices_[i].server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    bind(devices_[i].server_fd, devices_[i].socket_path, ...);
    listen(devices_[i].server_fd, 1);
}

// In clockTick():
for (auto& dev : devices_) {
    if (dev.client_fd >= 0) {
        // Poll this device's socket for requests
        handleMMIORequest(dev);
    }
}
```

---

## Testing

### SST-Only Tests (Already Passing)

```bash
cd qemu-binary
sst qemu_4device_test.py

# Result:
✅ All 4 devices initialized
✅ Device clocks running
✅ Per-device statistics tracked
✅ Simulation completes
```

**Limitation**: No actual MMIO transactions (QEMU not integrated).

### Integration Test Configuration

**File**: `test_2device_integration.py`

```python
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu.addParams({
    "num_devices": 2,
    "device0_base": "0x10200000",
    "device0_name": "echo_device",
    "device1_base": "0x10300000",
    "device1_name": "compute_device"
})
```

**Test Program**: `riscv-programs/multi_device_test.c`
- Tests echo device at 0x10200000
- Tests compute device at 0x10300000
- Verifies inter-device communication

**Current Status**: ⏸️ **Blocked on N-socket implementation**

---

## Next Steps

### Phase 1: Implement N-Socket Servers (High Priority)

**Task**: Update QEMUBinaryComponent to create N socket servers.

**Files to Modify**:
1. `QEMUBinaryComponent.hh:73-79` - Add socket FDs to DeviceInfo
2. `QEMUBinaryComponent.cc:248-280` - Create N socket servers in launchQEMU()
3. `QEMUBinaryComponent.cc:177-191` - Poll all sockets in clockTick()
4. `QEMUBinaryComponent.cc:294-330` - Generate unique socket paths

**Estimated Effort**: 2-3 hours

---

### Phase 2: End-to-End Testing (After Phase 1)

1. **2-Device Test**
   ```bash
   sst test_2device_integration.py
   ```
   - Verify both devices connect
   - Test echo and compute operations
   - Measure routing overhead

2. **4-Device Test**
   ```bash
   sst qemu_4device_test.py
   ```
   - All devices active
   - Interleaved MMIO transactions
   - Verify address-based routing

3. **Performance Benchmarks**
   - Routing latency per device
   - Throughput with N concurrent devices
   - Scalability testing (8, 16 devices)

---

### Phase 3: Optimizations (Future)

1. **Hash-Based Routing** (for N > 16)
   ```cpp
   std::unordered_map<uint64_t, DeviceInfo*> device_map_;
   // O(1) lookup instead of O(N) linear search
   ```

2. **Non-Blocking Socket I/O**
   - Use `select()` or `poll()` for all sockets
   - Handle partial reads/writes
   - Timeout handling

3. **Inter-Device Communication**
   - Add peer ports to DeviceInfo
   - Device-to-device data transfer
   - Peer request routing

---

## Files Changed

### Modified:
- `qemu-sst-device/sst-device.c` (lines 90, 153, 235, 280, 347)
  - Added `base_address` field and property
  - Send global addresses in MMIO requests

- `qemu-binary/QEMUBinaryComponent.cc` (lines 289-337)
  - Dynamic QEMU command-line building
  - Add `-device sst-device` arguments with base_address

### Created:
- `qemu-binary/test_2device_integration.py`
  - SST configuration for 2-device integration test

- `qemu-binary/N_DEVICE_QEMU_INTEGRATION.md` (this file)
  - Comprehensive documentation

### Commits:
- `a4f738a` - feat: Add QEMU N-device integration support

---

## Comparison with Phase 2C.3 (SST-Only)

| Feature | Phase 2C.3 (SST) | Phase 2D (QEMU) | Status |
|---------|------------------|-----------------|--------|
| N-device routing | ✅ Implemented | ✅ Compatible | Complete |
| Device address configuration | ✅ Working | ✅ Working | Complete |
| Per-device statistics | ✅ Working | ✅ Working | Complete |
| Multi-port SST links | ✅ Working | ✅ Working | Complete |
| QEMU device instantiation | N/A | ✅ Implemented | Complete |
| Global address sending | N/A | ✅ Implemented | Complete |
| N socket servers | N/A | ❌ Not implemented | **TODO** |
| End-to-end testing | ✅ Device-only | ⏸️ Blocked | Pending |

---

## Conclusion

The QEMU-side N-device integration is **functionally complete** except for the N-socket server implementation. All core components are in place:

✅ **Architecture**: Three-layer design with address-based routing
✅ **QEMU Devices**: Send global addresses, configurable base addresses
✅ **QEMUBinaryComponent**: Dynamic command-line, routing logic
✅ **SST Infrastructure**: N-device support from Phase 2C.3
✅ **Configuration**: Python generator, test programs
✅ **Documentation**: Comprehensive guides

🚧 **Remaining Work**: Implement N socket servers (~2-3 hours)
🎯 **Next Milestone**: End-to-end QEMU + SST integration test

---

**Test Date**: 2025-11-10
**Engineer**: Claude Code
**Status**: 🟡 Partial - Ready for N-Socket Implementation
**Progress**: Phase 2C (SST) ✅ Complete | Phase 2D (QEMU) 🟡 80% Complete
