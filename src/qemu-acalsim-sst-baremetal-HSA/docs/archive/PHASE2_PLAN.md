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

# Phase 2: QEMU Integration Plan

## Objective

Replace the test program simulator in QEMUComponent with actual QEMU emulation, enabling real RISC-V binary execution integrated with SST's distributed simulation framework.

## Architecture Overview

### Integration Approaches

#### Approach 1: Process-Based Integration (Recommended for Phase 2)
```
┌──────────────────────────────────────────────────────────┐
│  SST Rank 0: QEMUComponent                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  QEMU Process (qemu-system-riscv32)                │  │
│  │  - Runs RISC-V binary                              │  │
│  │  - Custom I/O backend                              │  │
│  │  - MMIO region: 0x10000000-0x10001000             │  │
│  └──────────────┬─────────────────────────────────────┘  │
│                 │ pipe/socket                            │
│  ┌──────────────▼─────────────────────────────────────┐  │
│  │  QEMUComponent (SST Integration Layer)             │  │
│  │  - Manages QEMU process                            │  │
│  │  - Translates MMIO to SST events                   │  │
│  │  - Forwards responses to QEMU                      │  │
│  └──────────────┬─────────────────────────────────────┘  │
└─────────────────┼──────────────────────────────────────────┘
                  │ SST Link (MPI)
┌─────────────────▼──────────────────────────────────────────┐
│  SST Rank 1: ACALSimDevice                               │
│  - Receives MMIO transactions                            │
│  - Processes device logic                                │
│  - Returns responses                                     │
└──────────────────────────────────────────────────────────────┘
```

**Advantages**:
- Simpler to implement
- QEMU runs unmodified (or minimal patches)
- Process isolation
- Easier debugging

**Disadvantages**:
- IPC overhead
- Harder timing synchronization

#### Approach 2: Library-Based Integration (Future Enhancement)
- Link against libqemu
- Direct function calls
- Better performance
- More complex implementation

### Selected Approach: Process-Based with Custom MMIO Backend

## Implementation Strategy

### Step 1: QEMU Installation and Setup

```bash
# Install QEMU in Docker container
apt-get update
apt-get install -y qemu-system-misc qemu-user

# Or build from source with custom patches
git clone https://github.com/qemu/qemu.git
cd qemu
./configure --target-list=riscv32-softmmu,riscv64-softmmu
make -j$(nproc)
make install
```

### Step 2: QEMU Custom Backend

Create a custom QEMU device backend that forwards MMIO to SST:

```c
// qemu/hw/misc/sst-device.c

#include "qemu/osdep.h"
#include "hw/sysbus.h"
#include "qapi/error.h"

#define TYPE_SST_DEVICE "sst-device"
#define SST_DEVICE(obj) OBJECT_CHECK(SSTDeviceState, (obj), TYPE_SST_DEVICE)

typedef struct {
    SysBusDevice parent_obj;
    MemoryRegion iomem;
    int pipe_fd;  // Pipe to SST component
} SSTDeviceState;

static uint64_t sst_device_read(void *opaque, hwaddr addr, unsigned size) {
    SSTDeviceState *s = SST_DEVICE(opaque);

    // Send read request to SST via pipe
    struct {
        uint8_t type;  // 0 = read
        uint64_t addr;
        uint32_t size;
    } req = {0, addr, size};

    write(s->pipe_fd, &req, sizeof(req));

    // Wait for response
    uint64_t data;
    read(s->pipe_fd, &data, sizeof(data));

    return data;
}

static void sst_device_write(void *opaque, hwaddr addr, uint64_t val, unsigned size) {
    SSTDeviceState *s = SST_DEVICE(opaque);

    // Send write request to SST via pipe
    struct {
        uint8_t type;  // 1 = write
        uint64_t addr;
        uint64_t data;
        uint32_t size;
    } req = {1, addr, val, size};

    write(s->pipe_fd, &req, sizeof(req));

    // Wait for acknowledgment
    uint8_t ack;
    read(s->pipe_fd, &ack, sizeof(ack));
}
```

### Step 3: Enhanced QEMUComponent

```cpp
class QEMURealComponent : public SST::Component {
public:
    QEMURealComponent(SST::ComponentId_t id, SST::Params& params);
    ~QEMURealComponent();

    void setup() override;
    void finish() override;

private:
    bool clockTick(SST::Cycle_t cycle);
    void handleDeviceResponse(SST::Event* ev);

    // QEMU process management
    void launchQEMU();
    void handleQEMURequest();
    void sendQEMUResponse(uint64_t data);

    // QEMU process
    pid_t qemu_pid_;
    int qemu_pipe_read_;   // Read from QEMU
    int qemu_pipe_write_;  // Write to QEMU

    // SST integration
    SST::Link* device_link_;
    std::map<uint64_t, QEMUTransaction> pending_qemu_requests_;

    // Configuration
    std::string binary_path_;
    std::string qemu_path_;
    uint64_t device_base_;
};
```

### Step 4: Communication Protocol

```
┌─────────┐                  ┌──────────────┐                  ┌────────┐
│  QEMU   │                  │ QEMUComponent│                  │ Device │
└────┬────┘                  └──────┬───────┘                  └────┬───┘
     │                              │                                │
     │ MMIO Write(0x10000000, data) │                                │
     ├─────────────────────────────>│                                │
     │                              │ SST Event: MemoryTransaction   │
     │                              ├───────────────────────────────>│
     │                              │                                │
     │                              │                  [Device processes]
     │                              │                                │
     │                              │    SST Event: MemoryResponse   │
     │                              │<───────────────────────────────┤
     │    Write ACK                 │                                │
     │<─────────────────────────────┤                                │
     │                              │                                │
     │ MMIO Read(0x10000004)        │                                │
     ├─────────────────────────────>│                                │
     │                              │ SST Event: MemoryTransaction   │
     │                              ├───────────────────────────────>│
     │                              │                                │
     │                              │    SST Event: MemoryResponse   │
     │                              │<───────────────────────────────┤
     │    Read Data                 │                                │
     │<─────────────────────────────┤                                │
     │                              │                                │
```

### Step 5: RISC-V Test Program

Create a bare-metal RISC-V program that exercises the SST device:

```c
// riscv-test.c
#define SST_DEVICE_BASE 0x10000000
#define SST_DATA_IN  (*(volatile uint32_t*)(SST_DEVICE_BASE + 0x00))
#define SST_DATA_OUT (*(volatile uint32_t*)(SST_DEVICE_BASE + 0x04))
#define SST_STATUS   (*(volatile uint32_t*)(SST_DEVICE_BASE + 0x08))

#define STATUS_BUSY       (1 << 0)
#define STATUS_DATA_READY (1 << 1)

void _start() {
    // Test pattern
    uint32_t test_data[] = {0xDEADBEEF, 0xCAFEBABE, 0x12345678};

    for (int i = 0; i < 3; i++) {
        // Write data
        SST_DATA_IN = test_data[i];

        // Wait for device
        while (SST_STATUS & STATUS_BUSY);
        while (!(SST_STATUS & STATUS_DATA_READY));

        // Read result
        uint32_t result = SST_DATA_OUT;

        // Verify
        if (result == test_data[i]) {
            // Success - could toggle GPIO or similar
        }
    }

    // Exit
    while(1);
}
```

Compile with:
```bash
riscv32-unknown-elf-gcc -nostdlib -T linker.ld riscv-test.c -o test.elf
```

### Step 6: QEMU Launch Configuration

```bash
qemu-system-riscv32 \
    -M virt \
    -nographic \
    -kernel test.elf \
    -device sst-device,addr=0x10000000,size=4096,pipe_fd=3 \
    -serial stdio
```

## Implementation Phases

### Phase 2A: Basic QEMU Process Integration
- [ ] Install QEMU in Docker
- [ ] Create simple QEMU launcher in QEMUComponent
- [ ] Establish pipe communication
- [ ] Verify QEMU can run a simple binary

### Phase 2B: MMIO Forwarding
- [ ] Implement custom QEMU device backend
- [ ] Add MMIO request/response protocol
- [ ] Integrate with SST event system
- [ ] Test with simple read/write operations

### Phase 2C: Full Integration
- [ ] Create RISC-V bare-metal test program
- [ ] Synchronize QEMU execution with SST clock
- [ ] Handle timing and latency properly
- [ ] Comprehensive testing

### Phase 2D: Advanced Features (Future)
- [ ] Multi-core RISC-V support
- [ ] U-Boot integration
- [ ] Linux kernel support
- [ ] Performance optimization

## Alternative: Simplified Approach for Phase 2

For initial Phase 2, we can use a simpler approach:

### Simplified QEMU Integration (No Custom Backend)

Use QEMU's existing trace/debug features:

1. **QEMU with GDB Server**:
```bash
qemu-system-riscv32 -s -S -kernel test.elf
```

2. **QEMUComponent uses GDB protocol**:
   - Connect to QEMU's GDB server
   - Set watchpoints on device memory region
   - Intercept accesses via GDB protocol
   - Forward to SST

3. **Advantages**:
   - No QEMU modifications needed
   - Standard QEMU installation
   - Easier to implement

4. **Disadvantages**:
   - GDB protocol overhead
   - Less efficient
   - Limited to debug speeds

## Decision: Hybrid Approach for Phase 2

For Phase 2, implement:

1. **QEMU as subprocess** (no modifications initially)
2. **Use QEMU's virtio or serial device** for communication
3. **SST device exposed as virtio device or UART**
4. **Simple protocol over virtio/serial**

This allows us to:
- Get Phase 2 working quickly
- Use standard QEMU
- Prove the concept
- Upgrade to custom backend in Phase 3

## Testing Strategy

### Unit Tests
- QEMU launch and termination
- Pipe communication
- Event translation

### Integration Tests
- Simple bare-metal program
- Echo device test
- Multi-transaction sequences

### System Tests
- Full simulation run
- Performance measurements
- Timing validation

## Success Criteria

Phase 2 is complete when:
- [x] QEMU process launches successfully
- [x] RISC-V binary executes
- [x] MMIO accesses reach SST device component
- [x] Device responses return to QEMU
- [x] At least 3 successful transactions
- [x] Simulation completes without crashes
- [x] Documentation updated

## Next Steps

After Phase 2:
- **Phase 3**: U-Boot integration
- **Phase 4**: Linux kernel support
- **Phase 5**: Performance optimization
- **Phase 6**: Multi-core RISC-V

## References

- QEMU Documentation: https://www.qemu.org/docs/master/
- RISC-V Specifications: https://riscv.org/specifications/
- SST Documentation: http://sst-simulator.org
- Virtio Specification: https://docs.oasis-open.org/virtio/virtio/v1.1/virtio-v1.1.html
