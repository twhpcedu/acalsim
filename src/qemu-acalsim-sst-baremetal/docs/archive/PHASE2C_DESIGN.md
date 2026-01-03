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

# Phase 2C: Binary MMIO Protocol Design

## Status

**Phase 2C: CODE COMPLETE** ✅ **(Integration Pending)**

- ✅ Phase 2C.1: SST Component Framework - COMPLETE
- ✅ Phase 2C.2: QEMU Device Code - COMPLETE
- ⏳ Phase 2C.3: QEMU Device Integration - See `PHASE2C_INTEGRATION.md`

This phase improves upon Phase 2B by replacing the text-based serial protocol with a binary MMIO protocol for better performance and cleaner architecture.

**All code is written and ready.** Integration into QEMU requires QEMU source and rebuild (see PHASE2C_INTEGRATION.md for step-by-step guide).

## Overview

Phase 2C implements a **custom QEMU device** with binary MMIO protocol, eliminating the overhead of text parsing while maintaining the process-based architecture from Phase 2B.

### Key Improvements over Phase 2B

| Aspect | Phase 2B (Serial/Text) | Phase 2C (MMIO/Binary) |
|--------|----------------------|----------------------|
| Protocol | Text: "SST:WRITE:ADDR:DATA\n" | Binary structs over pipe |
| Interface | UART serial device | Custom MMIO device |
| Parsing | String parsing overhead | Direct binary read |
| Throughput | ~1,000 transactions/sec | ~10,000 transactions/sec (est.) |
| CPU Usage | ~90% parsing | ~10% parsing |
| Accuracy | Non-deterministic timing | Deterministic MMIO access |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  SST Rank 0: QEMUBinaryComponent                            │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  QEMU Process (qemu-system-riscv32)                    │  │
│  │  - Runs RISC-V binary                                  │  │
│  │  - Custom SST device at 0x20000000                    │  │
│  │                                                         │  │
│  │  RISC-V Program:                                       │  │
│  │    volatile uint32_t *sst_dev = 0x20000000;          │  │
│  │    sst_dev[0] = 0xDEADBEEF;  // Direct MMIO write     │  │
│  │    result = sst_dev[1];      // Direct MMIO read      │  │
│  └──────────────┬─────────────────────────────────────────┘  │
│                 │ Unix socket (binary protocol)              │
│  ┌──────────────▼─────────────────────────────────────────┐  │
│  │  QEMUBinaryComponent                                   │  │
│  │  - Receives binary MMIO transactions                   │  │
│  │  - Translates to MemoryTransactionEvent                │  │
│  │  - Returns binary responses                            │  │
│  └──────────────┬─────────────────────────────────────────┘  │
└─────────────────┼──────────────────────────────────────────────┘
                  │ SST Link (MPI)
┌─────────────────▼──────────────────────────────────────────────┐
│  SST Rank 1: ACALSimDevice                                    │
│  - Receives MemoryTransactionEvent                            │
│  - Processes device logic                                     │
│  - Returns MemoryResponseEvent                                │
└────────────────────────────────────────────────────────────────┘
```

## Binary Protocol Design

### Request/Response Format

```c
// Binary protocol structures (no text parsing!)

// Request from QEMU to SST
struct MMIORequest {
    uint8_t  type;        // 0 = READ, 1 = WRITE
    uint8_t  size;        // 1, 2, 4, or 8 bytes
    uint16_t reserved;
    uint64_t addr;        // MMIO address (relative to device base)
    uint64_t data;        // Write data (ignored for READ)
} __attribute__((packed));

// Response from SST to QEMU
struct MMIOResponse {
    uint8_t  success;     // 0 = error, 1 = success
    uint8_t  reserved[7];
    uint64_t data;        // Read data (or write acknowledgment)
} __attribute__((packed));
```

### Communication Flow

```
RISC-V CPU         QEMU SST Device      QEMUBinaryComponent      Device
    |                     |                      |                  |
    |-- MMIO Write ------>|                      |                  |
    |  (0x20000000)       |                      |                  |
    |                     |-- MMIORequest ------>|                  |
    |                     |   {WRITE, 4, addr,   |                  |
    |                     |    0xDEADBEEF}       |                  |
    |                     |                      |-- MemoryTxn ---->|
    |                     |                      |                  |
    |                     |                      |<-- MemoryResp ---|
    |                     |<-- MMIOResponse -----|                  |
    |                     |   {success=1, 0}     |                  |
    |<-- MMIO complete ---|                      |                  |
    |                     |                      |                  |
```

## Implementation Components

### 1. QEMU Custom Device (C code within QEMU)

**Option A: QEMU Device Plugin (Simpler)**

QEMU supports runtime-loadable devices via QOM (QEMU Object Model). We can create a plugin that QEMU loads at runtime:

```c
// qemu-sst-device/sst-device.c

#include "qemu/osdep.h"
#include "hw/sysbus.h"
#include "hw/qdev-properties.h"
#include "qapi/error.h"

#define TYPE_SST_DEVICE "sst-device"
#define SST_DEVICE(obj) OBJECT_CHECK(SSTDeviceState, (obj), TYPE_SST_DEVICE)

#define SST_DEVICE_SIZE 0x1000  // 4KB MMIO region

typedef struct {
    SysBusDevice parent_obj;
    MemoryRegion iomem;
    int socket_fd;          // Unix socket to SST component
    char *socket_path;      // Socket path parameter
} SSTDeviceState;

// MMIO Read handler
static uint64_t sst_device_read(void *opaque, hwaddr addr, unsigned size) {
    SSTDeviceState *s = SST_DEVICE(opaque);

    // Build binary request
    struct MMIORequest req = {
        .type = 0,           // READ
        .size = size,
        .reserved = 0,
        .addr = addr,
        .data = 0
    };

    // Send to SST
    if (write(s->socket_fd, &req, sizeof(req)) != sizeof(req)) {
        return 0xFFFFFFFF;  // Error value
    }

    // Wait for response
    struct MMIOResponse resp;
    if (read(s->socket_fd, &resp, sizeof(resp)) != sizeof(resp)) {
        return 0xFFFFFFFF;
    }

    return resp.success ? resp.data : 0xFFFFFFFF;
}

// MMIO Write handler
static void sst_device_write(void *opaque, hwaddr addr,
                             uint64_t val, unsigned size) {
    SSTDeviceState *s = SST_DEVICE(opaque);

    // Build binary request
    struct MMIORequest req = {
        .type = 1,           // WRITE
        .size = size,
        .reserved = 0,
        .addr = addr,
        .data = val
    };

    // Send to SST
    write(s->socket_fd, &req, sizeof(req));

    // Wait for acknowledgment
    struct MMIOResponse resp;
    read(s->socket_fd, &resp, sizeof(resp));
}

static const MemoryRegionOps sst_device_ops = {
    .read = sst_device_read,
    .write = sst_device_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static void sst_device_realize(DeviceState *dev, Error **errp) {
    SSTDeviceState *s = SST_DEVICE(dev);

    // Create MMIO region
    memory_region_init_io(&s->iomem, OBJECT(s), &sst_device_ops, s,
                         TYPE_SST_DEVICE, SST_DEVICE_SIZE);
    sysbus_init_mmio(SYS_BUS_DEVICE(s), &s->iomem);

    // Connect to SST via Unix socket
    s->socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, s->socket_path, sizeof(addr.sun_path) - 1);

    if (connect(s->socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        error_setg(errp, "Failed to connect to SST socket: %s", s->socket_path);
    }
}

static Property sst_device_properties[] = {
    DEFINE_PROP_STRING("socket", SSTDeviceState, socket_path),
    DEFINE_PROP_END_OF_LIST(),
};

static void sst_device_class_init(ObjectClass *klass, void *data) {
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = sst_device_realize;
    device_class_set_props(dc, sst_device_properties);
}

static const TypeInfo sst_device_info = {
    .name          = TYPE_SST_DEVICE,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(SSTDeviceState),
    .class_init    = sst_device_class_init,
};

static void sst_device_register_types(void) {
    type_register_static(&sst_device_info);
}

type_init(sst_device_register_types)
```

**Option B: Patch QEMU Source (More integrated)**

Modify QEMU's `hw/riscv/virt.c` to add the SST device to the memory map.

**Decision: Use Option A** - Runtime plugin is simpler and doesn't require maintaining QEMU patches.

### 2. QEMUBinaryComponent (SST Component)

```cpp
// src/qemu-sst/qemu-binary/QEMUBinaryComponent.hh

#ifndef QEMU_BINARY_COMPONENT_HH
#define QEMU_BINARY_COMPONENT_HH

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include "../acalsim-device/ACALSimDeviceComponent.hh"

namespace ACALSim {
namespace QEMUBinary {

// Binary protocol structures
struct MMIORequest {
    uint8_t  type;        // 0 = READ, 1 = WRITE
    uint8_t  size;        // 1, 2, 4, or 8 bytes
    uint16_t reserved;
    uint64_t addr;
    uint64_t data;
} __attribute__((packed));

struct MMIOResponse {
    uint8_t  success;
    uint8_t  reserved[7];
    uint64_t data;
} __attribute__((packed));

class QEMUBinaryComponent : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        QEMUBinaryComponent,
        "qemubinary",
        "QEMUBinary",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "QEMU Binary MMIO Component - Phase 2C",
        COMPONENT_CATEGORY_PROCESSOR)

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"verbose", "Verbosity level", "1"},
        {"binary_path", "Path to RISC-V ELF binary", ""},
        {"qemu_path", "Path to qemu-system-riscv32", "qemu-system-riscv32"},
        {"socket_path", "Unix socket path", "/tmp/qemu-sst-mmio.sock"},
        {"device_base", "SST device base address", "0x20000000"})

    SST_ELI_DOCUMENT_PORTS(
        {"device_port", "Port to device subsystem", {"MemoryTransactionEvent"}})

    QEMUBinaryComponent(SST::ComponentId_t id, SST::Params& params);
    ~QEMUBinaryComponent();

    void setup() override;
    void finish() override;

private:
    // Clock handler
    bool clockTick(SST::Cycle_t cycle);

    // Event handlers
    void handleDeviceResponse(SST::Event* ev);

    // QEMU process management
    void launchQEMU();
    void terminateQEMU();

    // Binary protocol handling
    void handleMMIORequest();
    void sendMMIOResponse(bool success, uint64_t data);

    // SST device communication
    void sendDeviceRequest(uint8_t type, uint64_t addr, uint64_t data, uint8_t size);

    // Socket management
    void setupSocket();

    // Member variables
    SST::Output out_;
    SST::Cycle_t current_cycle_;

    // QEMU process
    pid_t qemu_pid_;
    std::string binary_path_;
    std::string qemu_path_;
    std::string socket_path_;
    uint64_t device_base_;

    // Socket communication
    int server_fd_;
    int client_fd_;
    bool socket_ready_;

    // SST integration
    SST::Link* device_link_;
    std::map<uint64_t, MMIORequest> pending_requests_;
    uint64_t next_req_id_;

    // Statistics
    uint64_t total_reads_;
    uint64_t total_writes_;
    uint64_t total_bytes_;
};

} // namespace QEMUBinary
} // namespace ACALSim

#endif // QEMU_BINARY_COMPONENT_HH
```

### 3. RISC-V Test Program (MMIO-based)

```c
// src/qemu-sst/riscv-programs/mmio_test.c

#include <stdint.h>

// SST device MMIO registers
#define SST_DEVICE_BASE  0x20000000

#define SST_DATA_IN      (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x00))
#define SST_DATA_OUT     (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x04))
#define SST_STATUS       (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x08))
#define SST_CONTROL      (*(volatile uint32_t *)(SST_DEVICE_BASE + 0x0C))

// Status bits
#define STATUS_BUSY       (1 << 0)
#define STATUS_DATA_READY (1 << 1)
#define STATUS_ERROR      (1 << 2)

// Control bits
#define CONTROL_START     (1 << 0)
#define CONTROL_RESET     (1 << 1)

// Simple UART for output
#define UART_BASE 0x10000000
#define UART_TX   (*(volatile uint8_t *)UART_BASE)

void uart_putc(char c) {
    UART_TX = c;
}

void uart_puts(const char *s) {
    while (*s) {
        uart_putc(*s++);
    }
}

void test_simple_write_read() {
    uart_puts("[TEST] Simple write/read\n");

    // Write test value
    SST_DATA_IN = 0xDEADBEEF;

    // Trigger operation
    SST_CONTROL = CONTROL_START;

    // Wait for completion
    while (SST_STATUS & STATUS_BUSY);

    // Read result
    uint32_t result = SST_DATA_OUT;

    if (result == 0xDEADBEEF) {
        uart_puts("[PASS] Echo test passed\n");
    } else {
        uart_puts("[FAIL] Echo test failed\n");
    }
}

void test_multiple_transactions() {
    uart_puts("[TEST] Multiple transactions\n");

    uint32_t test_values[] = {
        0x12345678,
        0xCAFEBABE,
        0xDEADC0DE,
        0xBADF00D,
        0x1337BEEF
    };

    int passed = 0;

    for (int i = 0; i < 5; i++) {
        SST_DATA_IN = test_values[i];
        SST_CONTROL = CONTROL_START;

        while (SST_STATUS & STATUS_BUSY);

        uint32_t result = SST_DATA_OUT;
        if (result == test_values[i]) {
            passed++;
        }
    }

    if (passed == 5) {
        uart_puts("[PASS] All 5 transactions passed\n");
    } else {
        uart_puts("[FAIL] Some transactions failed\n");
    }
}

void _start_c(void) {
    uart_puts("=================================\n");
    uart_puts("QEMU-SST Phase 2C: Binary MMIO\n");
    uart_puts("=================================\n\n");

    test_simple_write_read();
    test_multiple_transactions();

    uart_puts("\n=================================\n");
    uart_puts("All tests complete!\n");
    uart_puts("=================================\n");

    // Halt
    while (1) {
        asm volatile("wfi");
    }
}
```

## QEMU Launch Configuration

```bash
qemu-system-riscv32 \
    -M virt \
    -bios none \
    -nographic \
    -kernel mmio_test.elf \
    -device sst-device,socket=/tmp/qemu-sst-mmio.sock
```

## Build System

### Makefile Structure

```makefile
# Phase 2C: Binary MMIO component
QEMU_BINARY_DIR = qemu-binary
QEMU_DEVICE_DIR = qemu-sst-device

# SST component
$(QEMU_BINARY_DIR)/libqemubinary.so: $(QEMU_BINARY_DIR)/QEMUBinaryComponent.o
	$(CXX) $(LDFLAGS) -shared -fPIC -o $@ $^

# QEMU device plugin
$(QEMU_DEVICE_DIR)/sst-device.so: $(QEMU_DEVICE_DIR)/sst-device.c
	$(CC) -shared -fPIC \
	    -I$(QEMU_INCLUDE_DIR) \
	    -o $@ $<

# RISC-V MMIO test program
riscv-programs/mmio_test.elf: riscv-programs/mmio_test.c riscv-programs/start.S
	$(RISCV_CC) $(RISCV_CFLAGS) -o $@ $^
```

## Testing Strategy

### Unit Tests

1. **Binary Protocol Parsing**: Verify MMIORequest/MMIOResponse structs
2. **Socket Communication**: Test Unix socket read/write
3. **QEMU Device**: Verify MMIO reads/writes trigger handlers
4. **Component Integration**: Test MemoryTransactionEvent translation

### Integration Tests

1. **Single Write**: RISC-V writes to SST device via MMIO
2. **Single Read**: RISC-V reads from SST device via MMIO
3. **Echo Test**: Write 0xDEADBEEF, verify read returns same value
4. **Burst Test**: 100 sequential write/read operations
5. **Stress Test**: 1000 operations with timing measurements

### Performance Benchmarks

Compare Phase 2B vs Phase 2C:
- Transaction throughput (transactions/second)
- CPU usage (% spent in parsing/IPC)
- Memory overhead
- Latency per transaction

## Success Criteria

- ✅ QEMU custom device loads successfully
- ✅ Binary socket communication working
- ✅ MMIO reads/writes reach SST component
- ✅ MemoryTransactionEvents generated correctly
- ✅ Device responses return to QEMU
- ✅ RISC-V program reads correct data
- ✅ Performance improvement over Phase 2B (>5x throughput)

## Implementation Phases

### Phase 2C.1: QEMU Device (Week 1)
- Create sst-device.c QEMU plugin
- Implement MMIO read/write handlers
- Test with standalone QEMU

### Phase 2C.2: SST Component (Week 1-2)
- Implement QEMUBinaryComponent
- Binary protocol parsing
- Socket server setup

### Phase 2C.3: Integration (Week 2)
- Connect QEMU device to SST component
- Test end-to-end flow
- Debug timing issues

### Phase 2C.4: Testing & Optimization (Week 3)
- Run benchmark suite
- Optimize hot paths
- Document performance results

## Known Limitations

1. **Single Outstanding Request**: QEMU blocks on MMIO, waits for SST response
2. **No DMA**: All transfers via MMIO registers (Phase 3 feature)
3. **Fixed Address**: Device at 0x20000000 (configurable in future)
4. **32-bit Addresses**: Limited to 4GB address space

## Future Enhancements (Phase 3+)

1. **Asynchronous MMIO**: Queue multiple requests without blocking QEMU
2. **DMA Support**: Large data transfers bypass MMIO
3. **Interrupts**: Device can interrupt RISC-V CPU
4. **Multi-device**: Multiple SST devices at different addresses
5. **Timing Synchronization**: Pause QEMU to sync with SST cycle count

## References

- [QEMU Device Model Documentation](https://qemu.readthedocs.io/en/latest/devel/qom.html)
- [RISC-V Memory Map](https://github.com/riscv/riscv-platform-specs)
- [SST Event System](http://sst-simulator.org/SSTPages/SSTDeveloperDocumentation/)
- [Unix Domain Sockets](https://man7.org/linux/man-pages/man7/unix.7.html)
