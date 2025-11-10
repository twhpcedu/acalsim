# Phase 2B: QEMU-SST Integration Design

## Overview

Phase 2B implements basic QEMU-SST integration using a **simplified protocol-based approach** over QEMU's serial interface. This avoids modifying QEMU while demonstrating the concept.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  SST Rank 0: QEMURealComponent                               │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  QEMU Process                                          │  │
│  │  - Runs RISC-V binary                                  │  │
│  │  - Serial output: unix socket                          │  │
│  │                                                         │  │
│  │  RISC-V Program:                                       │  │
│  │    uart_puts("SST:WRITE:20000000:DEADBEEF\n");       │  │
│  │    response = uart_gets();                            │  │
│  └──────────────┬─────────────────────────────────────────┘  │
│                 │ Unix socket (serial)                       │
│  ┌──────────────▼─────────────────────────────────────────┐  │
│  │  QEMURealComponent                                     │  │
│  │  - Monitors QEMU serial output                         │  │
│  │  - Parses SST protocol commands                        │  │
│  │  - Translates to MemoryTransactionEvent                │  │
│  │  - Sends response back to QEMU                         │  │
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

## Communication Protocol

### Protocol Format

All SST device communication uses UART with a simple text protocol:

**Request Format**:
```
SST:<CMD>:<ADDR>:<DATA>\n
```

Where:
- `CMD`: Operation type (`WRITE`, `READ`)
- `ADDR`: Hexadecimal address (8 digits)
- `DATA`: Hexadecimal data (8 digits, optional for READ)

**Response Format**:
```
SST:OK:<DATA>\n      # Success
SST:ERR:<CODE>\n     # Error
```

### Example Exchange

```
RISC-V → UART: "SST:WRITE:20000000:DEADBEEF\n"
UART → RISC-V: "SST:OK:00000000\n"

RISC-V → UART: "SST:READ:20000004:00000000\n"
UART → RISC-V: "SST:OK:DEADBEEF\n"
```

## Implementation Details

### 1. QEMU Launch Configuration

```cpp
// QEMURealComponent launches QEMU with:
const char* qemu_args[] = {
    "qemu-system-riscv32",
    "-M", "virt",
    "-bios", "none",
    "-nographic",
    "-kernel", binary_path_.c_str(),
    "-serial", "unix:/tmp/qemu-sst.sock,server",
    NULL
};
```

### 2. Serial Socket Communication

```cpp
class QEMURealComponent {
private:
    int serial_fd_;          // Unix socket to QEMU serial
    pid_t qemu_pid_;         // QEMU process ID

    void handleSerialData();  // Read from QEMU serial
    void parseCommand(const std::string& line);
    void sendResponse(const std::string& resp);
};
```

### 3. RISC-V Helper Library

```c
// sst_device.h - Helper functions for RISC-V programs

// Write to SST device
bool sst_write(uint32_t addr, uint32_t data) {
    uart_printf("SST:WRITE:%08X:%08X\n", addr, data);
    char response[64];
    uart_gets(response, sizeof(response));
    return (strncmp(response, "SST:OK", 6) == 0);
}

// Read from SST device
bool sst_read(uint32_t addr, uint32_t* data) {
    uart_printf("SST:READ:%08X:00000000\n", addr);
    char response[64];
    uart_gets(response, sizeof(response));
    if (strncmp(response, "SST:OK:", 7) == 0) {
        *data = parse_hex(&response[7]);
        return true;
    }
    return false;
}
```

### 4. State Machine

```cpp
enum class QEMUState {
    LAUNCHING,      // Starting QEMU process
    RUNNING,        // QEMU running, monitoring serial
    WAITING_DEVICE, // Waiting for SST device response
    COMPLETED,      // Binary finished
    ERROR           // Error occurred
};
```

## Modified Components

### QEMURealComponent.hh (New)

Enhanced component with QEMU process management:

```cpp
class QEMURealComponent : public SST::Component {
public:
    // Constructor, destructor
    QEMURealComponent(SST::ComponentId_t id, SST::Params& params);
    ~QEMURealComponent();

    // SST lifecycle
    void setup() override;
    void finish() override;

private:
    // Clock and event handlers
    bool clockTick(SST::Cycle_t cycle);
    void handleDeviceResponse(SST::Event* ev);

    // QEMU management
    void launchQEMU();
    void monitorQEMU();
    void terminateQEMU();

    // Serial communication
    void handleSerialData();
    void parseCommand(const std::string& line);
    void sendDeviceRequest(const std::string& cmd);
    void sendSerialResponse(const std::string& resp);

    // State
    QEMUState state_;
    pid_t qemu_pid_;
    int serial_fd_;
    std::string serial_buffer_;

    // SST integration
    SST::Link* device_link_;
    std::map<uint64_t, std::string> pending_requests_;
    uint64_t next_req_id_;

    // Configuration
    std::string binary_path_;
    std::string qemu_path_;
};
```

### sst_device_test.c (New)

RISC-V program using SST device:

```c
#include "sst_device.h"

void _start_c(void) {
    uart_puts("QEMU-SST Integration Test\n");

    // Test 1: Write to SST device
    if (sst_write(SST_DATA_IN, 0xDEADBEEF)) {
        uart_puts("Write successful\n");
    }

    // Test 2: Poll status
    uint32_t status;
    do {
        sst_read(SST_STATUS, &status);
    } while (status & STATUS_BUSY);

    // Test 3: Read result
    uint32_t result;
    if (sst_read(SST_DATA_OUT, &result)) {
        uart_printf("Read: 0x%08X\n", result);
    }

    uart_puts("Test complete!\n");
}
```

## Build Integration

### Updated Makefile

```makefile
# QEMU real component
qemu-real/QEMURealComponent.o: qemu-real/QEMURealComponent.cc
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

libqemu-real.so: qemu-real/QEMURealComponent.o
	$(CXX) $(LDFLAGS) -shared -fPIC -o $@ $^

# RISC-V program with SST support
sst_device_test.elf: start.o sst_device_test.o sst_device.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@
```

## Testing Strategy

### Unit Tests

1. **QEMU Launch**: Verify QEMU starts correctly
2. **Serial Communication**: Test socket read/write
3. **Protocol Parsing**: Validate command parsing
4. **Process Management**: Test clean shutdown

### Integration Tests

1. **Simple Write**: RISC-V writes to SST device
2. **Simple Read**: RISC-V reads from SST device
3. **Echo Test**: Write then read back same data
4. **Multiple Transactions**: 5 write/read cycles

### Success Criteria

- ✅ QEMU process launches successfully
- ✅ Serial socket establishes connection
- ✅ RISC-V program sends valid SST commands
- ✅ QEMURealComponent parses commands correctly
- ✅ MemoryTransactionEvents reach device
- ✅ Responses return to RISC-V program
- ✅ Echo test passes (write 0xDEADBEEF, read 0xDEADBEEF)
- ✅ Simulation completes without crashes

## Limitations & Future Work

### Current Limitations

1. **Text Protocol**: Not cycle-accurate, uses text parsing
2. **Serial Bottleneck**: All communication via UART
3. **No Timing Accuracy**: QEMU runs at full speed
4. **Single Transaction**: One request at a time

### Phase 2C Improvements

1. **Custom QEMU Device**: Replace text protocol with binary MMIO
2. **Timing Synchronization**: Pause QEMU between SST cycles
3. **Parallel Transactions**: Support multiple outstanding requests
4. **Performance**: Direct memory access instead of parsing

### Phase 3+ Enhancements

1. **U-Boot Integration**: Boot real bootloader
2. **Linux Support**: Run Linux kernel
3. **Multi-core**: Multiple RISC-V cores
4. **DMA**: Direct memory access support

## Performance Expectations

For Phase 2B (text protocol):
- Throughput: ~1000 transactions/second
- Latency: ~1ms per transaction
- Overhead: ~90% parsing, 10% simulation

This is acceptable for proof-of-concept. Phase 2C will improve dramatically with binary protocol.

## Development Workflow

```bash
# 1. Build SST components
cd src/qemu-sst/qemu-real
make && make install

# 2. Build RISC-V program
cd ../riscv-programs
make sst_device_test.elf

# 3. Run distributed simulation
cd ../config
mpirun -n 2 sst qemu_real_test.py
```

## Expected Output

```
[Rank 0] Launching QEMU process...
[Rank 0] QEMU PID: 12345
[Rank 0] Serial socket connected
[QEMU] QEMU-SST Integration Test
[Rank 0] Received: SST:WRITE:20000000:DEADBEEF
[Rank 0] Sending MemoryTransactionEvent to device
[Rank 1] Device received WRITE: addr=0x20000000 data=0xDEADBEEF
[Rank 1] Sending MemoryResponseEvent
[Rank 0] Received device response, forwarding to QEMU
[QEMU] Write successful
[QEMU] Read: 0xDEADBEEF
[QEMU] Test complete!
[Rank 0] QEMU exited with code 0
[Rank 0] Simulation complete
```

## Next Steps

After Phase 2B completion:
1. Measure performance and identify bottlenecks
2. Design binary MMIO protocol for Phase 2C
3. Investigate QEMU custom device backend
4. Plan timing synchronization mechanism
5. Prepare for U-Boot integration (Phase 3)

## References

- [QEMU Serial Device](https://qemu.readthedocs.io/en/latest/system/devices/serial.html)
- [Unix Domain Sockets](https://man7.org/linux/man-pages/man7/unix.7.html)
- [QEMU Process Management](https://qemu.readthedocs.io/en/latest/system/invocation.html)
