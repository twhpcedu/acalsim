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

# ACALSim MMIO Device with Interrupt Support

Complete guide for creating ACALSim-based MMIO devices that communicate with QEMU via load/store operations and interrupts.

## Table of Contents

- [Overview](#overview)
- [Communication Pattern](#communication-pattern)
- [Device Architecture](#device-architecture)
- [Register Map](#register-map)
- [Implementation Example](#implementation-example)
- [Driver Code Example](#driver-code-example)
- [SST Configuration](#sst-configuration)
- [Best Practices](#best-practices)

## Overview

ACALSim MMIO devices provide a realistic model of hardware devices that communicate with the CPU through:
- **Memory-Mapped I/O (MMIO)**: CPU uses load/store instructions to access device registers
- **Interrupts**: Device signals CPU asynchronously when events occur

This pattern is typical in real hardware devices like:
- DMA controllers
- Network interfaces
- Storage controllers
- GPU compute engines
- Custom accelerators

## Communication Pattern

### 1. MMIO Transactions (CPU → Device)

```
┌─────────┐                              ┌──────────────┐
│  QEMU   │                              │  ACALSim     │
│  (CPU)  │                              │  Device      │
└─────────┘                              └──────────────┘
     │                                          │
     │  MemoryTransactionEvent (STORE)         │
     │  - address: 0x10001000                  │
     │  - data: 0x00000001                     │
     │  - size: 4 bytes                        │
     │  - req_id: 123                          │
     │────────────────────────────────────────>│
     │                                          │
     │                                          │ Process write
     │                                          │ Update register
     │                                          │
     │  MemoryResponseEvent                    │
     │  - req_id: 123                          │
     │  - success: true                        │
     │<────────────────────────────────────────│
```

### 2. Interrupt Signaling (Device → CPU)

```
┌─────────┐                              ┌──────────────┐
│  QEMU   │                              │  ACALSim     │
│  (CPU)  │                              │  Device      │
└─────────┘                              └──────────────┘
     │                                          │
     │                                          │ Operation complete
     │                                          │ Set INT_STATUS
     │                                          │
     │  InterruptEvent (ASSERT)                │
     │  - irq_num: 1                           │
     │  - type: ASSERT                         │
     │<────────────────────────────────────────│
     │                                          │
     │  Enter ISR                               │
     │  Read INT_STATUS (LOAD)                 │
     │────────────────────────────────────────>│
     │                                          │
     │  Write INT_STATUS to clear (STORE)      │
     │────────────────────────────────────────>│
     │                                          │
     │  InterruptEvent (DEASSERT)              │
     │  - irq_num: 1                           │
     │  - type: DEASSERT                       │
     │<────────────────────────────────────────│
     │                                          │
     │  Exit ISR                                │
```

### 3. Complete Operation Flow

```
1. CPU Configuration Phase:
   - CPU writes SRC_ADDR register    (MMIO STORE)
   - CPU writes DST_ADDR register    (MMIO STORE)
   - CPU writes LENGTH register      (MMIO STORE)
   - CPU writes INT_ENABLE register  (MMIO STORE)

2. CPU Start Operation:
   - CPU writes CTRL[START]=1        (MMIO STORE)
   - Device sets STATUS[BUSY]=1
   - Device begins operation

3. Device Processing:
   - Cycle-accurate modeling
   - Configurable latency
   - Realistic resource contention

4. Operation Completion:
   - Device sets STATUS[DONE]=1
   - Device sets INT_STATUS[COMPLETE]=1
   - Device sends InterruptEvent (ASSERT) to QEMU

5. CPU Interrupt Handler:
   - CPU enters ISR
   - CPU reads STATUS register       (MMIO LOAD)
   - CPU reads INT_STATUS register   (MMIO LOAD)
   - CPU processes completion
   - CPU writes INT_STATUS[COMPLETE]=1 to clear (MMIO STORE)
   - Device sends InterruptEvent (DEASSERT)
   - CPU exits ISR
```

## Device Architecture

### Event Types

#### 1. MemoryTransactionEvent (CPU → Device)

Used for MMIO load/store operations:

```cpp
class MemoryTransactionEvent : public SST::Event {
public:
    enum class Type { LOAD, STORE };

    MemoryTransactionEvent(Type type, uint64_t addr,
                          uint32_t data, uint32_t size,
                          uint64_t req_id);

    Type     getType() const;
    uint64_t getAddress() const;
    uint32_t getData() const;
    uint32_t getSize() const;
    uint64_t getReqId() const;
};
```

#### 2. MemoryResponseEvent (Device → CPU)

Response to MMIO transactions:

```cpp
class MemoryResponseEvent : public SST::Event {
public:
    MemoryResponseEvent(uint64_t req_id, uint32_t data, bool success);

    uint64_t getReqId() const;
    uint32_t getData() const;
    bool     getSuccess() const;
};
```

#### 3. InterruptEvent (Device → CPU)

Interrupt signaling:

```cpp
class InterruptEvent : public SST::Event {
public:
    enum class Type { ASSERT, DEASSERT };

    InterruptEvent(uint32_t irq_num, Type type);

    uint32_t getIrqNum() const;
    Type     getType() const;
    bool     isAssert() const;
};
```

### SST Links

MMIO devices require two links:

```cpp
// 1. CPU Link - for MMIO transactions
cpu_link_ = configureLink("cpu_port",
    new SST::Event::Handler<Device>(this, &Device::handleMemoryTransaction));

// 2. IRQ Link - for interrupt signaling
irq_link_ = configureLink("irq_port");
```

## Register Map

### ACALSimMMIODevice Register Map

Standard 4KB device with comprehensive register set:

| Offset | Name          | Access | Description                                      |
|--------|---------------|--------|--------------------------------------------------|
| 0x00   | CTRL          | RW     | Control register                                 |
|        |               |        | [0]: Start operation (auto-clear)                |
|        |               |        | [1]: Reset device                                |
|        |               |        | [2]: Enable interrupts                           |
| 0x04   | STATUS        | R      | Status register                                  |
|        |               |        | [0]: Busy (operation in progress)                |
|        |               |        | [1]: Done (operation completed)                  |
|        |               |        | [2]: Error                                       |
| 0x08   | INT_STATUS    | R/W1C  | Interrupt status (Write 1 to Clear)              |
|        |               |        | [0]: Operation complete IRQ                      |
|        |               |        | [1]: Error IRQ                                   |
| 0x0C   | INT_ENABLE    | RW     | Interrupt enable mask                            |
|        |               |        | [0]: Enable completion IRQ                       |
|        |               |        | [1]: Enable error IRQ                            |
| 0x10   | SRC_ADDR      | RW     | Source address for DMA-like operations           |
| 0x14   | DST_ADDR      | RW     | Destination address for DMA-like operations      |
| 0x18   | LENGTH        | RW     | Transfer length in bytes                         |
| 0x1C   | LATENCY       | RW     | Operation latency (cycles)                       |
| 0x20   | DATA_IN       | W      | Data input (for simple operations)               |
| 0x24   | DATA_OUT      | R      | Data output (for simple operations)              |
| 0x28   | CYCLE_COUNT   | R      | Device cycle counter                             |

### Register Access Patterns

**Write-1-to-Clear (W1C)**: Write 1 to bit to clear it
```c
// Clear completion interrupt
*INT_STATUS = (1 << 0);  // Clears bit 0, other bits unchanged
```

**Read-Modify-Write (RMW)**: Careful with side effects
```c
// Enable completion interrupt without affecting other bits
uint32_t val = *INT_ENABLE;
val |= (1 << 0);
*INT_ENABLE = val;
```

**Auto-clear**: Some bits clear automatically after operation
```c
*CTRL = CTRL_START;  // Starts operation, bit cleared by hardware
```

## Implementation Example

### Device Class Structure

```cpp
class ACALSimMMIODevice : public SST::Component {
public:
    // Constructor with parameter handling
    ACALSimMMIODevice(SST::ComponentId_t id, SST::Params& params);

    // SST lifecycle
    void setup() override;
    void finish() override;
    bool clockTick(SST::Cycle_t cycle);

    // MMIO handler
    void handleMemoryTransaction(SST::Event* ev);

private:
    // Register access
    uint32_t readRegister(uint64_t offset);
    void     writeRegister(uint64_t offset, uint32_t value);

    // Device operations
    void startOperation();
    void completeOperation();
    void resetDevice();

    // Interrupt handling
    void generateInterrupt(uint32_t irq_bits);
    void clearInterrupt(uint32_t irq_bits);
    void updateInterruptLine();

    // SST links
    SST::Link* cpu_link_;
    SST::Link* irq_link_;

    // Device registers
    uint32_t reg_ctrl_;
    uint32_t reg_status_;
    uint32_t reg_int_status_;
    uint32_t reg_int_enable_;
    // ... other registers

    // Device state
    bool irq_asserted_;
    struct Operation {
        bool active;
        uint64_t end_cycle;
        // ... operation details
    } current_op_;
};
```

### Key Implementation Patterns

#### 1. MMIO Transaction Handler

```cpp
void ACALSimMMIODevice::handleMemoryTransaction(SST::Event* ev) {
    auto* trans = dynamic_cast<MemoryTransactionEvent*>(ev);

    uint64_t offset = trans->getAddress() - base_addr_;
    uint32_t resp_data = 0;

    if (trans->getType() == TransactionType::LOAD) {
        resp_data = readRegister(offset);
    } else {
        writeRegister(offset, trans->getData());
    }

    // Send response
    auto* resp = new MemoryResponseEvent(
        trans->getReqId(), resp_data, true);
    cpu_link_->send(resp);

    delete ev;
}
```

#### 2. Register Write with Side Effects

```cpp
void ACALSimMMIODevice::writeRegister(uint64_t offset, uint32_t value) {
    switch (offset) {
        case REG_CTRL:
            reg_ctrl_ = value;
            if (value & CTRL_START) {
                startOperation();
                reg_ctrl_ &= ~CTRL_START;  // Auto-clear
            }
            if (value & CTRL_RESET) {
                resetDevice();
            }
            break;

        case REG_INT_STATUS:
            // Write-1-to-clear
            reg_int_status_ &= ~value;
            updateInterruptLine();
            break;

        // ... other registers
    }
}
```

#### 3. Interrupt Generation

```cpp
void ACALSimMMIODevice::generateInterrupt(uint32_t irq_bits) {
    // Set interrupt status
    reg_int_status_ |= irq_bits;

    updateInterruptLine();
}

void ACALSimMMIODevice::updateInterruptLine() {
    // Assert IRQ if any enabled interrupt is pending
    bool should_assert = (reg_int_status_ & reg_int_enable_) != 0;

    if (should_assert && !irq_asserted_) {
        auto* irq = new InterruptEvent(irq_num_, InterruptEvent::Type::ASSERT);
        irq_link_->send(irq);
        irq_asserted_ = true;
    } else if (!should_assert && irq_asserted_) {
        auto* irq = new InterruptEvent(irq_num_, InterruptEvent::Type::DEASSERT);
        irq_link_->send(irq);
        irq_asserted_ = false;
    }
}
```

#### 4. Cycle-Accurate Operation

```cpp
bool ACALSimMMIODevice::clockTick(SST::Cycle_t cycle) {
    // Check for operation completion
    if (current_op_.active && cycle >= current_op_.end_cycle) {
        completeOperation();
    }
    return false;
}

void ACALSimMMIODevice::startOperation() {
    current_op_.active = true;
    current_op_.start_cycle = getCurrentCycle();
    current_op_.end_cycle = current_op_.start_cycle + reg_latency_;

    reg_status_ |= STATUS_BUSY;
}

void ACALSimMMIODevice::completeOperation() {
    reg_status_ &= ~STATUS_BUSY;
    reg_status_ |= STATUS_DONE;

    current_op_.active = false;

    // Generate interrupt if enabled
    if (reg_ctrl_ & CTRL_INT_EN) {
        generateInterrupt(INT_COMPLETE);
    }
}
```

## Driver Code Example

### Bare-Metal C Driver

```c
// Device base address (from memory map)
#define MMIO_DEV_BASE   0x10001000

// Register offsets
#define REG_CTRL        0x00
#define REG_STATUS      0x04
#define REG_INT_STATUS  0x08
#define REG_INT_ENABLE  0x0C
#define REG_SRC_ADDR    0x10
#define REG_DST_ADDR    0x14
#define REG_LENGTH      0x18
#define REG_LATENCY     0x1C
#define REG_DATA_IN     0x20
#define REG_DATA_OUT    0x24

// Control bits
#define CTRL_START      (1 << 0)
#define CTRL_RESET      (1 << 1)
#define CTRL_INT_EN     (1 << 2)

// Status bits
#define STATUS_BUSY     (1 << 0)
#define STATUS_DONE     (1 << 1)
#define STATUS_ERROR    (1 << 2)

// Interrupt bits
#define INT_COMPLETE    (1 << 0)
#define INT_ERROR       (1 << 1)

// Register access macros
#define MMIO_WRITE(offset, value) \
    (*(volatile uint32_t *)(MMIO_DEV_BASE + (offset)) = (value))

#define MMIO_READ(offset) \
    (*(volatile uint32_t *)(MMIO_DEV_BASE + (offset)))

// Global flag for interrupt handling
volatile uint32_t operation_complete = 0;

/**
 * Interrupt Service Routine
 */
void mmio_device_isr(void) {
    // Read interrupt status
    uint32_t int_status = MMIO_READ(REG_INT_STATUS);

    if (int_status & INT_COMPLETE) {
        // Operation completed
        operation_complete = 1;

        // Clear interrupt by writing 1
        MMIO_WRITE(REG_INT_STATUS, INT_COMPLETE);
    }

    if (int_status & INT_ERROR) {
        // Handle error
        MMIO_WRITE(REG_INT_STATUS, INT_ERROR);
    }
}

/**
 * Initialize device
 */
void mmio_device_init(void) {
    // Reset device
    MMIO_WRITE(REG_CTRL, CTRL_RESET);

    // Wait for reset
    while (MMIO_READ(REG_STATUS) & STATUS_BUSY);

    // Enable interrupts
    MMIO_WRITE(REG_INT_ENABLE, INT_COMPLETE | INT_ERROR);
}

/**
 * Start DMA-like operation with interrupt
 */
int mmio_device_transfer(uint32_t src, uint32_t dst, uint32_t length) {
    // Check if device is busy
    if (MMIO_READ(REG_STATUS) & STATUS_BUSY) {
        return -1;  // Device busy
    }

    // Configure operation
    MMIO_WRITE(REG_SRC_ADDR, src);
    MMIO_WRITE(REG_DST_ADDR, dst);
    MMIO_WRITE(REG_LENGTH, length);

    // Clear completion flag
    operation_complete = 0;

    // Start operation with interrupt enabled
    MMIO_WRITE(REG_CTRL, CTRL_START | CTRL_INT_EN);

    // Wait for interrupt (could do other work here)
    while (!operation_complete) {
        // Could use WFI (Wait For Interrupt) instruction
        asm volatile("wfi");
    }

    // Check status
    uint32_t status = MMIO_READ(REG_STATUS);
    if (status & STATUS_ERROR) {
        return -1;
    }

    return 0;  // Success
}

/**
 * Polling-based transfer (no interrupts)
 */
int mmio_device_transfer_poll(uint32_t src, uint32_t dst, uint32_t length) {
    // Check if device is busy
    if (MMIO_READ(REG_STATUS) & STATUS_BUSY) {
        return -1;
    }

    // Configure operation
    MMIO_WRITE(REG_SRC_ADDR, src);
    MMIO_WRITE(REG_DST_ADDR, dst);
    MMIO_WRITE(REG_LENGTH, length);

    // Start operation (no interrupts)
    MMIO_WRITE(REG_CTRL, CTRL_START);

    // Poll until done
    while (MMIO_READ(REG_STATUS) & STATUS_BUSY);

    // Check for errors
    if (MMIO_READ(REG_STATUS) & STATUS_ERROR) {
        return -1;
    }

    return 0;
}

/**
 * Simple echo operation
 */
uint32_t mmio_device_echo(uint32_t value) {
    MMIO_WRITE(REG_DATA_IN, value);
    return MMIO_READ(REG_DATA_OUT);
}

/**
 * Example usage
 */
int main(void) {
    // Initialize device
    mmio_device_init();

    // Register ISR (platform-specific)
    register_irq_handler(IRQ_MMIO_DEVICE, mmio_device_isr);
    enable_irq(IRQ_MMIO_DEVICE);

    // Perform transfer with interrupt
    int result = mmio_device_transfer(
        0x80000000,  // Source
        0x80001000,  // Destination
        1024         // Length
    );

    if (result < 0) {
        // Handle error
    }

    // Simple echo test
    uint32_t echo_result = mmio_device_echo(0xDEADBEEF);

    return 0;
}
```

## SST Configuration

### Python Configuration Example

```python
import sst

# Create QEMU component
qemu = sst.Component("qemu", "acalsim.QEMUBinary")
qemu.addParams({
    "binary_path": "./baremetal.elf",
    "verbose": 2
})

# Create MMIO Device
mmio_dev = sst.Component("mmio_device", "acalsim.MMIODevice")
mmio_dev.addParams({
    "clock": "1GHz",
    "base_addr": "0x10001000",
    "size": "4096",
    "verbose": 2,
    "default_latency": "100",
    "irq_num": "1"
})

# Connect MMIO link (for load/store)
mmio_link = sst.Link("mmio_link")
mmio_link.connect(
    (qemu, "device_port_0", "1ns"),
    (mmio_dev, "cpu_port", "1ns")
)

# Connect IRQ link (for interrupts)
irq_link = sst.Link("irq_link")
irq_link.connect(
    (qemu, "irq_port_0", "1ns"),
    (mmio_dev, "irq_port", "1ns")
)

# Enable statistics
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputConsole")

mmio_dev.enableAllStatistics()

# Simulation parameters
sst.setProgramOption("stop-at", "10ms")
```

## Best Practices

### 1. Register Design

- **Group related registers**: Control, status, interrupt registers together
- **Use standard patterns**: W1C for interrupt status, auto-clear for commands
- **Provide read-back**: All writable registers should be readable
- **Document side effects**: Clear documentation when writes trigger actions

### 2. Interrupt Handling

- **Level-triggered preferred**: Easier to debug, no risk of missing edges
- **Clear separation**: Status vs. enable vs. pending
- **Auto-deassert**: Clear IRQ when status cleared
- **Minimize latency**: Fast ISR, defer work to handlers

### 3. Cycle Accuracy

- **Model realistic delays**: Based on real hardware characteristics
- **Configurable latency**: Allow tuning for different scenarios
- **Resource contention**: Model queues, pipelines, conflicts
- **Performance counters**: Expose cycle counts, throughput metrics

### 4. Debugging

- **Verbosity levels**: 0=quiet, 1=major events, 2=transactions, 3=detailed
- **Trace transactions**: Log every MMIO access with timestamp
- **State dumps**: Provide register snapshot capability
- **Statistics**: Track MMIO counts, interrupt counts, latencies

### 5. Testing

- **Unit tests**: Test each register independently
- **Integration tests**: Full operation flow with driver
- **Stress tests**: Rapid transactions, queue overflow
- **Error injection**: Test error paths, recovery

## Related Documentation

- `ACALSimDeviceComponent.hh` - Base echo device
- `ACALSimComputeDeviceComponent.hh` - Compute device example
- `HSAEvents.hh` - HSA protocol events
- SST Documentation: https://sst-simulator.org/

---

**Last Updated**: 2025-11-10
