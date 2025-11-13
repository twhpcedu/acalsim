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

# QEMU-AcalSim-SST Concrete Demo Examples

**Purpose**: Complete, working examples demonstrating how to create and use custom devices in the QEMU-SST simulation framework.

**Target Audience**: Users who want to model their own homogeneous or heterogeneous device systems.

---

## Table of Contents

1. [Example 1: Homogeneous 4-Accelerator System](#example-1-homogeneous-4-accelerator-system)
2. [Example 2: Heterogeneous Multi-Device System](#example-2-heterogeneous-multi-device-system)
3. [Creating Custom Devices](#creating-custom-devices)
4. [RISC-V Program Development](#risc-v-program-development)
5. [Running and Testing](#running-and-testing)

---

## Example 1: Homogeneous 4-Accelerator System

### Overview

This example demonstrates a system with 4 identical compute accelerators. Each accelerator:
- Performs vector addition operations
- Has 16 compute units
- Has configurable latency
- Operates independently

**Use Case**: Parallel workload distribution across identical processing units (like multi-GPU systems).

### System Architecture

```
┌──────────────────────────────────────────────────────────┐
│  RISC-V Program (QEMU)                                    │
│  - Distributes data across 4 accelerators                │
│  - Each accelerator: 0x10200000, 0x10300000, ...         │
└────────────┬─────────────────────────────────────────────┘
             │
       ┌─────┴──────┬──────┬──────┬──────┐
       │            │      │      │      │
   ┌───▼───┐   ┌───▼───┐ ┌▼────┐ ┌▼────┐
   │Accel 0│   │Accel 1│ │Acc 2│ │Acc 3│
   │ 16 CUs│   │ 16 CUs│ │16 CU│ │16 CU│
   │10 cyc │   │15 cyc │ │20 cy│ │25 cy│
   └───────┘   └───────┘ └─────┘ └─────┘
```

### Step 1: Create ComputeAccelerator Device

**File**: `demo-devices/ComputeAccelerator.hh`

```cpp
#ifndef _COMPUTE_ACCELERATOR_HH
#define _COMPUTE_ACCELERATOR_HH

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/event.h>
#include <queue>
#include <vector>

namespace DemoDevices {

class ComputeAccelerator : public SST::Component {
public:
    ComputeAccelerator(SST::ComponentId_t id, SST::Params& params);
    ~ComputeAccelerator();

    void setup() override;
    void finish() override;

    bool clockTick(SST::Cycle_t cycle);
    void handleCPURequest(SST::Event* event);

    SST_ELI_REGISTER_COMPONENT(
        ComputeAccelerator,
        "demodevices",
        "ComputeAccelerator",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Vector compute accelerator with N compute units",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"latency_cycles", "Processing latency in cycles", "10"},
        {"compute_units", "Number of parallel compute units", "16"},
        {"vector_size", "Maximum vector size", "256"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"cpu_port", "Port to QEMU CPU", {}}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"operations_completed", "Total operations", "ops", 1},
        {"avg_latency", "Average latency", "cycles", 1},
        {"utilization", "Compute unit utilization", "percent", 1}
    )

private:
    // Device registers (memory-mapped)
    enum Registers {
        REG_CONTROL     = 0x00,  // Control register
        REG_STATUS      = 0x04,  // Status register
        REG_SRC_ADDR_A  = 0x08,  // Source vector A address
        REG_SRC_ADDR_B  = 0x0C,  // Source vector B address
        REG_DST_ADDR    = 0x10,  // Destination address
        REG_VECTOR_SIZE = 0x14,  // Vector size (elements)
        REG_RESULT      = 0x18   // Result/error code
    };

    // Device state
    struct DeviceState {
        uint32_t control;
        uint32_t status;
        uint64_t src_addr_a;
        uint64_t src_addr_b;
        uint64_t dst_addr;
        uint32_t vector_size;
        uint32_t result;
    } dev_state_;

    // Configuration
    uint32_t latency_cycles_;
    uint32_t compute_units_;
    uint32_t max_vector_size_;

    // Runtime state
    SST::Link* cpu_link_;
    uint64_t cycles_remaining_;
    bool busy_;

    // Statistics
    SST::Statistic<uint64_t>* stat_ops_;
    SST::Statistic<uint64_t>* stat_latency_;
    SST::Statistic<double>* stat_utilization_;

    SST::Output output_;

    // Helper methods
    void processRead(uint64_t addr, SST::Event* req_event);
    void processWrite(uint64_t addr, uint64_t data, SST::Event* req_event);
    void startComputation();
};

} // namespace DemoDevices

#endif
```

**File**: `demo-devices/ComputeAccelerator.cc`

```cpp
#include "ComputeAccelerator.hh"

using namespace DemoDevices;

// Simple event for CPU communication
class CPUEvent : public SST::Event {
public:
    enum Type { READ, WRITE, RESPONSE };

    CPUEvent(Type t, uint64_t a, uint64_t d = 0)
        : type(t), addr(a), data(d) {}

    Type type;
    uint64_t addr;
    uint64_t data;

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        ser& type;
        ser& addr;
        ser& data;
    }

    ImplementSerializable(CPUEvent);
};

ComputeAccelerator::ComputeAccelerator(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), busy_(false), cycles_remaining_(0)
{
    output_.init("ComputeAccelerator[@p:@l]: ", 1, 0, SST::Output::STDOUT);

    // Read configuration
    latency_cycles_ = params.find<uint32_t>("latency_cycles", 10);
    compute_units_ = params.find<uint32_t>("compute_units", 16);
    max_vector_size_ = params.find<uint32_t>("vector_size", 256);

    // Register clock
    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    registerClock(clock_freq,
        new SST::Clock::Handler<ComputeAccelerator>(this, &ComputeAccelerator::clockTick));

    // Configure CPU link
    cpu_link_ = configureLink("cpu_port",
        new SST::Event::Handler<ComputeAccelerator>(this, &ComputeAccelerator::handleCPURequest));

    if (!cpu_link_) {
        output_.fatal(CALL_INFO, -1, "Failed to configure cpu_port\n");
    }

    // Register statistics
    stat_ops_ = registerStatistic<uint64_t>("operations_completed");
    stat_latency_ = registerStatistic<uint64_t>("avg_latency");
    stat_utilization_ = registerStatistic<double>("utilization");

    // Initialize device state
    memset(&dev_state_, 0, sizeof(dev_state_));
    dev_state_.status = 0x1;  // Bit 0: Ready

    output_.verbose(CALL_INFO, 1, 0,
        "Initialized: %u CUs, %u-cycle latency, max vector size %u\n",
        compute_units_, latency_cycles_, max_vector_size_);
}

ComputeAccelerator::~ComputeAccelerator() {}

void ComputeAccelerator::setup() {
    output_.verbose(CALL_INFO, 1, 0, "Setup complete\n");
}

void ComputeAccelerator::finish() {
    output_.verbose(CALL_INFO, 1, 0, "Finishing\n");
}

bool ComputeAccelerator::clockTick(SST::Cycle_t cycle) {
    if (busy_ && cycles_remaining_ > 0) {
        cycles_remaining_--;

        if (cycles_remaining_ == 0) {
            // Computation complete
            busy_ = false;
            dev_state_.status = 0x1;  // Ready
            dev_state_.result = 0;    // Success

            stat_ops_->addData(1);
            stat_latency_->addData(latency_cycles_);

            output_.verbose(CALL_INFO, 2, 0,
                "Computation complete (vector size: %u)\n", dev_state_.vector_size);
        }
    }

    return false;
}

void ComputeAccelerator::handleCPURequest(SST::Event* event) {
    CPUEvent* cpu_ev = dynamic_cast<CPUEvent*>(event);
    if (!cpu_ev) {
        output_.fatal(CALL_INFO, -1, "Invalid event type\n");
    }

    if (cpu_ev->type == CPUEvent::READ) {
        processRead(cpu_ev->addr, event);
    } else if (cpu_ev->type == CPUEvent::WRITE) {
        processWrite(cpu_ev->addr, cpu_ev->data, event);
    }

    delete event;
}

void ComputeAccelerator::processRead(uint64_t addr, SST::Event* req_event) {
    uint64_t data = 0;

    switch (addr & 0xFF) {
        case REG_CONTROL:
            data = dev_state_.control;
            break;
        case REG_STATUS:
            data = dev_state_.status;
            break;
        case REG_SRC_ADDR_A:
            data = dev_state_.src_addr_a;
            break;
        case REG_SRC_ADDR_B:
            data = dev_state_.src_addr_b;
            break;
        case REG_DST_ADDR:
            data = dev_state_.dst_addr;
            break;
        case REG_VECTOR_SIZE:
            data = dev_state_.vector_size;
            break;
        case REG_RESULT:
            data = dev_state_.result;
            break;
        default:
            output_.verbose(CALL_INFO, 2, 0, "Read from unknown register 0x%lx\n", addr);
            data = 0xDEADBEEF;
    }

    // Send response
    CPUEvent* resp = new CPUEvent(CPUEvent::RESPONSE, addr, data);
    cpu_link_->send(resp);
}

void ComputeAccelerator::processWrite(uint64_t addr, uint64_t data, SST::Event* req_event) {
    switch (addr & 0xFF) {
        case REG_CONTROL:
            dev_state_.control = data;
            if (data & 0x1) {  // Start bit
                startComputation();
            }
            break;
        case REG_SRC_ADDR_A:
            dev_state_.src_addr_a = data;
            break;
        case REG_SRC_ADDR_B:
            dev_state_.src_addr_b = data;
            break;
        case REG_DST_ADDR:
            dev_state_.dst_addr = data;
            break;
        case REG_VECTOR_SIZE:
            dev_state_.vector_size = data;
            break;
        default:
            output_.verbose(CALL_INFO, 2, 0, "Write to unknown register 0x%lx\n", addr);
    }

    // Send acknowledgment
    CPUEvent* resp = new CPUEvent(CPUEvent::RESPONSE, addr, 0);
    cpu_link_->send(resp);
}

void ComputeAccelerator::startComputation() {
    if (busy_) {
        output_.verbose(CALL_INFO, 1, 0, "Warning: Start while busy\n");
        dev_state_.result = 0xFFFFFFFF;  // Error: busy
        return;
    }

    if (dev_state_.vector_size > max_vector_size_) {
        output_.verbose(CALL_INFO, 1, 0, "Error: Vector size %u > max %u\n",
            dev_state_.vector_size, max_vector_size_);
        dev_state_.result = 0xFFFFFFFE;  // Error: size
        return;
    }

    // Calculate latency based on vector size and compute units
    uint32_t elements_per_cu = (dev_state_.vector_size + compute_units_ - 1) / compute_units_;
    cycles_remaining_ = latency_cycles_ * elements_per_cu;

    busy_ = true;
    dev_state_.status = 0x2;  // Bit 1: Busy
    dev_state_.control = 0;   // Clear start bit

    output_.verbose(CALL_INFO, 2, 0,
        "Starting computation: size=%u, latency=%lu cycles\n",
        dev_state_.vector_size, cycles_remaining_);
}
```

### Step 2: SST Configuration for 4 Accelerators

**File**: `demo-configs/quad_accel_test.py`

```python
#!/usr/bin/env python3
"""
4-Accelerator Homogeneous System Test

Demonstrates parallel workload distribution across 4 identical
compute accelerators with varying latencies.
"""

import sst

# System configuration
NUM_ACCELERATORS = 4
BASE_ADDR = 0x10200000
ADDR_STRIDE = 0x100000  # 1MB per accelerator

# QEMU component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu_params = {
    "clock": "1GHz",
    "binary_path": "../riscv-programs/quad_accel_test.elf",
    "qemu_path": "/home/user/qemu-build/qemu/build/qemu-system-riscv32",
    "num_devices": NUM_ACCELERATORS,
    "verbose": 2
}

# Create accelerators with increasing latency
for i in range(NUM_ACCELERATORS):
    addr = BASE_ADDR + (i * ADDR_STRIDE)
    qemu_params[f"device{i}_base"] = f"0x{addr:x}"
    qemu_params[f"device{i}_size"] = 4096
    qemu_params[f"device{i}_name"] = f"accel_{i}"

qemu.addParams(qemu_params)

# Create accelerator components
for i in range(NUM_ACCELERATORS):
    accel = sst.Component(f"accel_{i}", "demodevices.ComputeAccelerator")
    accel.addParams({
        "clock": "2GHz",  # Accelerators run at 2GHz
        "latency_cycles": 10 + (i * 5),  # Increasing latency: 10, 15, 20, 25 cycles
        "compute_units": 16,
        "vector_size": 256
    })

    # Connect to QEMU
    link = sst.Link(f"link_{i}")
    link.connect(
        (qemu, f"device_port_{i}", "1ns"),
        (accel, "cpu_port", "1ns")
    )

# Enable statistics
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputConsole")

qemu.enableAllStatistics()
for i in range(NUM_ACCELERATORS):
    sst.Component(f"accel_{i}").enableAllStatistics()

# Simulation settings
sst.setProgramOption("stop-at", "10ms")
```

### Step 3: RISC-V Test Program

**File**: `riscv-programs/quad_accel_test.c`

```c
#include <stdint.h>

// Accelerator register definitions
#define ACCEL_BASE(n)       (0x10200000 + ((n) * 0x100000))
#define ACCEL_REG(base, r)  (*(volatile uint32_t*)((base) + (r)))

#define REG_CONTROL         0x00
#define REG_STATUS          0x04
#define REG_SRC_ADDR_A      0x08
#define REG_SRC_ADDR_B      0x0C
#define REG_DST_ADDR        0x10
#define REG_VECTOR_SIZE     0x14
#define REG_RESULT          0x18

#define STATUS_READY        0x1
#define STATUS_BUSY         0x2

// Test data
#define VECTOR_SIZE         64
#define NUM_ACCELS          4

uint32_t vector_a[NUM_ACCELS][VECTOR_SIZE];
uint32_t vector_b[NUM_ACCELS][VECTOR_SIZE];
uint32_t results[NUM_ACCELS][VECTOR_SIZE];

// Initialize test vectors
void init_vectors(void) {
    for (int accel = 0; accel < NUM_ACCELS; accel++) {
        for (int i = 0; i < VECTOR_SIZE; i++) {
            vector_a[accel][i] = (accel * 1000) + i;
            vector_b[accel][i] = (accel * 2000) + i;
            results[accel][i] = 0;
        }
    }
}

// Submit work to accelerator
void submit_work(int accel_id, uint64_t base_addr) {
    // Set source addresses
    ACCEL_REG(base_addr, REG_SRC_ADDR_A) = (uint32_t)vector_a[accel_id];
    ACCEL_REG(base_addr, REG_SRC_ADDR_B) = (uint32_t)vector_b[accel_id];

    // Set destination
    ACCEL_REG(base_addr, REG_DST_ADDR) = (uint32_t)results[accel_id];

    // Set vector size
    ACCEL_REG(base_addr, REG_VECTOR_SIZE) = VECTOR_SIZE;

    // Start computation
    ACCEL_REG(base_addr, REG_CONTROL) = 0x1;
}

// Wait for accelerator to complete
int wait_complete(uint64_t base_addr) {
    uint32_t status;
    uint32_t timeout = 1000000;

    while (timeout-- > 0) {
        status = ACCEL_REG(base_addr, REG_STATUS);
        if (status & STATUS_READY) {
            return ACCEL_REG(base_addr, REG_RESULT);
        }
    }

    return -1;  // Timeout
}

int main(void) {
    // Initialize test data
    init_vectors();

    // Submit work to all accelerators in parallel
    for (int i = 0; i < NUM_ACCELS; i++) {
        uint64_t base = ACCEL_BASE(i);
        submit_work(i, base);
    }

    // Wait for all accelerators to complete
    int all_success = 1;
    for (int i = 0; i < NUM_ACCELS; i++) {
        uint64_t base = ACCEL_BASE(i);
        int result = wait_complete(base);

        if (result != 0) {
            all_success = 0;
        }
    }

    // Return 0 on success, 1 on failure
    return all_success ? 0 : 1;
}
```

**Build**:
```bash
cd riscv-programs
riscv32-unknown-elf-gcc -march=rv32gc -mabi=ilp32d -nostdlib -nostartfiles \
    -T linker.ld crt0.S quad_accel_test.c -o quad_accel_test.elf
```

### Step 4: Run the Example

```bash
# Build device component
cd demo-devices
make clean && make && make install

# Run simulation
cd ../demo-configs
sst quad_accel_test.py

# Expected output:
# ComputeAccelerator: accel_0 started, latency=10 cycles
# ComputeAccelerator: accel_1 started, latency=15 cycles
# ComputeAccelerator: accel_2 started, latency=20 cycles
# ComputeAccelerator: accel_3 started, latency=25 cycles
# ...
# ComputeAccelerator: accel_0 complete
# ComputeAccelerator: accel_1 complete
# ComputeAccelerator: accel_2 complete
# ComputeAccelerator: accel_3 complete
```

---

## Example 2: Heterogeneous Multi-Device System

### Overview

This example demonstrates a system with 4 different device types:
1. **Echo Device**: Simple register echo (latency: 1 cycle)
2. **Compute Device**: Vector operations (latency: 100 cycles)
3. **Memory Device**: Simulated SRAM (latency: 10 cycles)
4. **Custom DMA Device**: Direct memory access (latency: 50 cycles)

**Use Case**: Complex system simulation with specialized accelerators (like SoC with GPU, DSP, memory controller, and DMA engine).

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  RISC-V Program (QEMU)                                   │
│  - Tests each device type                               │
│  - Demonstrates device interaction                      │
└──────┬──────┬───────┬──────┬────────────────────────────┘
       │      │       │      │
   ┌───▼──┐ ┌─▼───┐ ┌▼────┐ ┌▼────┐
   │Echo  │ │Comp │ │Mem  │ │DMA  │
   │1 cyc │ │100c │ │10 c │ │50 c │
   │Simple│ │Vec  │ │SRAM │ │Bulk │
   └──────┘ └─────┘ └─────┘ └─────┘
```

### Device 1: Echo Device

**Purpose**: Simple device for testing - echoes written values back.

**File**: `demo-devices/EchoDevice.cc`

```cpp
#include "EchoDevice.hh"

using namespace DemoDevices;

EchoDevice::EchoDevice(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), last_value_(0)
{
    output_.init("EchoDevice[@p:@l]: ", 1, 0, SST::Output::STDOUT);

    latency_ = params.find<uint32_t>("latency", 1);

    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    registerClock(clock_freq,
        new SST::Clock::Handler<EchoDevice>(this, &EchoDevice::clockTick));

    cpu_link_ = configureLink("cpu_port",
        new SST::Event::Handler<EchoDevice>(this, &EchoDevice::handleRequest));

    if (!cpu_link_) {
        output_.fatal(CALL_INFO, -1, "Failed to configure cpu_port\n");
    }

    stat_requests_ = registerStatistic<uint64_t>("requests");
}

bool EchoDevice::clockTick(SST::Cycle_t cycle) {
    return false;
}

void EchoDevice::handleRequest(SST::Event* event) {
    CPUEvent* cpu_ev = dynamic_cast<CPUEvent*>(event);

    if (cpu_ev->type == CPUEvent::WRITE) {
        last_value_ = cpu_ev->data;
        stat_requests_->addData(1);

        output_.verbose(CALL_INFO, 2, 0, "Stored value: 0x%lx\n", last_value_);
    }

    // Always send response with last stored value
    CPUEvent* resp = new CPUEvent(CPUEvent::RESPONSE, cpu_ev->addr, last_value_);
    cpu_link_->send(latency_, resp);

    delete event;
}
```

### Device 2: Memory Device

**Purpose**: Simulated SRAM with configurable size and latency.

**File**: `demo-devices/MemoryDevice.hh`

```cpp
class MemoryDevice : public SST::Component {
public:
    MemoryDevice(SST::ComponentId_t id, SST::Params& params);
    ~MemoryDevice();

    bool clockTick(SST::Cycle_t cycle);
    void handleRequest(SST::Event* event);

    SST_ELI_REGISTER_COMPONENT(
        MemoryDevice,
        "demodevices",
        "MemoryDevice",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Simulated SRAM memory device",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"latency_cycles", "Access latency", "10"},
        {"size_kb", "Memory size in KB", "64"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"cpu_port", "Port to QEMU CPU", {}}
    )

private:
    std::vector<uint8_t> memory_;  // Storage
    uint32_t latency_cycles_;
    uint32_t size_kb_;

    SST::Link* cpu_link_;
    SST::Statistic<uint64_t>* stat_reads_;
    SST::Statistic<uint64_t>* stat_writes_;
    SST::Output output_;

    void processRead(uint64_t addr, SST::Event* req);
    void processWrite(uint64_t addr, uint64_t data, uint32_t size, SST::Event* req);
};
```

**Implementation** (`MemoryDevice.cc`):

```cpp
MemoryDevice::MemoryDevice(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id)
{
    output_.init("MemoryDevice[@p:@l]: ", 1, 0, SST::Output::STDOUT);

    latency_cycles_ = params.find<uint32_t>("latency_cycles", 10);
    size_kb_ = params.find<uint32_t>("size_kb", 64);

    // Allocate memory
    memory_.resize(size_kb_ * 1024, 0);

    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    registerClock(clock_freq,
        new SST::Clock::Handler<MemoryDevice>(this, &MemoryDevice::clockTick));

    cpu_link_ = configureLink("cpu_port",
        new SST::Event::Handler<MemoryDevice>(this, &MemoryDevice::handleRequest));

    stat_reads_ = registerStatistic<uint64_t>("read_requests");
    stat_writes_ = registerStatistic<uint64_t>("write_requests");

    output_.verbose(CALL_INFO, 1, 0, "Initialized %u KB memory, %u-cycle latency\n",
        size_kb_, latency_cycles_);
}

void MemoryDevice::processRead(uint64_t addr, SST::Event* req) {
    if (addr >= memory_.size()) {
        output_.verbose(CALL_INFO, 1, 0, "Read out of bounds: 0x%lx\n", addr);
        CPUEvent* resp = new CPUEvent(CPUEvent::RESPONSE, addr, 0xDEADC0DE);
        cpu_link_->send(latency_cycles_, resp);
        return;
    }

    // Read 32-bit word
    uint32_t value = *((uint32_t*)&memory_[addr]);

    stat_reads_->addData(1);

    CPUEvent* resp = new CPUEvent(CPUEvent::RESPONSE, addr, value);
    cpu_link_->send(latency_cycles_, resp);
}

void MemoryDevice::processWrite(uint64_t addr, uint64_t data, uint32_t size, SST::Event* req) {
    if (addr + size > memory_.size()) {
        output_.verbose(CALL_INFO, 1, 0, "Write out of bounds: 0x%lx\n", addr);
        return;
    }

    // Write data
    if (size == 4) {
        *((uint32_t*)&memory_[addr]) = (uint32_t)data;
    } else if (size == 2) {
        *((uint16_t*)&memory_[addr]) = (uint16_t)data;
    } else if (size == 1) {
        memory_[addr] = (uint8_t)data;
    }

    stat_writes_->addData(1);

    CPUEvent* resp = new CPUEvent(CPUEvent::RESPONSE, addr, 0);
    cpu_link_->send(latency_cycles_, resp);
}

void MemoryDevice::handleRequest(SST::Event* event) {
    CPUEvent* cpu_ev = dynamic_cast<CPUEvent*>(event);

    if (cpu_ev->type == CPUEvent::READ) {
        processRead(cpu_ev->addr, event);
    } else if (cpu_ev->type == CPUEvent::WRITE) {
        processWrite(cpu_ev->addr, cpu_ev->data, 4, event);
    }

    delete event;
}
```

### SST Configuration for Heterogeneous System

**File**: `demo-configs/heterogeneous_test.py`

```python
#!/usr/bin/env python3
"""
Heterogeneous Multi-Device System Test

Demonstrates integration of 4 different device types:
- Echo device (simple register)
- Compute accelerator (vector operations)
- Memory device (simulated SRAM)
- DMA engine (bulk transfer)
"""

import sst

# Device configuration database
DEVICES = [
    {
        "type": "demodevices.EchoDevice",
        "name": "echo",
        "base": 0x10200000,
        "params": {"latency": 1}
    },
    {
        "type": "demodevices.ComputeAccelerator",
        "name": "compute",
        "base": 0x10300000,
        "params": {
            "latency_cycles": 100,
            "compute_units": 16,
            "vector_size": 256
        }
    },
    {
        "type": "demodevices.MemoryDevice",
        "name": "memory",
        "base": 0x10400000,
        "params": {
            "latency_cycles": 10,
            "size_kb": 64
        }
    },
    {
        "type": "demodevices.DMADevice",
        "name": "dma",
        "base": 0x10500000,
        "params": {
            "latency_cycles": 50,
            "max_transfer_size": 4096
        }
    }
]

# QEMU component
qemu = sst.Component("qemu", "qemubinary.QEMUBinary")
qemu_params = {
    "clock": "1GHz",
    "binary_path": "../riscv-programs/heterogeneous_test.elf",
    "qemu_path": "/home/user/qemu-build/qemu/build/qemu-system-riscv32",
    "num_devices": len(DEVICES),
    "verbose": 2
}

# Configure devices
for i, dev_cfg in enumerate(DEVICES):
    qemu_params[f"device{i}_base"] = f"0x{dev_cfg['base']:x}"
    qemu_params[f"device{i}_size"] = 4096
    qemu_params[f"device{i}_name"] = dev_cfg["name"]

qemu.addParams(qemu_params)

# Create device components
for i, dev_cfg in enumerate(DEVICES):
    dev = sst.Component(dev_cfg["name"], dev_cfg["type"])

    dev_params = {"clock": "1GHz"}
    dev_params.update(dev_cfg["params"])
    dev.addParams(dev_params)

    # Connect to QEMU
    link = sst.Link(f"link_{i}")
    link.connect(
        (qemu, f"device_port_{i}", "1ns"),
        (dev, "cpu_port", "1ns")
    )

# Enable statistics
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputConsole")

# Simulation settings
sst.setProgramOption("stop-at", "100ms")
```

### RISC-V Test Program for Heterogeneous System

**File**: `riscv-programs/heterogeneous_test.c`

```c
#include <stdint.h>

// Device base addresses
#define ECHO_BASE       0x10200000
#define COMPUTE_BASE    0x10300000
#define MEMORY_BASE     0x10400000
#define DMA_BASE        0x10500000

// Helper macro
#define REG(base, offset)  (*(volatile uint32_t*)((base) + (offset)))

// Test functions
int test_echo_device(void) {
    volatile uint32_t *echo_reg = (volatile uint32_t*)ECHO_BASE;

    // Write test pattern
    *echo_reg = 0xCAFEBABE;

    // Read back
    uint32_t value = *echo_reg;

    return (value == 0xCAFEBABE) ? 0 : 1;
}

int test_compute_device(void) {
    // Configure compute accelerator
    REG(COMPUTE_BASE, 0x08) = 0x80000000;  // SRC_A address
    REG(COMPUTE_BASE, 0x0C) = 0x80001000;  // SRC_B address
    REG(COMPUTE_BASE, 0x10) = 0x80002000;  // DST address
    REG(COMPUTE_BASE, 0x14) = 64;          // Vector size

    // Start computation
    REG(COMPUTE_BASE, 0x00) = 0x1;

    // Wait for completion
    uint32_t timeout = 1000000;
    while (timeout-- > 0) {
        uint32_t status = REG(COMPUTE_BASE, 0x04);
        if (status & 0x1) {  // Ready
            uint32_t result = REG(COMPUTE_BASE, 0x18);
            return (result == 0) ? 0 : 1;
        }
    }

    return 1;  // Timeout
}

int test_memory_device(void) {
    volatile uint32_t *mem = (volatile uint32_t*)MEMORY_BASE;

    // Write test data
    for (int i = 0; i < 16; i++) {
        mem[i] = i * 0x11111111;
    }

    // Read back and verify
    for (int i = 0; i < 16; i++) {
        if (mem[i] != i * 0x11111111) {
            return 1;  // Failure
        }
    }

    return 0;  // Success
}

int test_dma_device(void) {
    // Configure DMA transfer
    REG(DMA_BASE, 0x00) = 0x80000000;  // Source
    REG(DMA_BASE, 0x04) = 0x80010000;  // Destination
    REG(DMA_BASE, 0x08) = 1024;        // Size (bytes)

    // Start DMA
    REG(DMA_BASE, 0x0C) = 0x1;

    // Wait for completion
    uint32_t timeout = 1000000;
    while (timeout-- > 0) {
        uint32_t status = REG(DMA_BASE, 0x10);
        if (status & 0x1) {  // Complete
            return 0;
        }
    }

    return 1;  // Timeout
}

int main(void) {
    int failures = 0;

    // Test each device
    failures += test_echo_device();
    failures += test_compute_device();
    failures += test_memory_device();
    failures += test_dma_device();

    return failures;  // 0 = all passed
}
```

---

## Creating Custom Devices

### Device Development Checklist

1. **Define Device Registers** (memory map)
2. **Implement SST Component** (device.hh, device.cc)
3. **Handle MMIO Requests** (read/write operations)
4. **Add Latency Modeling** (cycle-accurate timing)
5. **Register Statistics** (performance monitoring)
6. **Create Test Program** (RISC-V code)
7. **Write SST Config** (Python configuration)
8. **Test and Validate**

### Template: Custom Device

**File**: `demo-devices/TemplateDevice.hh`

```cpp
#ifndef _TEMPLATE_DEVICE_HH
#define _TEMPLATE_DEVICE_HH

#include <sst/core/component.h>
#include <sst/core/link.h>

namespace DemoDevices {

class TemplateDevice : public SST::Component {
public:
    TemplateDevice(SST::ComponentId_t id, SST::Params& params);
    ~TemplateDevice();

    void setup() override;
    void finish() override;

    bool clockTick(SST::Cycle_t cycle);
    void handleRequest(SST::Event* event);

    // SST ELI macros
    SST_ELI_REGISTER_COMPONENT(
        TemplateDevice,
        "demodevices",           // Library name
        "TemplateDevice",        // Component name
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Template for custom devices",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"param1", "Description of param1", "default1"},
        {"param2", "Description of param2", "default2"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"cpu_port", "Port to QEMU CPU", {}}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"stat1", "Description of stat1", "units", 1},
        {"stat2", "Description of stat2", "units", 1}
    )

private:
    // Device registers (customize for your device)
    enum Registers {
        REG_CONTROL = 0x00,
        REG_STATUS  = 0x04,
        REG_DATA    = 0x08
    };

    // Device state
    struct {
        uint32_t control;
        uint32_t status;
        uint32_t data;
    } state_;

    // Configuration parameters
    uint32_t param1_;
    std::string param2_;

    // SST components
    SST::Link* cpu_link_;
    SST::Statistic<uint64_t>* stat1_;
    SST::Statistic<uint64_t>* stat2_;
    SST::Output output_;

    // Helper methods
    void processRead(uint64_t addr, SST::Event* req);
    void processWrite(uint64_t addr, uint64_t data, SST::Event* req);
};

} // namespace DemoDevices

#endif
```

**Implementation**:
```cpp
#include "TemplateDevice.hh"

using namespace DemoDevices;

TemplateDevice::TemplateDevice(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id)
{
    // Initialize output
    output_.init("TemplateDevice[@p:@l]: ", 1, 0, SST::Output::STDOUT);

    // Read parameters
    param1_ = params.find<uint32_t>("param1", 0);
    param2_ = params.find<std::string>("param2", "default");

    // Register clock
    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    registerClock(clock_freq,
        new SST::Clock::Handler<TemplateDevice>(this, &TemplateDevice::clockTick));

    // Configure link
    cpu_link_ = configureLink("cpu_port",
        new SST::Event::Handler<TemplateDevice>(this, &TemplateDevice::handleRequest));

    if (!cpu_link_) {
        output_.fatal(CALL_INFO, -1, "Failed to configure cpu_port\n");
    }

    // Register statistics
    stat1_ = registerStatistic<uint64_t>("stat1");
    stat2_ = registerStatistic<uint64_t>("stat2");

    // Initialize device state
    state_.control = 0;
    state_.status = 0x1;  // Ready
    state_.data = 0;

    output_.verbose(CALL_INFO, 1, 0, "Initialized with param1=%u, param2=%s\n",
        param1_, param2_.c_str());
}

void TemplateDevice::handleRequest(SST::Event* event) {
    CPUEvent* cpu_ev = dynamic_cast<CPUEvent*>(event);

    if (cpu_ev->type == CPUEvent::READ) {
        processRead(cpu_ev->addr, event);
    } else if (cpu_ev->type == CPUEvent::WRITE) {
        processWrite(cpu_ev->addr, cpu_ev->data, event);
    }

    delete event;
}

void TemplateDevice::processRead(uint64_t addr, SST::Event* req) {
    uint64_t data = 0;

    switch (addr & 0xFF) {
        case REG_CONTROL:
            data = state_.control;
            break;
        case REG_STATUS:
            data = state_.status;
            break;
        case REG_DATA:
            data = state_.data;
            stat1_->addData(1);  // Count reads
            break;
        default:
            data = 0xDEADBEEF;
    }

    CPUEvent* resp = new CPUEvent(CPUEvent::RESPONSE, addr, data);
    cpu_link_->send(resp);
}

void TemplateDevice::processWrite(uint64_t addr, uint64_t data, SST::Event* req) {
    switch (addr & 0xFF) {
        case REG_CONTROL:
            state_.control = data;
            break;
        case REG_DATA:
            state_.data = data;
            stat2_->addData(1);  // Count writes
            break;
    }

    CPUEvent* resp = new CPUEvent(CPUEvent::RESPONSE, addr, 0);
    cpu_link_->send(resp);
}

bool TemplateDevice::clockTick(SST::Cycle_t cycle) {
    // Add device-specific logic here
    return false;
}
```

### Makefile for Custom Devices

**File**: `demo-devices/Makefile`

```makefile
CXX = $(shell sst-config --CXX)
CXXFLAGS = $(shell sst-config --ELEMENT_CXXFLAGS)
CXXFLAGS += -std=c++14 -g -O2 -I.
LDFLAGS = $(shell sst-config --ELEMENT_LDFLAGS)

# All device sources
SOURCES = ComputeAccelerator.cc \
          EchoDevice.cc \
          MemoryDevice.cc \
          DMADevice.cc \
          TemplateDevice.cc \
          CPUEvent.cc

HEADERS = $(SOURCES:.cc=.hh)
OBJECTS = $(SOURCES:.cc=.o)
TARGET = libdemodevices.so

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -shared -o $@ $^

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

install: $(TARGET)
	install -D $(TARGET) $(shell sst-config --prefix)/lib/sstcore/$(TARGET)

clean:
	rm -f *.o $(TARGET)

.PHONY: all install clean
```

---

## RISC-V Program Development

### Linker Script

**File**: `riscv-programs/linker.ld`

```ld
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
    RAM : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS {
    .text : {
        *(.text.init)
        *(.text*)
    } > RAM

    .rodata : {
        *(.rodata*)
    } > RAM

    .data : {
        __data_start = .;
        *(.data*)
        __data_end = .;
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    . = ALIGN(16);
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);

    /DISCARD/ : {
        *(.comment)
        *(.eh_frame)
    }
}
```

### Startup Code

**File**: `riscv-programs/crt0.S`

```assembly
.section .text.init
.global _start

_start:
    # Disable interrupts
    csrci mstatus, 0x8

    # Set global pointer
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop

    # Set stack pointer
    la sp, _stack_top

    # Clear .bss
    la t0, __bss_start
    la t1, __bss_end
1:  bge t0, t1, 2f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 1b

    # Call main
2:  li a0, 0
    li a1, 0
    call main

    # Exit loop
    j .
```

### Device Driver Library

**File**: `riscv-programs/device_lib.h`

```c
#ifndef DEVICE_LIB_H
#define DEVICE_LIB_H

#include <stdint.h>

// Generic device register access
static inline uint32_t device_read(uint64_t base, uint32_t offset) {
    return *(volatile uint32_t*)(base + offset);
}

static inline void device_write(uint64_t base, uint32_t offset, uint32_t value) {
    *(volatile uint32_t*)(base + offset) = value;
}

// Wait for device ready
static inline int device_wait_ready(uint64_t base, uint32_t status_offset,
                                   uint32_t ready_bit, uint32_t timeout) {
    while (timeout-- > 0) {
        uint32_t status = device_read(base, status_offset);
        if (status & ready_bit) {
            return 0;  // Success
        }
    }
    return -1;  // Timeout
}

// Echo device wrapper
static inline uint32_t echo_device_test(uint64_t base, uint32_t value) {
    device_write(base, 0, value);
    return device_read(base, 0);
}

#endif // DEVICE_LIB_H
```

---

## Running and Testing

### Build and Test Workflow

```bash
# 1. Build custom devices
cd demo-devices
make clean && make && make install

# Verify installation
sst-info demodevices

# 2. Build RISC-V programs
cd ../riscv-programs
make quad_accel_test.elf
make heterogeneous_test.elf

# 3. Run homogeneous test
cd ../demo-configs
sst quad_accel_test.py

# 4. Run heterogeneous test
sst heterogeneous_test.py

# 5. View statistics
sst --output-config stats.cfg quad_accel_test.py
cat stats.csv
```

### Debugging Tips

**Enable Verbose Output**:
```python
qemu.addParams({"verbose": 3})  # Max verbosity
```

**Check Device Connections**:
```bash
# Monitor QEMU output
sst quad_accel_test.py 2>&1 | grep -i "device\|accel"

# Check socket connections
ls -la /tmp/qemu-sst-device*.sock
```

**GDB Debugging**:
```bash
# Debug RISC-V program
qemu-system-riscv32 -s -S -M virt -kernel test.elf &
riscv32-unknown-elf-gdb test.elf
(gdb) target remote :1234
(gdb) break main
(gdb) continue
```

---

## Customization Guide

### Adapting Examples for Your Use Case

#### 1. **Create Specialized Accelerator**

Modify `ComputeAccelerator` for specific operations:
- Change register layout for your algorithm
- Adjust latency model based on operation type
- Add operation-specific statistics

#### 2. **Implement Custom Memory Hierarchy**

Extend `MemoryDevice` with:
- Cache simulation (LRU, FIFO, etc.)
- Bank conflicts modeling
- Power/energy tracking

#### 3. **Build Multi-Device Pipelines**

Connect devices together:
```python
# Device 0 outputs to Device 1 input
pipeline_link = sst.Link("pipeline")
pipeline_link.connect(
    (device0, "output_port", "10ns"),
    (device1, "input_port", "10ns")
)
```

#### 4. **Add External Interfaces**

Extend devices with file I/O, network, etc.:
```cpp
// In your device
void loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    // Load configuration or data
}
```

---

## Summary

These examples provide complete, working code for:

**Homogeneous Systems**:
- 4 identical compute accelerators
- Parallel workload distribution
- Varying latencies for load balancing

**Heterogeneous Systems**:
- 4 different device types
- Specialized device interactions
- System-level simulation

**Key Takeaways**:
1. Device development follows standard SST patterns
2. RISC-V programs use memory-mapped I/O
3. SST Python configs enable flexible system composition
4. Statistics provide performance insights
5. Same device code works on single or multiple servers

**Next Steps**:
- Customize device behavior for your domain
- Add advanced features (interrupts, DMA, etc.)
- Scale to larger systems (8+, 16+ devices)
- Integrate with real applications

---

**Document Version**: 1.0
**Last Updated**: 2025-11-10
**Status**: Complete Demo Examples
