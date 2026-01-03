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

# Distributed SST Application Design Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Design](#component-design)
4. [Implementation Guide](#implementation-guide)
5. [Build System](#build-system)
6. [Testing and Debugging](#testing-and-debugging)
7. [Best Practices](#best-practices)

## Overview

This document explains how to design and implement a distributed SST (Structural Simulation Toolkit) application using MPI for inter-process communication. The example demonstrates a QEMU component communicating with an ACALSim device component across different MPI ranks.

### What You'll Learn
- How to create SST components in C++
- How to set up distributed simulation with MPI
- How to implement memory-mapped I/O communication
- How to build and test SST components

### Prerequisites
- SST-Core installed and configured
- MPI (OpenMPI or MPICH)
- C++17 compiler
- Basic understanding of discrete-event simulation

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Distributed SST Simulation               │
├──────────────────────────────┬──────────────────────────────┤
│         MPI Rank 0           │         MPI Rank 1           │
│  ┌────────────────────────┐  │  ┌────────────────────────┐  │
│  │   QEMU Component       │  │  │  ACALSim Device        │  │
│  │  - Test Program        │  │  │  - Echo Device         │  │
│  │  - Memory Transactions │  │  │  - Register Interface  │  │
│  │  - Clock: 1GHz         │  │  │  - Clock: 1GHz         │  │
│  └──────────┬─────────────┘  │  └──────────┬─────────────┘  │
│             │                │             │                │
│             └────────────────┼─────────────┘                │
│                  SST Link    │    (3-cycle latency)         │
│               (IPC via MPI)  │                              │
└──────────────────────────────┴──────────────────────────────┘
```

### Communication Flow

1. **QEMU Component (Rank 0)**:
   - Runs test program state machine
   - Issues memory transactions (LOAD/STORE)
   - Sends transactions via SST link

2. **ACALSim Device (Rank 1)**:
   - Receives memory transactions
   - Implements echo device with configurable latency
   - Returns responses via SST link

3. **SST Link**:
   - Transparent inter-process communication
   - Configurable latency (e.g., 3 cycles)
   - Marked with `setNoCut()` for distributed simulation

## Component Design

### Component Structure

Every SST component follows this pattern:

```cpp
class MyComponent : public SST::Component {
public:
    // Registration macro - CRITICAL for SST to find your component
    SST_ELI_REGISTER_COMPONENT(
        MyComponent,                    // Class name
        "library_name",                 // Library name (e.g., "qemu", "acalsim")
        "ComponentName",                // Component name in Python
        SST_ELI_ELEMENT_VERSION(1,0,0), // Version
        "Description",                  // Description
        COMPONENT_CATEGORY_PROCESSOR    // Category
    )

    // Parameter documentation
    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"verbose", "Verbosity level", "1"}
    )

    // Port documentation
    SST_ELI_DOCUMENT_PORTS(
        {"port_name", "Port description", {"event_types"}}
    )

    // Constructor
    MyComponent(SST::ComponentId_t id, SST::Params& params);

    // Destructor
    ~MyComponent();

    // Lifecycle methods
    void setup();
    void finish();

private:
    // Clock handler
    bool clockTick(SST::Cycle_t cycle);

    // Event handler
    void handleEvent(SST::Event* ev);

    // Member variables
    SST::Output out_;
    SST::Link* link_;
    SST::TimeConverter* tc_;
};
```

### Key Design Patterns

#### 1. Primary Component Pattern

For components that control simulation end:

```cpp
// In constructor
MyComponent::MyComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id) {

    // Tell SST this component controls simulation lifecycle
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();  // Not ready to end yet

    // ... rest of initialization
}

// In clock handler when done
bool MyComponent::clockTick(SST::Cycle_t cycle) {
    if (work_complete) {
        primaryComponentOKToEndSim();  // Signal ready to end
        return true;  // Stop clock
    }
    return false;  // Continue ticking
}
```

#### 2. Clock Handler Pattern

**CRITICAL**: Clock handler return value semantics:
- `return false` → Continue ticking (keep simulation running)
- `return true` → Stop clock (component done)

```cpp
bool MyComponent::clockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;

    // Process events
    processEvents();

    // Do work
    doWork();

    // Check if done
    bool is_done = (state_ == DONE);

    if (is_done) {
        primaryComponentOKToEndSim();
    }

    return is_done;  // false = continue, true = done
}
```

#### 3. Event Communication Pattern

```cpp
// Define custom event
class MyEvent : public SST::Event {
public:
    MyEvent(uint64_t data) : data_(data) {}

    uint64_t getData() const { return data_; }

    // Serialization for distributed simulation
    void serialize_order(SST::Core::Serialization::serializer &ser) override {
        Event::serialize_order(ser);
        ser & data_;
    }

    ImplementSerializable(MyEvent);

private:
    uint64_t data_;
};

// Send event
void MyComponent::sendEvent() {
    auto* event = new MyEvent(data);
    link_->send(event);  // SST takes ownership
}

// Receive event
void MyComponent::handleEvent(SST::Event* ev) {
    MyEvent* my_ev = dynamic_cast<MyEvent*>(ev);
    if (my_ev) {
        processData(my_ev->getData());
    }
    delete ev;  // We own the event now
}
```

## Implementation Guide

### Step 1: Define Your Events

Create events for communication between components:

```cpp
// MemoryTransactionEvent.hh
#pragma once
#include <sst/core/event.h>

enum class TransactionType { LOAD, STORE };

class MemoryTransactionEvent : public SST::Event {
public:
    MemoryTransactionEvent(TransactionType type, uint64_t addr,
                          uint32_t data, uint32_t size, uint64_t req_id)
        : type_(type), addr_(addr), data_(data), size_(size), req_id_(req_id) {}

    // Getters
    TransactionType getType() const { return type_; }
    uint64_t getAddress() const { return addr_; }
    uint32_t getData() const { return data_; }
    uint32_t getSize() const { return size_; }
    uint64_t getReqId() const { return req_id_; }

    // Serialization
    void serialize_order(SST::Core::Serialization::serializer &ser) override {
        Event::serialize_order(ser);
        ser & type_;
        ser & addr_;
        ser & data_;
        ser & size_;
        ser & req_id_;
    }

    ImplementSerializable(MemoryTransactionEvent);

private:
    TransactionType type_;
    uint64_t addr_;
    uint32_t data_;
    uint32_t size_;
    uint64_t req_id_;
};
```

### Step 2: Implement Component Header

```cpp
// MyComponent.hh
#pragma once
#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include "MemoryTransactionEvent.hh"

namespace MyNamespace {

class MyComponent : public SST::Component {
public:
    // Registration
    SST_ELI_REGISTER_COMPONENT(
        MyComponent,
        "mylib",
        "MyComponent",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "My custom component",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"verbose", "Verbosity level", "1"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"port", "Communication port", {"MemoryTransactionEvent"}}
    )

    // Constructor/Destructor
    MyComponent(SST::ComponentId_t id, SST::Params& params);
    ~MyComponent();

    // Lifecycle
    void setup() override;
    void finish() override;

private:
    // Handlers
    bool clockTick(SST::Cycle_t cycle);
    void handleEvent(SST::Event* ev);

    // State
    SST::Output out_;
    SST::Link* link_;
    SST::Cycle_t current_cycle_;
};

} // namespace MyNamespace
```

### Step 3: Implement Component Source

```cpp
// MyComponent.cc
#include "MyComponent.hh"

using namespace MyNamespace;

MyComponent::MyComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), current_cycle_(0) {

    // Initialize output
    int verbose = params.find<int>("verbose", 1);
    out_.init("MyComponent[@p:@l]: ", verbose, 0, SST::Output::STDOUT);

    // Get parameters
    std::string clock_freq = params.find<std::string>("clock", "1GHz");

    // Register clock
    registerClock(clock_freq,
        new SST::Clock::Handler<MyComponent>(this, &MyComponent::clockTick));

    // Configure link
    link_ = configureLink("port",
        new SST::Event::Handler<MyComponent>(this, &MyComponent::handleEvent));

    if (!link_) {
        out_.fatal(CALL_INFO, -1, "Failed to configure link\n");
    }

    // Primary component setup
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

MyComponent::~MyComponent() {
    // Cleanup
}

void MyComponent::setup() {
    out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");
}

void MyComponent::finish() {
    out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");
}

bool MyComponent::clockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;

    // Do work here

    return false;  // Continue ticking
}

void MyComponent::handleEvent(SST::Event* ev) {
    auto* trans = dynamic_cast<MemoryTransactionEvent*>(ev);
    if (trans) {
        // Process transaction
    }
    delete ev;
}
```

### Step 4: Create Python Configuration

```python
import sst

# Configure distributed simulation
sst.setProgramOption("partitioner", "self")  # Manual partitioning
sst.setProgramOption("timebase", "1ps")      # Prevent overflow
sst.setProgramOption("stop-at", "100us")     # Max simulation time

# Get MPI info
rank = sst.getMyMPIRank()
nranks = sst.getMPIRankCount()

# Create components (ALL ranks create ALL components)
comp1 = sst.Component("comp1", "mylib.MyComponent")
comp1.setRank(0)  # Assign to rank 0
comp1.addParams({
    "clock": "1GHz",
    "verbose": 2
})

comp2 = sst.Component("comp2", "mylib.MyComponent")
comp2.setRank(1)  # Assign to rank 1
comp2.addParams({
    "clock": "1GHz",
    "verbose": 2
})

# Create and connect link
link = sst.Link("comp_link")
link.setNoCut()  # Don't cut for distributed simulation
comp1.addLink(link, "port", "3ns")  # 3ns latency
comp2.addLink(link, "port", "3ns")

print(f"Configuration complete on rank {rank}")
```

### Step 5: Create Makefile

```makefile
# Get SST configuration
CXX = $(shell sst-config --CXX)
CXXFLAGS = $(shell sst-config --ELEMENT_CXXFLAGS) -std=c++17 -Wall -Wextra
LDFLAGS = $(shell sst-config --ELEMENT_LDFLAGS)

# SST installation
SST_INSTALL_DIR = $(shell sst-config --prefix)
SST_ELEMENT_DIR = $(SST_INSTALL_DIR)/lib/sstcore

# Component
COMPONENT_NAME = libmylib.so
SOURCES = MyComponent.cc
OBJECTS = $(SOURCES:.cc=.o)
HEADERS = MyComponent.hh MemoryTransactionEvent.hh

# Build
all: $(COMPONENT_NAME)

$(COMPONENT_NAME): $(OBJECTS)
	$(CXX) $(LDFLAGS) -shared -fPIC -o $@ $^

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Install
install: $(COMPONENT_NAME)
	@mkdir -p $(SST_ELEMENT_DIR)
	@cp $(COMPONENT_NAME) $(SST_ELEMENT_DIR)/

# Clean
clean:
	rm -f $(OBJECTS) $(COMPONENT_NAME)

.PHONY: all install clean
```

## Build System

### Directory Structure

```
my-sst-app/
├── component1/
│   ├── Component1.hh
│   ├── Component1.cc
│   └── Makefile
├── component2/
│   ├── Component2.hh
│   ├── Component2.cc
│   └── Makefile
├── config/
│   └── simulation.py
├── build.sh
└── README.md
```

### Build Script

```bash
#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Building SST components...${NC}"

# Build component 1
cd component1
make clean && make && make install
cd ..

# Build component 2
cd component2
make clean && make && make install
cd ..

echo -e "${GREEN}Verifying installation...${NC}"

# Verify
if sst-info mylib > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Components registered${NC}"
else
    echo -e "${RED}✗ Components not found${NC}"
    exit 1
fi

# Run simulation
echo -e "${GREEN}Running simulation...${NC}"
cd config
mpirun -n 2 sst simulation.py
```

## Testing and Debugging

### Debugging Techniques

1. **Verbose Output**:
```cpp
out_.verbose(CALL_INFO, 1, 0, "Debug message: value=%d\n", value);
```

2. **State Logging**:
```cpp
out_.verbose(CALL_INFO, 1, 0, "[CLOCK] Cycle %lu, State: %d\n",
             cycle, static_cast<int>(state_));
```

3. **Transaction Tracking**:
```cpp
// Track pending transactions with unique IDs
std::map<uint64_t, PendingTransaction> pending_transactions_;
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Clock ticks once | Wrong return value | Return `false` to continue |
| Element not found | Wrong install path | Use `lib/sstcore` not `lib/sst-elements-library` |
| Dangling link | Conditional component creation | All ranks must create all components |
| API error | Wrong function name | Use `sst.getMyMPIRank()` not `getMPIRank()` |
| Compiler not found | Full command with flags | Extract first word: `CXX=$(echo $CXX_FULL \| awk '{print $1}')` |

### Verification Checklist

- [ ] Components compile without errors
- [ ] Libraries install to `lib/sstcore`
- [ ] `sst-info mylib` shows components
- [ ] All ranks create all components
- [ ] Links marked with `setNoCut()`
- [ ] Clock handlers return correct values
- [ ] Primary component API used correctly
- [ ] Events properly serialized

## Best Practices

### 1. Component Design

**DO**:
- Use primary component API for lifecycle control
- Return `false` from clock handler to continue
- Delete events after handling (you own them)
- Use verbose output for debugging
- Track pending transactions with IDs

**DON'T**:
- Return `true` from clock handler unless done
- Forget to mark links with `setNoCut()`
- Create components conditionally based on rank
- Forget serialization for distributed events
- Use malloc/free (use new/delete)

### 2. Memory Management

```cpp
// Sending event - SST takes ownership
void sendEvent() {
    auto* ev = new MyEvent(data);
    link_->send(ev);  // DON'T delete
}

// Receiving event - you own it
void handleEvent(SST::Event* ev) {
    processEvent(ev);
    delete ev;  // YOU must delete
}
```

### 3. Distributed Simulation

```python
# Python configuration for distributed simulation

# REQUIRED: Use "self" partitioner
sst.setProgramOption("partitioner", "self")

# ALL ranks must execute same Python script
comp1 = sst.Component("comp1", "lib.Comp1")
comp1.setRank(0)  # Explicitly assign rank

comp2 = sst.Component("comp2", "lib.Comp2")
comp2.setRank(1)  # Explicitly assign rank

# Mark links for distribution
link = sst.Link("link")
link.setNoCut()  # CRITICAL for distributed simulation
```

### 4. Clock Synchronization

```cpp
// Both components at same frequency
// QEMU: 1GHz
// Device: 1GHz
// This ensures cycle-accurate synchronization
```

### 5. Error Handling

```cpp
// Fatal errors
if (!link_) {
    out_.fatal(CALL_INFO, -1, "Failed to configure link\n");
}

// Warnings
if (pending_transactions_.find(id) == pending_transactions_.end()) {
    out_.verbose(CALL_INFO, 2, 0, "Warning: Unknown request %lu\n", id);
}
```

## Example: Echo Device

See the complete implementation in:
- `qemu-component/QEMUComponent.{hh,cc}` - Test program simulator
- `acalsim-device/ACALSimDeviceComponent.{hh,cc}` - Echo device
- `config/echo_device.py` - Distributed configuration

Key features demonstrated:
- Memory-mapped register interface
- Cycle-accurate latency modeling
- Transaction tracking with unique IDs
- State machine implementation
- Distributed communication

## Running the Simulation

```bash
# Build and install
./build.sh install

# Run distributed simulation
cd config
mpirun -n 2 sst echo_device.py

# Expected output
✓ Test iteration 1 PASSED (read=0xdeadbeef)
✓ Test iteration 2 PASSED (read=0xdeadbef0)
...
*** TEST PASSED ***
```

## Next Steps

1. **Extend with Real QEMU**: Replace test program with actual QEMU integration
2. **Add More Devices**: Create additional device components
3. **Performance Modeling**: Add detailed timing models
4. **Statistics**: Use SST statistics API for performance metrics
5. **Checkpointing**: Implement state save/restore

## References

- [SST Documentation](http://sst-simulator.org)
- [SST-Core GitHub](https://github.com/sstsimulator/sst-core)
- [MPI Documentation](https://www.open-mpi.org)
- This implementation: `src/qemu-sst/`

## Support

For issues or questions:
1. Check the build output for error messages
2. Verify SST installation with `sst-info`
3. Enable verbose output (`verbose=3`)
4. Review this design document
5. Examine the working example code
