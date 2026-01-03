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

# SST Integration Guide for ACALSim Applications

This guide explains how to integrate ACALSim applications with the Structural Simulation Toolkit (SST), enabling your event-driven simulators to run within SST's large-scale architecture simulation framework.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Integration](#architecture-integration)
3. [Completed Integrations](#completed-integrations)
4. [Integration Pattern](#integration-pattern)
5. [Step-by-Step Integration Guide](#step-by-step-integration-guide)
6. [Candidate Applications](#candidate-applications)
7. [Build and Test](#build-and-test)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

### What is SST?

The **Structural Simulation Toolkit (SST)** is a framework for simulating large-scale computing systems. It provides:
- **Cycle-based discrete-event simulation** engine
- **Component-based architecture** for modular system design
- **Link-based communication** between components
- **Python configuration** for flexible system composition
- **Parallel simulation** capabilities for large-scale models

### Why Integrate ACALSim with SST?

ACALSim provides powerful **event-driven simulation** capabilities with:
- 2-phase execution model (Phase 1: parallel execution, Phase 2: synchronization)
- Channel and port-based communication
- Configuration management
- Resource recycling and memory optimization

By integrating ACALSim applications with SST, you can:
- ✅ Leverage SST's ecosystem of architecture components
- ✅ Build large-scale system simulations combining multiple ACALSim simulators
- ✅ Use SST's parallel simulation infrastructure
- ✅ Access SST's statistics and visualization tools
- ✅ Integrate with existing SST models (memory systems, networks, etc.)

---

## Architecture Integration

### ACALSim Event-Driven Model

ACALSim uses a **2-phase event-driven architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                    SimTop                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │              ThreadManager                         │  │
│  │  ┌──────────────┐          ┌──────────────┐      │  │
│  │  │   Phase 1    │          │   Phase 2    │      │  │
│  │  │  (Parallel)  │    →     │    (Sync)    │      │  │
│  │  │              │          │              │      │  │
│  │  │ - Simulator  │          │ - Channel    │      │  │
│  │  │   execution  │          │   toggle     │      │  │
│  │  │ - Event      │          │ - PipeReg    │      │  │
│  │  │   handling   │          │   update     │      │  │
│  │  │              │          │ - Fast fwd   │      │  │
│  │  └──────────────┘          └──────────────┘      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- **Variable time steps**: Can skip empty ticks via fast-forward
- **Event-driven**: Simulators execute only when events are present
- **2-phase synchronization**: Ensures deterministic parallel execution
- **Global tick**: All simulators synchronized to a common time base

### SST Cycle-Based Model

SST uses a **clock-driven cycle-based architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                    SST Core                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Clock Handler                         │  │
│  │                                                    │  │
│  │  Every clock cycle:                               │  │
│  │    1. Call component's clockTick()                │  │
│  │    2. Component returns:                          │  │
│  │       - false → continue clock                    │  │
│  │       - true  → stop clock                        │  │
│  │                                                    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- **Fixed time steps**: Every clock tick advances by one cycle
- **Clock-driven**: Components execute on every clock tick
- **Synchronous**: All components advance together
- **Primary component lifecycle**: Components control simulation termination

### Integration Mapping

The integration bridges these two models:

```
┌──────────────────────────────────────────────────────────────┐
│            SST Component Wrapper                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         clockTick() (called every SST cycle)           │  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │        ACALSim SimTop Instance                   │ │  │
│  │  │                                                  │ │  │
│  │  │  Phase 1: startPhase1() → finishPhase1()       │ │  │
│  │  │  Phase 2: startPhase2() → finishPhase2()       │ │  │
│  │  │           - Fast-forward N ticks                │ │  │
│  │  │           - Channel toggle                      │ │  │
│  │  │           - Check completion                    │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │                                                        │  │
│  │  Return: false (continue) or true (stop)              │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Mapping Details:**
- **1 SST cycle = 1 ACALSim iteration** (may span multiple ACALSim ticks)
- **clockTick() executes both Phase 1 and Phase 2** of ACALSim iteration
- **Fast-forward** allows ACALSim to skip empty cycles
- **2-phase termination** ensures clean shutdown

---

## Completed Integrations

### 1. RISCVSoCStandalone

**Status:** ✅ **FULLY WORKING**

**Description:** Complete RISC-V RV32I ISA simulator with pipeline stages

**Components:**
- `SOC` - System-on-Chip orchestrator
- `CPU` - Single-cycle RV32I processor
- `IFStage` - Instruction Fetch timing model
- `EXEStage` - Execution timing model
- `WBStage` - Writeback timing model
- `DataMemory` - Data memory model
- Pipe registers for inter-stage communication

**SST Component:** `RISCVSoCStandalone`

**Files:**
- `RISCVSoCStandalone.hh` - Component header
- `RISCVSoCStandalone.cc` - Component implementation
- `examples/riscv_single_core.py` - SST Python configuration

**Usage:**
```bash
# Build and install
cd src/sst-riscv
make && make install

# Run RISC-V simulation
cd examples
sst riscv_single_core.py
```

**Configuration Parameters:**
- `clock` - Clock frequency (default: 1GHz)
- `asm_file` - Path to RISC-V assembly file
- `memory_size` - Memory size in bytes (default: 65536)
- `text_offset` - Text segment offset (default: 0)
- `data_offset` - Data segment offset (default: 8192)
- `max_cycles` - Maximum SST cycles (0=unlimited)
- `verbose` - Verbosity level (0-3)

**Example Output:**
```
Tick=1 Info: [CPU] Instruction ADDI is completed at Tick = 1 | PC = 0
Tick=2 Info: [CPU] Instruction ADDI is completed at Tick = 2 | PC = 4
Tick=3 Info: [CPU] Instruction ADD is completed at Tick = 3 | PC = 8
Tick=4 Info: [CPU] Instruction HCF is completed at Tick = 4 | PC = 12
...
Simulation is complete, simulated time: 8 ns
```

---

## Integration Pattern

### SST Component Lifecycle

SST components go through these phases:

1. **Construction** (`Constructor`)
   - Register as primary component
   - Call `primaryComponentDoNotEndSim()`
   - Register clock handler
   - Create SimTop instance (defer initialization)

2. **Init** (`init()`)
   - Initialize SimTop
   - Load configuration
   - Register simulators, channels, ports

3. **Setup** (`setup()`)
   - Final setup before simulation starts
   - Can establish links with other components

4. **Run** (`clockTick()`)
   - Called every SST cycle
   - Execute ACALSim Phase 1 & Phase 2
   - Return `false` to continue, `true` to stop

5. **Finish** (`finish()`)
   - Cleanup resources
   - Print statistics
   - Note: May need to leak SimTop to avoid destruction crashes

### SST Clock Handler Return Values

⚠️ **CRITICAL**: SST clock handler return values are **counter-intuitive**:

```cpp
bool clockTick(Cycle_t cycle) {
    // ... do work ...

    if (should_continue) {
        return false;  // FALSE = continue clock
    } else {
        return true;   // TRUE = stop clock
    }
}
```

- **`false`** → Continue clock (keep simulation running)
- **`true`** → Stop clock (end simulation)

This is the **opposite** of typical boolean conventions!

### Primary Component Pattern

To prevent SST from ending simulation prematurely:

```cpp
// In constructor:
registerAsPrimaryComponent();
primaryComponentDoNotEndSim();  // Tells SST we're not done yet

// When ready to end:
primaryComponentOKToEndSim();   // Tells SST we can end now
return true;                     // Stop clock
```

### 2-Phase Termination

ACALSim requires **2 iterations** to terminate cleanly:

```cpp
// Iteration N: Detect completion
if (threadManager->isAllSimulatorDone()) {
    threadManager->issueExitEvent(next_tick);
    ready_to_terminate_ = true;     // Flag for next iteration
    primaryComponentOKToEndSim();   // Tell SST we can end
    // Continue to next iteration (return false)
}

// Iteration N+1: Actually terminate
if (ready_to_terminate_) {
    return true;  // Stop clock now
}
```

**Why 2 phases?**
- Phase N: ACALSim issues ExitEvent for cleanup
- Phase N+1: ACALSim processes ExitEvent and terminates threads
- Premature termination causes crashes in RecycleContainer destructors

---

## Step-by-Step Integration Guide

### Step 1: Analyze the Application

Before integrating, understand your ACALSim application:

1. **Identify the SimTop class**
   - What is the top-level simulation class?
   - Example: `SOCTop` for RISC-V

2. **Identify configuration parameters**
   - What parameters does it need?
   - Example: `asm_file_path`, `memory_size`

3. **Identify simulators**
   - What `SimBase` subclasses are used?
   - How are they connected (channels/ports)?

4. **Check initialization**
   - Does it use `registerConfigs()`?
   - Does it use `registerCLIArguments()`?
   - Does it use `registerSimulators()`?

### Step 2: Create SST Component Header

Create `YourComponentStandalone.hh`:

```cpp
#ifndef __YOUR_COMPONENT_STANDALONE_HH__
#define __YOUR_COMPONENT_STANDALONE_HH__

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

// Forward declare your SimTop class
class YourSimTop;

namespace ACALSim {
namespace SSTIntegration {

class YourComponentStandalone : public ::SST::Component {
public:
    // SST ELI (Element Library Interface) registration
    SST_ELI_REGISTER_COMPONENT(
        YourComponentStandalone,
        "acalsim",              // Library name
        "YourComponentStandalone",  // Component name
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Description of your component",
        COMPONENT_CATEGORY_PROCESSOR  // Or appropriate category
    )

    // Document parameters
    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"param1", "Description of param1", "default_value"},
        {"param2", "Description of param2", "default_value"},
        // ... add all your parameters
    )

    // Constructor & Destructor
    YourComponentStandalone(::SST::ComponentId_t id, ::SST::Params& params);
    ~YourComponentStandalone() override;

    // SST lifecycle methods
    void setup() override;
    void init(unsigned int phase) override;
    void finish() override;

    // Clock tick handler
    bool clockTick(::SST::Cycle_t cycle);

private:
    ::SST::Output out_;
    ::SST::TimeConverter* tc_;

    std::shared_ptr<YourSimTop> sim_top_;

    uint64_t current_cycle_;
    uint64_t max_cycles_;
    bool ready_to_terminate_;

    // Add your config file / parameter storage
    std::string config_file_;
    std::string param1_;
    int param2_;
    // ...
};

} // namespace SSTIntegration
} // namespace ACALSim

#endif
```

### Step 3: Implement SST Component

Create `YourComponentStandalone.cc`:

```cpp
#include "YourComponentStandalone.hh"
#include "YourSimTop.hh"  // Your SimTop header
#include <fstream>

using namespace ACALSim::SSTIntegration;

YourComponentStandalone::YourComponentStandalone(
    ::SST::ComponentId_t id, ::SST::Params& params)
    : ::SST::Component(id),
      current_cycle_(0),
      max_cycles_(0),
      ready_to_terminate_(false)
{
    // Set up output
    out_.init("YourComponent-" + getName() + "-> ", 1, 0, ::SST::Output::STDOUT);

    // Get parameters
    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    config_file_ = params.find<std::string>("config_file", "");
    param1_ = params.find<std::string>("param1", "default");
    param2_ = params.find<int>("param2", 0);
    max_cycles_ = params.find<uint64_t>("max_cycles", 0);

    // Create temporary config file if needed
    if (config_file_.empty()) {
        config_file_ = "/tmp/sst_your_component_config.json";
        std::ofstream config(config_file_);
        config << "{\n";
        config << "  \"YourConfig\": {\n";
        config << "    \"param1\": \"" << param1_ << "\",\n";
        config << "    \"param2\": " << param2_ << "\n";
        config << "  }\n";
        config << "}\n";
        config.close();
    }

    // Create SimTop instance (don't initialize yet!)
    sim_top_ = std::make_shared<YourSimTop>("YourSimTop", config_file_);

    // Configure single-threaded mode for SST integration
    sim_top_->setSingleThreadedMode();

    // Register as primary component and tell SST not to end yet
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();

    // Register clock handler
    tc_ = registerClock(clock_freq,
        new ::SST::Clock::Handler2<YourComponentStandalone,
            &YourComponentStandalone::clockTick>(this));

    out_.verbose(CALL_INFO, 1, 0, "Component initialized\n");
}

YourComponentStandalone::~YourComponentStandalone() {
    out_.verbose(CALL_INFO, 1, 0, "Destroying component\n");
}

void YourComponentStandalone::init(unsigned int phase) {
    if (phase == 0) {
        out_.verbose(CALL_INFO, 1, 0, "Initializing SimTop\n");
        sim_top_->init();
    }
}

void YourComponentStandalone::setup() {
    out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");
}

bool YourComponentStandalone::clockTick(::SST::Cycle_t cycle) {
    try {
        current_cycle_++;

        // Check if we need to terminate
        if (ready_to_terminate_) {
            out_.verbose(CALL_INFO, 1, 0, "Terminating simulation\n");
            return true;  // TRUE = stop clock
        }

        // First iteration: start thread manager
        if (current_cycle_ == 1) {
            sim_top_->startSimThreadsPublic();
            sim_top_->startRunning();
        }

        // Execute ACALSim Phase 1 (parallel execution)
        sim_top_->startPhase1();
        sim_top_->finishPhase1();

        // Execute ACALSim Phase 2 (synchronization)
        sim_top_->startPhase2();

        // Check if all simulators are done
        acalsim::Tick current_tick = sim_top_->getGlobalTick();
        bool all_done = sim_top_->isAllSimulatorDone();

        if (all_done) {
            // ACALSim 2-phase termination
            acalsim::Tick next_tick = sim_top_->getFastForwardCycles();
            sim_top_->issueExitEvent(next_tick);
            ready_to_terminate_ = true;
            primaryComponentOKToEndSim();
        } else {
            // Fast-forward to next event
            acalsim::Tick next_tick = sim_top_->getFastForwardCycles();
            sim_top_->fastForwardGlobalTick(next_tick);
        }

        // Channel toggle and finish Phase 2
        SimChannelGlobal::toggleChannelDualQueueStatus();
        sim_top_->finishPhase2();

        // Check max cycles
        if (max_cycles_ > 0 && current_cycle_ >= max_cycles_) {
            out_.verbose(CALL_INFO, 1, 0, "Max cycles reached\n");
            primaryComponentOKToEndSim();
            return true;  // TRUE = stop clock
        }

        return false;  // FALSE = continue clock

    } catch (const std::exception& e) {
        out_.fatal(CALL_INFO, -1, "Exception: %s\n", e.what());
        return true;  // Stop on error
    }
}

void YourComponentStandalone::finish() {
    out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");

    // WORKAROUND: Leak SimTop to avoid destruction crashes
    if (sim_top_) {
        new std::shared_ptr<YourSimTop>(sim_top_);
        sim_top_.reset();
    }
    if (acalsim::top) {
        new std::shared_ptr<acalsim::SimTopBase>(acalsim::top);
        acalsim::top.reset();
    }

    // Clean up temp config
    if (config_file_.find("/tmp/") == 0) {
        std::remove(config_file_.c_str());
    }

    out_.output(CALL_INFO, "Simulation Complete!\n");
    out_.output(CALL_INFO, "Total SST cycles: %lu\n", current_cycle_);
}
```

### Step 4: Update Makefile

Add your component to `src/sst-riscv/Makefile`:

```makefile
# Add to SOURCES
SOURCES = ACALSimComponent.cc \
          RISCVSoCStandalone.cc \
          YourComponentStandalone.cc

# Add dependency
YourComponentStandalone.o: YourComponentStandalone.cc YourComponentStandalone.hh
```

### Step 5: Create Python Configuration

Create `examples/your_component_example.py`:

```python
import sst

# Create component
comp = sst.Component("your_comp", "acalsim.YourComponentStandalone")

# Set parameters
comp.addParams({
    "clock": "1GHz",
    "param1": "value1",
    "param2": 42,
    "max_cycles": 100000,
    "verbose": 2
})

# Set statistics
comp.enableAllStatistics({"type":"sst.AccumulatorStatistic"})

# Print configuration
print("=" * 60)
print("Your Component SST Configuration")
print("=" * 60)
print("Running simulation...")
```

### Step 6: Build and Test

```bash
# Build
cd src/sst-riscv
make clean && make && make install

# Verify installation
sst-info acalsim

# Run test
cd examples
sst your_component_example.py
```

---

## Candidate Applications

### Applications Ready for Integration

These ACALSim applications are good candidates for SST integration:

#### 1. **riscv** - Complete RISC-V Simulator
- **Status:** ✅ Already integrated as `RISCVSoCStandalone`
- **Complexity:** High
- **SimTop:** `SOCTop`
- **Components:** CPU, IFStage, EXEStage, WBStage, DataMemory
- **Use Case:** Architecture research, ISA simulation

#### 2. **riscvSimTemplate** - Simplified RISC-V Template
- **Status:** 🟡 Not yet integrated
- **Complexity:** Medium
- **SimTop:** `SOCTop`
- **Components:** CPU (simplified), DataMemory
- **Use Case:** Educational, learning ACALSim
- **Integration Effort:** Low (similar to riscv but simpler)

#### 3. **testSimPort** - Port Communication Example
- **Status:** 🟡 Not yet integrated
- **Complexity:** Low
- **SimTop:** Custom `SimTop`
- **Components:** CPUCore, CrossBar, Memory
- **Use Case:** Testing port-based communication
- **Integration Effort:** Very Low
- **Value:** Demonstrates SST component interconnection

#### 4. **testConfig** - Configuration System Demo
- **Status:** 🟡 Not yet integrated
- **Complexity:** Low
- **SimTop:** `STSim` template
- **Components:** SimpleProcessor
- **Use Case:** Configuration testing
- **Integration Effort:** Very Low
- **Value:** Shows SST parameter passing

#### 5. **testPETile** - Processing Element Tile
- **Status:** 🟡 Not yet integrated
- **Complexity:** Medium-High
- **Components:** PE tiles, interconnect
- **Use Case:** Multi-core/many-core architectures
- **Integration Effort:** Medium
- **Value:** Demonstrates multi-component SST systems

#### 6. **testBlackBear** - Accelerator Example
- **Status:** 🟡 Not yet integrated
- **Complexity:** Medium-High
- **Components:** Custom accelerator logic
- **Use Case:** Hardware accelerator modeling
- **Integration Effort:** Medium
- **Value:** Shows domain-specific accelerators in SST

### Applications Not Suitable for SST Integration

These are internal tests and templates:

- ❌ **test** - Basic framework tests
- ❌ **testAccelerator** - Internal accelerator tests
- ❌ **testChannel** - Channel infrastructure tests
- ❌ **testCommunication** - Communication tests
- ❌ **testResourceRecycling** - Memory management tests
- ❌ **testSimChannel** - SimChannel tests
- ❌ **testSTSim** - STSim template tests
- ❌ **testSTSystemC** - SystemC integration tests
- ❌ **ProjectTemplate** - Template for new projects

---

## Build and Test

### Prerequisites

1. **SST-Core** installed
2. **ACALSim** built (static library required)
3. **PyTorch/LibTorch** (for RISC-V components)
4. **C++20 compiler**

### Build Process

```bash
# Navigate to SST integration directory
cd src/sst-riscv

# Clean previous builds
make clean

# Build SST element library
make

# Install to SST
make install

# Verify installation
sst-info acalsim
```

### Running Tests

```bash
# Navigate to examples
cd examples

# Run RISC-V single-core example
sst riscv_single_core.py

# Run with verbose output
sst --verbose riscv_single_core.py

# Run with debugging
sst --debug-file=debug.txt riscv_single_core.py
```

### Expected Output

Successful simulation output:

```
Creating RISC-V RV32I Single-Core System...
============================================================
RISC-V RV32I Single-Core Configuration
============================================================
Clock frequency:     1GHz
Max cycles:          100000
Memory size:         65536 bytes (64KB)
...

Tick=1 Info: [CPU] Instruction ADDI is completed at Tick = 1
Tick=2 Info: [CPU] Instruction ADDI is completed at Tick = 2
...

=== RISC-V Simulation Complete ===
Total cycles: X
==================================
Simulation is complete, simulated time: X ns
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Double-Free Crash on Termination

**Symptom:**
```
free(): double free detected in tcache 2
Signal: Aborted (6)
```

**Cause:** SST's Exit class conflicts with ACALSim's RecycleContainer cleanup

**Solution:**
```cpp
void finish() {
    // Leak SimTop to avoid destructor crashes
    if (sim_top_) {
        new std::shared_ptr<YourSimTop>(sim_top_);
        sim_top_.reset();
    }
    if (acalsim::top) {
        new std::shared_ptr<acalsim::SimTopBase>(acalsim::top);
        acalsim::top.reset();
    }
}
```

#### 2. Event Queue Empty - Immediate Exit

**Symptom:**
```
*** Event queue empty, exiting simulation... ***
```

**Cause:** Not registering as primary component or wrong return values

**Solution:**
```cpp
// In constructor:
registerAsPrimaryComponent();
primaryComponentDoNotEndSim();

// In clockTick():
return false;  // FALSE = continue (not true!)
```

#### 3. Premature Termination

**Symptom:** Simulation ends before completion, no proper cleanup

**Cause:** Not implementing 2-phase termination

**Solution:**
```cpp
if (sim_top_->isAllSimulatorDone()) {
    sim_top_->issueExitEvent(next_tick);
    ready_to_terminate_ = true;  // Don't terminate yet!
    primaryComponentOKToEndSim();
    // Return false to continue one more iteration
}
```

#### 4. Symbol Resolution Errors

**Symptom:**
```
undefined symbol: _ZN7acalsim...
```

**Cause:** RTLD_LOCAL plugin loading doesn't export symbols

**Solution:** Use static linking with `--whole-archive`:
```makefile
$(CXX) -shared -o $@ $^ \
    -Wl,--whole-archive $(ACALSIM_LIB_STATIC)/libacalsim.a -Wl,--no-whole-archive
```

#### 5. Threading Conflicts

**Symptom:** Crashes or hangs during simulation

**Cause:** ACALSim worker threads conflict with SST's single-threaded model

**Solution:**
```cpp
// Before init():
sim_top_->setSingleThreadedMode();
```

#### 6. Clock Handler Return Value Confusion

**Symptom:** Simulation ends immediately or never ends

**Cause:** Incorrect return values (SST is counter-intuitive!)

**Remember:**
```cpp
return false;  // FALSE = continue clock ✓
return true;   // TRUE = stop clock ✓
```

---

## Best Practices

### 1. Component Design

✅ **DO:**
- Keep SST wrapper thin - let SimTop do the work
- Use descriptive parameter names
- Document all ELI parameters
- Handle errors gracefully with try-catch
- Provide reasonable default values

❌ **DON'T:**
- Duplicate logic from SimTop in SST component
- Create threads in SST component (use SimTop's threading)
- Modify global state directly

### 2. Configuration Management

✅ **DO:**
- Generate temporary JSON config from SST parameters
- Clean up temporary config files in `finish()`
- Support both config file and direct parameters
- Validate parameters in constructor

❌ **DON'T:**
- Hardcode configuration values
- Leave temporary files behind
- Ignore parameter validation

### 3. Lifecycle Management

✅ **DO:**
- Initialize SimTop in `init()` phase 0
- Start thread manager in first `clockTick()`
- Implement 2-phase termination correctly
- Leak SimTop in `finish()` to avoid crashes

❌ **DON'T:**
- Initialize SimTop in constructor
- Start threads before SST is ready
- Terminate immediately when done detected

### 4. Debugging and Logging

✅ **DO:**
- Use SST's `Output` class for logging
- Provide verbose levels for different detail
- Include helpful debug messages
- Log important state transitions

❌ **DON'T:**
- Use `std::cout` directly (breaks SST output)
- Log on every cycle (performance impact)
- Suppress all error messages

### 5. Error Handling

✅ **DO:**
- Wrap `clockTick()` in try-catch
- Use `out_.fatal()` for unrecoverable errors
- Validate all inputs
- Provide meaningful error messages

❌ **DON'T:**
- Let exceptions propagate to SST
- Silently ignore errors
- Use assert() in production code

### 6. Performance Optimization

✅ **DO:**
- Minimize work in `clockTick()` overhead
- Use fast-forward to skip empty cycles
- Leverage ACALSim's event-driven nature
- Profile with SST's statistics

❌ **DON'T:**
- Do unnecessary work every cycle
- Copy large data structures
- Allocate memory in hot paths

---

## Next Steps

### Recommended Integration Order

1. ✅ **riscv** - Already complete, use as reference
2. 🎯 **testSimPort** - Simple, demonstrates port communication
3. 🎯 **riscvSimTemplate** - Similar to riscv but simpler
4. 🎯 **testConfig** - Shows configuration patterns
5. 🔜 **testPETile** - Multi-core architecture
6. 🔜 **testBlackBear** - Accelerator modeling

### Contributing

To add your integration:

1. Follow the integration pattern above
2. Add component `.hh` and `.cc` files
3. Update `Makefile`
4. Create Python configuration example
5. Test thoroughly
6. Document parameters and usage
7. Submit pull request

### Resources

- **SST Documentation:** http://sst-simulator.org/sst-docs/
- **ACALSim Documentation:** See `docs/` directory
- **Example Code:** `src/sst-riscv/RISCVSoCStandalone.cc`
- **SST Tutorials:** SST-Core repository examples

---

## Conclusion

This guide provides everything needed to integrate ACALSim applications with SST. The pattern has been proven with the RISC-V integration and can be applied to other ACALSim simulators.

Key takeaways:
- Use the primary component pattern correctly
- Implement 2-phase termination for clean shutdown
- Remember SST clock return values (false=continue, true=stop)
- Leak SimTop in `finish()` to avoid crashes
- Use static linking to avoid symbol resolution issues

Happy integrating! 🚀

---

**Copyright 2023-2026 Playlab/ACAL**
Licensed under the Apache License, Version 2.0
