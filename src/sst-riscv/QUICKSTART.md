# ACALSim SST Integration - Quick Start Guide

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

Get started with ACALSim SST integration in 5 minutes!

## Prerequisites Check

```bash
# Check SST installation
sst --version
# Should output: SST-Core Version 12.x.x

# Check sst-config
which sst-config
# Should output path to sst-config

# Check ACALSim build
ls ../../lib/libacalsim.*
# Should show libacalsim.a or libacalsim.so
```

If any checks fail, install the required software first (see [README.md](README.md#prerequisites)).

## Quick Build & Install

### Option 1: Using Make (Recommended)

```bash
# Navigate to SST integration directory
cd src/sst-riscv

# Build and install in one command
make install

# Verify installation
sst-info acalsim
```

Expected output:
```
Registered Components:
  acalsim.ACALSimComponent
```

### Option 2: Using CMake

```bash
# Navigate to SST integration directory
cd src/sst-riscv

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make

# Install
sudo make install

# Test
make test_sst
```

## Run Your First Example

### Example 1: Simple System (Processor + Memory)

```bash
# Run the simple system example
sst examples/simple_system.py
```

Expected output:
```
Creating components...
Simulation running...
=== ACALSim Component Statistics ===
Ticks executed:    1000
Events received:   ...
Events sent:       ...
====================================
Simulation complete.
```

### Example 2: Multi-Core System with NoC

```bash
# Run the multi-core example
sst examples/multicore_system.py
```

This simulates a 4-core system with:
- 4 processors with L1 caches
- 2x2 mesh Network-on-Chip
- Shared L2 cache
- Main memory

Output includes detailed statistics saved to `multicore_stats.csv`.

## Create Your First Custom Component

### Step 1: Create Component Header

Create `MyProcessor.hh`:

```cpp
#include "../ACALSimComponent.hh"
#include "ACALSim.hh"

class MyProcessor : public SimBase {
public:
    MyProcessor(SimConfig* config) : SimBase(config), tick_count_(0) {}

    void init() override {
        SimBase::init();
        std::cout << "MyProcessor initialized" << std::endl;
    }

    void step() override {
        SimBase::step();
        tick_count_++;
        if (tick_count_ % 1000 == 0) {
            std::cout << "Tick: " << tick_count_ << std::endl;
        }
    }

    bool isDone() const override {
        return tick_count_ >= 10000;
    }

private:
    uint64_t tick_count_;
};

class MyProcessorSST : public ACALSim::SST::ACALSimComponent {
public:
    SST_ELI_REGISTER_COMPONENT(
        MyProcessorSST,
        "acalsim",
        "MyProcessor",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "My Custom Processor",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"}
    )

    MyProcessorSST(::SST::ComponentId_t id, ::SST::Params& params)
        : ACALSimComponent(id, params) {
        // Initialize your processor here
    }
};
```

### Step 2: Update Build System

Add to `Makefile`:
```makefile
SOURCES = ACALSimComponent.cc examples/MyProcessor.cc
```

### Step 3: Rebuild and Install

```bash
make clean
make install
sst-info acalsim.MyProcessor
```

### Step 4: Create Python Configuration

Create `my_sim.py`:

```python
import sst

processor = sst.Component("my_cpu", "acalsim.MyProcessor")
processor.addParams({
    "clock": "2GHz",
    "verbose": 2
})
```

### Step 5: Run!

```bash
sst my_sim.py
```

## Common First-Time Issues

### Issue 1: "Element 'acalsim' not found"

**Solution**: Element library not installed correctly.

```bash
# Check SST library path
sst-config --prefix

# Verify library exists
ls $(sst-config --prefix)/lib/sst-elements-library/libacalsim_sst.so

# If missing, reinstall
make clean install
```

### Issue 2: Linking Errors

**Solution**: ACALSim not built or wrong path.

```bash
# Build ACALSim first
cd ../..
make

# Verify library exists
ls lib/libacalsim.*

# Rebuild SST integration
cd src/sst-riscv
make clean install
```

### Issue 3: Python Configuration Errors

**Solution**: Check component and port names.

```python
# Wrong:
cpu = sst.Component("cpu", "MyProcessor")  # Missing library prefix

# Correct:
cpu = sst.Component("cpu", "acalsim.MyProcessor")
```

## Next Steps

Now that you have the basics working:

1. **Explore Examples**: Study `examples/multicore_system.py` for advanced features
2. **Read Documentation**: See [README.md](README.md) for comprehensive guide
3. **Create Components**: Wrap your own ACALSim simulators as SST components
4. **Mix with SST**: Combine with native SST components (Ariel, Merlin, etc.)
5. **Collect Statistics**: Add statistics tracking to your components

## Quick Reference

### Build Commands
```bash
make              # Build library
make install      # Build and install
make test         # Verify installation
make clean        # Clean build artifacts
make examples     # Show example commands
```

### SST Commands
```bash
sst config.py              # Run simulation
sst --version              # Check SST version
sst-info                   # List all elements
sst-info acalsim           # Show ACALSim components
sst --stop-at=1ms conf.py  # Run with time limit
```

### Debugging Commands
```bash
# Verbose component output
# Set "verbose": 5 in component params

# SST debug output
sst --debug-file=debug.txt config.py

# Check element registration
sst-info -x acalsim
```

## Getting Help

- **Full Documentation**: [README.md](README.md)
- **ACALSim Docs**: [../../docs/README.md](../../docs/README.md)
- **SST Docs**: https://sst-simulator.org/sst-docs/
- **Issues**: GitHub Issues or email maintainers

## Example Output

When everything is working, you should see output like:

```
$ sst examples/simple_system.py
SST Configuration Complete:
  - Processor clock: 2GHz
  - Max instructions: 1000
  - Memory latency: 50ns
  - Link: processor <-> memory

ACALSimComponent[@p:@l]: Initializing ACALSim SST Component
ACALSimComponent[@p:@l]: Clock frequency: 2GHz
ACALSimComponent[@p:@l]: Component initialization complete
...
=== ACALSim Component Statistics ===
Ticks executed:    1000
Events received:   100
Events sent:       100
====================================
Simulation is complete, simulated time: 500 ns
```

Congratulations! You're now running ACALSim components in SST.

---

**Need more help?** Check the full [README.md](README.md) or open an issue on GitHub.
