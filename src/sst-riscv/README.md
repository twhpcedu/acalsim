# ACALSim SST Integration

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

This directory contains the SST (Structural Simulation Toolkit) integration for ACALSim, enabling ACALSim components to participate in SST simulations.

## Overview

The SST integration allows you to:
- Wrap ACALSim `SimBase` instances as SST Components
- Connect ACALSim components with native SST components via SST Links
- Use SST's Python configuration system to build complex systems
- Leverage SST's statistics, visualization, and analysis tools
- Combine ACALSim's high-performance parallel simulation with SST's component ecosystem

## Architecture

### Key Components

1. **ACALSimComponent** (`ACALSimComponent.hh/cc`)
   - Base SST Component wrapper for ACALSim simulators
   - Bridges ACALSim's event-driven model to SST's discrete event simulation
   - Maps ACALSim SimPorts to SST Links
   - Synchronizes ACALSim ticks with SST simulation time

2. **ACALSimSSTEvent** (`ACALSimComponent.hh`)
   - SST Event wrapper for ACALSim SimPackets
   - Enables communication between ACALSim and SST components

3. **Integration Points**
   - **Time Synchronization**: Maps ACALSim ticks to SST time units
   - **Event Scheduling**: Translates between ACALSim EventQueue and SST scheduler
   - **Communication**: Bridges SimPort/SimChannel to SST::Link
   - **Configuration**: Translates SST::Params to ACALSim SimConfig
   - **Lifecycle**: Adapts init/step/cleanup to SST's setup/init/clock/finish

## Prerequisites

### Required Software

1. **SST-Core** (version 12.0 or later)
   ```bash
   # Download from https://github.com/sstsimulator/sst-core
   # Or install via package manager
   ```

2. **ACALSim** (built and installed)
   ```bash
   cd /path/to/acalsim
   make
   # Ensure lib/libacalsim.a or lib/libacalsim.so exists
   ```

3. **Build Tools**
   - C++17 compatible compiler (GCC 7+, Clang 5+)
   - Python 3.6+
   - Make or CMake

### Environment Setup

Ensure `sst-config` is in your PATH:
```bash
which sst-config
# Should output the path to sst-config

# Test SST installation
sst --version
sst-info
```

## Building the SST Integration

### Using Make

```bash
cd src/sst-riscv

# Build the SST element library
make

# Install to SST
make install

# Verify installation
make test
# Or manually:
sst-info acalsim
```

### Build Output

- `libacalsim_sst.so` - SST element library containing ACALSim components
- Installed to SST's element library directory (typically `$SST_PREFIX/lib/sst-elements-library/`)

## Using ACALSim Components in SST

### Basic Example: Simple System

The simplest example demonstrates a single ACALSim processor connected to memory:

```python
#!/usr/bin/env python3
import sst

# Create processor
cpu = sst.Component("processor", "acalsim.SimpleProcessor")
cpu.addParams({
    "clock": "2GHz",
    "max_instructions": 1000,
    "verbose": 2
})

# Create memory
memory = sst.Component("memory", "acalsim.SimpleMemory")
memory.addParams({
    "clock": "2GHz",
    "latency": "50ns",
    "size": "1GiB"
})

# Connect via SST Link
link = sst.Link("cpu_mem_link")
link.connect(
    (cpu, "mem_port", "50ns"),
    (memory, "cpu_port", "50ns")
)
```

Run with:
```bash
sst examples/simple_system.py
```

### Advanced Example: Multi-Core System with NoC

See `examples/multicore_system.py` for a complete 4-core system with:
- 4 processors with private L1 caches
- 2x2 mesh Network-on-Chip interconnect
- Shared L2 cache
- Main memory
- Statistics collection

Run with:
```bash
sst examples/multicore_system.py
```

## Creating Custom ACALSim Components for SST

### Step 1: Implement Your ACALSim Simulator

```cpp
// MyCustomSimulator.hh
#include "ACALSim.hh"

class MyCustomSimulator : public SimBase {
public:
    MyCustomSimulator(SimConfig* config) : SimBase(config) {}

    void init() override {
        SimBase::init();
        // Your initialization
    }

    void step() override {
        SimBase::step();
        // Your per-tick simulation logic
    }

    void cleanup() override {
        // Your cleanup
        SimBase::cleanup();
    }
};
```

### Step 2: Create SST Component Wrapper

```cpp
// MyCustomSSTComponent.hh
#include "ACALSimComponent.hh"
#include "MyCustomSimulator.hh"

class MyCustomSSTComponent : public ACALSim::SST::ACALSimComponent {
public:
    // SST registration
    SST_ELI_REGISTER_COMPONENT(
        MyCustomSSTComponent,
        "acalsim",
        "MyCustom",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "My Custom ACALSim Component",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"custom_param", "Custom parameter", "default_value"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"input_port", "Input port", {}},
        {"output_port", "Output port", {}}
    )

    MyCustomSSTComponent(::SST::ComponentId_t id, ::SST::Params& params)
        : ACALSimComponent(id, params) {
        // Extract custom parameters
        std::string custom_param = params.find<std::string>("custom_param", "default");

        // Create your simulator instance
        // simulator_ = std::make_unique<MyCustomSimulator>(&config_);
    }
};
```

### Step 3: Build and Install

Add your source files to the Makefile:
```makefile
SOURCES = ACALSimComponent.cc MyCustomSSTComponent.cc
```

Build and install:
```bash
make clean
make install
```

### Step 4: Use in Python Configuration

```python
import sst

component = sst.Component("my_component", "acalsim.MyCustom")
component.addParams({
    "clock": "2GHz",
    "custom_param": "my_value"
})
```

## SST Component Parameters

### Common Parameters

All ACALSim SST components support these base parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clock` | string | "1GHz" | Clock frequency (e.g., "2GHz", "500MHz") |
| `verbose` | int | 1 | Verbosity level (0-5) |
| `name` | string | "acalsim0" | Component instance name |
| `max_ticks` | uint64 | 0 | Maximum simulation ticks (0 = unlimited) |
| `config_file` | string | "" | Path to ACALSim JSON configuration |

### Component-Specific Parameters

Add your own parameters using `SST_ELI_DOCUMENT_PARAMS` in your component definition.

## Port/Link Configuration

### Defining Ports

```cpp
SST_ELI_DOCUMENT_PORTS(
    {"port_name", "Port description", {"event_types"}}
)
```

### Connecting Links in Python

```python
# Create link
link = sst.Link("link_name")

# Connect two components
link.connect(
    (component1, "port1", "latency1"),
    (component2, "port2", "latency2")
)
```

## Statistics Collection

### Enable Statistics in Python

```python
# Set global statistics configuration
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputCSV", {
    "filepath": "stats.csv"
})

# Enable per-component statistics
component.enableAllStatistics({
    "type": "sst.AccumulatorStatistic",
    "rate": "1us"
})
```

### Implementing Statistics in C++

```cpp
// In your component constructor
registerStatistic<uint64_t>("packets_sent", "Number of packets sent");
registerStatistic<uint64_t>("packets_received", "Number of packets received");

// Update during simulation
self->packets_sent->addData(1);
```

## Debugging

### Enable Verbose Output

```python
component.addParams({
    "verbose": 5  # Maximum verbosity
})
```

### SST Debug Flags

```bash
# Run with SST debug output
sst --debug-file=debug.txt --verbose config.py

# Run with specific debug flags
sst --debug-mask=TRACE config.py
```

### Common Issues

1. **"Element not found" error**
   - Verify installation: `sst-info acalsim`
   - Check library path: `echo $SST_ELEMENT_PATH`
   - Reinstall: `make clean install`

2. **Linking errors**
   - Ensure ACALSim is built: `ls ../../lib/libacalsim.*`
   - Check compiler flags in Makefile

3. **Runtime crashes**
   - Enable verbose logging
   - Check port connections are bidirectional
   - Verify parameter types match expectations

## Examples

### Provided Examples

1. **simple_system.py**
   - Single processor with memory
   - Basic SST configuration
   - ~50 lines

2. **multicore_system.py**
   - 4-core SMP system
   - Network-on-Chip interconnect
   - Cache hierarchy (L1/L2)
   - Statistics collection
   - ~200 lines

3. **riscv_single_core.py** 🆕
   - Complete RISC-V RV32I processor
   - Runs RISC-V assembly programs
   - Event-driven timing model
   - Pipeline visualization (IF, EXE, WB)
   - Register file and memory dumps
   - ~100 lines

4. **riscv_dual_core.py** 🆕
   - Dual RISC-V processors
   - Independent program execution per core
   - Parallel simulation
   - Per-core statistics collection
   - ~150 lines

### Running Examples

```bash
# Simple system
sst examples/simple_system.py

# Multi-core system with statistics
sst examples/multicore_system.py

# With custom simulation time
sst --stop-at=1ms examples/multicore_system.py

# RISC-V single core
sst examples/riscv_single_core.py

# RISC-V dual core with different programs
sst examples/riscv_dual_core.py
```

### RISC-V Examples

The RISC-V integration provides complete RV32I processor simulations:

**Features**:
- Complete RV32I ISA (32 base integer instructions)
- Event-driven timing accurate simulation
- Pipeline stages: IF (Instruction Fetch), EXE (Execute), WB (Write-Back)
- Data and control hazard detection
- Assembly program execution from .s/.txt files

**Available Assembly Programs** (`../../src/riscv/asm/`):
- `branch_simple.txt` - Branch instruction tests
- `load_store_simple.txt` - Memory access tests
- `full_test.txt` - Comprehensive ISA test
- `test.txt` - Basic arithmetic operations

**Supported Instructions**:
- Arithmetic: ADD, SUB, ADDI
- Logical: AND, OR, XOR, ANDI, ORI, XORI
- Shifts: SLL, SRL, SRA, SLLI, SRLI, SRAI
- Memory: LB, LBU, LH, LHU, LW, SB, SH, SW
- Branches: BEQ, BNE, BLT, BLTU, BGE, BGEU
- Jumps: JAL, JALR
- Upper Immediate: LUI, AUIPC
- System: HCF (Halt and Catch Fire)

**See `examples/RISCV_README.md` for comprehensive RISC-V documentation.**

## Integration Testing

### Unit Tests

Test individual component registration:
```bash
sst-info acalsim
sst-info acalsim.ACALSimComponent
```

### Functional Tests

Create simple test configurations:
```python
# test_basic.py
import sst
comp = sst.Component("test", "acalsim.SimpleProcessor")
comp.addParams({"max_instructions": 10})
```

Run:
```bash
sst test_basic.py
```

## Performance Considerations

### ACALSim Threading vs SST Event-Driven

- ACALSim uses parallel thread pools for multi-simulator execution
- SST uses sequential discrete event simulation
- For best performance, wrap logical simulator groups as components
- Minimize cross-component communication

### Latency Modeling

- Set realistic link latencies based on system architecture
- Use SST's TimeConverter for accurate time synchronization
- Consider ACALSim's tick granularity vs SST's picosecond precision

## Advanced Topics

### Distributed Simulation

For parallel/distributed SST simulations:
1. Implement serialization for ACALSimSSTEvent
2. Use SST's partitioning hints
3. Minimize cross-partition links

### Integration with SST Elements

Mix ACALSim components with:
- `sst-elements` libraries (Ariel, Miranda, etc.)
- Ramulator memory models
- Merlin network simulator
- Native SST processors

Example:
```python
# ACALSim CPU with SST Merlin NoC
cpu = sst.Component("cpu", "acalsim.SimpleProcessor")
router = sst.Component("router", "merlin.hr_router")
link = sst.Link("cpu_router")
link.connect((cpu, "noc_port", "1ns"), (router, "port0", "1ns"))
```

## References

### SST Documentation
- [SST-Core Documentation](https://sst-simulator.org/sst-docs/)
- [Component Writers Guide](https://sst-simulator.org/sst-docs/docs/guides/component/)
- [Python Configuration](https://sst-simulator.org/sst-docs/docs/guides/config/)

### ACALSim Documentation
- [ACALSim Overview](../../docs/README.md)
- [Event-Driven Simulation](../../docs/for-users/event-driven-simulation.md)
- [SimPort Usage](../../docs/for-users/simport.md)
- [SimChannel Usage](../../docs/for-users/simchannel.md)

## Contributing

To contribute improvements to the SST integration:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/sst-enhancement`
3. Make your changes
4. Test thoroughly with various SST configurations
5. Submit a pull request

## License

Copyright 2023-2026 Playlab/ACAL

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.

## Support

For questions and issues:
- GitHub Issues: https://github.com/[your-repo]/acalsim/issues
- Email: weifen@twhpcedu.org
- Documentation: https://[your-docs-site]

## Acknowledgments

- SST-Simulator Team for the excellent simulation framework
- ACALSim contributors for the high-performance simulation engine
- Taiwan High-Performance Computing Education Association
