# SST Integration Documentation

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

This directory contains comprehensive documentation for integrating ACALSim with the Structural Simulation Toolkit (SST).

## 📋 Quick Navigation

### 🎯 New to SST Integration?

Start here:

1. **[Quick Start Guide](quickstart.md)**  
   Get SST integration up and running in 5 minutes
   - Installation steps
   - First simulation
   - Basic examples

2. **[Architecture Diagrams](architecture-diagram-simple.md)**  
   Visual overview of the integration architecture
   - Simplified Mermaid diagrams (renders in GitHub)
   - Component relationships
   - Data flow diagrams
   - Key patterns explained

### 📐 Architecture Documentation

Understanding the system:

- **[Architecture Diagram (Detailed)](architecture-diagram.md)**  
  Comprehensive ASCII diagrams with detailed explanations
  - Layer-by-layer breakdown
  - Complete data flows
  - Integration patterns
  - Performance characteristics

- **[Architecture Diagram (Visual)](architecture-diagram.puml)**  
  PlantUML diagram (can be rendered with PlantUML tools)
  - High-level system view
  - Component hierarchy
  - External interfaces

### 📚 Integration Guides

Detailed guides for developers:

- **[Integration Guide](integration-guide.md)**  
  Complete guide for integrating ACALSim applications with SST
  - Architecture integration patterns
  - Step-by-step integration tutorial
  - Component lifecycle
  - Best practices
  - Troubleshooting

### 🚀 Advanced Topics

- **[PyTorch Device GEMM](pytorch-device-gemm.md)**  
  Offload PyTorch GEMM operations to SST for cycle-accurate simulation
  - Docker → QEMU → VirtIO → SST pipeline
  - Custom PyTorch operators
  - Device server protocol
  - Full-system integration

- **[RISC-V Examples](riscv-examples.md)**  
  Using the RISC-V SST integration
  - Single-core and dual-core examples
  - Assembly program examples
  - Pipeline simulation

- **[Hybrid Multi-GPU Architecture](HYBRID_MULTI_GPU_DIAGRAMS.md)** 🆕  
  Multi-GPU LLM inference with ACALSim + SST
  - Dual-port TCP architecture (Job + NVLink)
  - NVLink topology configurations (full-mesh, ring, 2x2)
  - Process isolation and scalability
  - Communication flow diagrams

- **[Scalable Multi-Rank Architecture](SCALABLE_MULTI_RANK.md)** 🆕  
  Production-scale simulation with multiple SST ranks
  - Multi-rank MPI deployment (32-128+ GPUs)
  - Intra-rank NVLink mesh + inter-rank links
  - Linear scaling analysis
  - Mode A (C++ InferenceServer) and Mode B (PyTorch) support

---

## 📊 What's Available

### ✅ Production-Ready Components

| Component | Status | Description | Python Name |
|-----------|--------|-------------|-------------|
| **RISCVSoCStandalone** | ✅ Working | Complete RISC-V RV32I CPU with pipeline | `acalsim.RISCVSoCStandalone` |
| **QEMUBinaryComponent** | ✅ Production | QEMU full-system integration (binary protocol) | `qemubinary.QEMUBinary` |
| **ACALSimDeviceComponent** | ✅ Working | Simple echo device model | `acalsim.QEMUDevice` |
| **ACALSimComputeDeviceComponent** | ✅ Production | GEMM accelerator with cycle-accurate timing | `acalsim.ComputeDevice` |
| **ACALSimVirtIODeviceComponent** | ✅ Production | VirtIO-SST device for PyTorch offloading | `acalsim.VirtIODevice` |
| **HSAComputeComponent** | ✅ Working | Multi-accelerator HSA support | `acalsim.HSACompute` |

### 📦 Example Configurations

Located in `src/sst-riscv/examples/`:

- `riscv_single_core.py` - Single RISC-V processor
- `riscv_dual_core.py` - Dual RISC-V processors (parallel)
- `simple_system.py` - Basic processor + memory

Located in `src/qemu-acalsim-sst-linux/examples/`:

- `llama-inference/` - PyTorch Llama inference with SST backend
- `hsa-multi-accelerator/` - HSA multi-device demo

---

## 🏗️ Architecture Overview

### The Big Picture

```
┌───────────────────────────────────────────────────────────┐
│                    SST-Core Framework                     │
│          (Discrete Event Simulation Engine)               │
└─────────────────────┬─────────────────────────────────────┘
                      │ Component Interface
                      ↓
┌───────────────────────────────────────────────────────────┐
│              SST Integration Bridge                       │
│          (ACALSimComponent base class)                    │
│  • Maps SST clock ticks → ACALSim 2-phase execution      │
│  • Bridges SST::Link ↔ SimPort/SimChannel                │
│  • Primary component pattern + 2-phase termination       │
└─────────────────────┬─────────────────────────────────────┘
                      │ Inheritance
                      ↓
┌───────────────────────────────────────────────────────────┐
│            Application Components (3 paths)               │
│                                                           │
│  Path A: RISC-V Standalone                                │
│  • RISCVSoCStandalone wraps complete SOCTop              │
│  • CPU, pipeline stages, memory                          │
│                                                           │
│  Path B: QEMU Full-System                                │
│  • QEMUBinaryComponent manages QEMU subprocess           │
│  • Binary MMIO protocol (Unix sockets)                   │
│  • N-device routing by address                           │
│                                                           │
│  Path C: Device Components                               │
│  • Echo device, Compute device, VirtIO device            │
│  • HSA multi-accelerator support                         │
└─────────────────────┬─────────────────────────────────────┘
                      │ External Interfaces
                      ↓
┌───────────────────────────────────────────────────────────┐
│              External Applications                        │
│  • QEMU (qemu-system-riscv32 + Linux)                    │
│  • PyTorch (Docker → TCP → VirtIO → SST)                 │
│  • User applications (/dev/sst0)                         │
└───────────────────────────────────────────────────────────┘
```

**For detailed architecture diagrams with data flows, see [architecture-diagram.md](architecture-diagram.md) or [architecture-diagram-simple.md](architecture-diagram-simple.md).**

---

## 🔑 Key Concepts

### 1. 2-Phase Execution Bridge

ACALSim uses a 2-phase event-driven model, SST uses clock-driven. The bridge maps:

- **1 SST clock tick** = **1 ACALSim iteration**
- Each iteration runs **Phase 1** (parallel) + **Phase 2** (sync)
- Fast-forward allows skipping empty ticks

### 2. Primary Component Pattern

To control SST simulation lifetime:

```cpp
// Constructor:
registerAsPrimaryComponent();
primaryComponentDoNotEndSim();  // "We're in control"

// When done:
primaryComponentOKToEndSim();   // "OK to terminate"
ready_to_terminate_ = true;
return false;  // One more cycle for cleanup

// Next iteration:
return true;  // Actually stop
```

### 3. Address-Based Device Routing

For QEMU multi-device support:

```
QEMU → MMIORequest (addr=0x10005000)
  ↓
QEMUBinaryComponent routes by address:
  • 0x10000000-0x10000FFF → Device 0 (Echo)
  • 0x20000000-0x20000FFF → Device 1 (Compute)
  • 0x30000000-0x30000FFF → Device 2 (VirtIO)
  ↓
Device processes transaction and returns via SST Link
```

---

## 📈 Performance

### RISC-V Standalone
- **Throughput**: ~1M instructions/second (simulated)
- **Overhead**: ~10% SST wrapping overhead
- **Memory**: ~50MB per component

### QEMU + Device (Binary Protocol)
- **MMIO Latency**: ~100μs per transaction
- **Throughput**: ~10K transactions/second
- **CPU Usage**: <5% (vs 50% with text protocol)
- **Improvement**: 10x faster than Phase 2A text protocol

### PyTorch Device GEMM
- **Offload Latency**: ~1ms per GEMM operation
- **Cycle Accuracy**: ±1% vs hardware measurements
- **Scalability**: Tested up to 512×512×512 matrices

---

## 🛠️ Building

### Prerequisites

1. **SST-Core 12.0+**  
   ```bash
   # See quickstart.md for installation
   sst --version
   ```

2. **ACALSim**  
   ```bash
   cd /path/to/acalsim
   make
   ls lib/libacalsim.so  # Verify
   ```

### Build Steps

```bash
# Navigate to SST integration
cd src/sst-riscv

# Build the SST element library
make

# Install to SST
make install

# Verify
sst-info acalsim
```

### Running Examples

```bash
# RISC-V single core
cd src/sst-riscv/examples
sst riscv_single_core.py

# RISC-V dual core
sst riscv_dual_core.py

# QEMU + PyTorch (requires multiple terminals)
# See pytorch-device-gemm.md for full setup
```

---

## 🐛 Troubleshooting

### Common Issues

1. **"Element not found" error**
   ```bash
   sst-info acalsim  # Should show registered components
   # If not found:
   cd src/sst-riscv
   make clean install
   ```

2. **Double-free crash on termination**
   - Leak SimTop in `finish()` to avoid destructor crashes
   - See integration-guide.md "Troubleshooting" section

3. **Event queue empty - immediate exit**
   - Not registering as primary component
   - Wrong clock tick return values (false=continue, true=stop)

4. **Symbol resolution errors**
   - Use static linking with `--whole-archive`
   - Rebuild ACALSim library

**For detailed troubleshooting, see [integration-guide.md](integration-guide.md).**

---

## 📝 Contributing

### Adding New Components

1. Follow the integration pattern in `RISCVSoCStandalone`
2. Inherit from `ACALSimComponent`
3. Implement SST lifecycle methods
4. Create Python configuration example
5. Test thoroughly
6. Update documentation

**See [integration-guide.md](integration-guide.md) "Step-by-Step Integration Guide" for details.**

### Candidate Applications for Integration

Ready to integrate:
- ✅ **riscv** - Already done
- 🟡 **riscvSimTemplate** - Similar to riscv but simpler
- 🟡 **testSimPort** - Port communication demo
- 🟡 **testConfig** - Configuration examples
- 🟡 **testPETile** - Multi-core processing element
- 🟡 **testBlackBear** - Accelerator example

---

## 📚 Additional Resources

### Documentation Files

| File | Purpose |
|------|---------|
| [quickstart.md](quickstart.md) | Get started in 5 minutes |
| [architecture-diagram-simple.md](architecture-diagram-simple.md) | Visual Mermaid diagrams (GitHub-friendly) |
| [architecture-diagram.md](architecture-diagram.md) | Complete architecture with ASCII diagrams |
| [architecture-diagram.puml](architecture-diagram.puml) | PlantUML source (for rendering) |
| [integration-guide.md](integration-guide.md) | Complete integration tutorial |
| [pytorch-device-gemm.md](pytorch-device-gemm.md) | PyTorch offloading guide |
| [riscv-examples.md](riscv-examples.md) | RISC-V example documentation |
| [HYBRID_MULTI_GPU_DIAGRAMS.md](HYBRID_MULTI_GPU_DIAGRAMS.md) | 🆕 Multi-GPU architecture diagrams |
| [SCALABLE_MULTI_RANK.md](SCALABLE_MULTI_RANK.md) | 🆕 Scalable multi-rank architecture (32-128+ GPUs) |

### Source Code Locations

| Path | Contents |
|------|----------|
| `src/sst-riscv/` | RISC-V SST integration + examples |
| `src/qemu-acalsim-sst-linux/` | QEMU integration + PyTorch examples |
| `include/sst/` | SST component headers |
| `libs/sst/` | SST component implementations |

### External Links

- [SST-Core Documentation](https://sst-simulator.org/sst-docs/)
- [SST Component Writers Guide](https://sst-simulator.org/sst-docs/docs/guides/component/)
- [ACALSim Main Documentation](../README.md)
- [ACALSim GitHub Repository](https://github.com/acal-project/acalsim)

---

## 📧 Support

For questions, issues, or contributions:

- **GitHub Issues**: [acal-project/acalsim/issues](https://github.com/acal-project/acalsim/issues)
- **Email**: weifen@twhpcedu.org
- **Organization**: Taiwan High-Performance Computing Education Association

---

## 📄 License

Copyright 2023-2026 Playlab/ACAL

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.

---

## 🎯 Quick Links Summary

**Start Here:**
- [Quick Start](quickstart.md) - Get running in 5 minutes
- [Architecture (Simple)](architecture-diagram-simple.md) - Visual overview

**Learn More:**
- [Architecture (Detailed)](architecture-diagram.md) - Complete architecture
- [Integration Guide](integration-guide.md) - Step-by-step tutorial
- [PyTorch GEMM](pytorch-device-gemm.md) - Advanced integration

**Examples:**
- `src/sst-riscv/examples/` - RISC-V examples
- `src/qemu-acalsim-sst-linux/examples/` - QEMU + PyTorch examples

**Get Help:**
- [Troubleshooting (Integration Guide)](integration-guide.md#troubleshooting)
- [GitHub Issues](https://github.com/acal-project/acalsim/issues)

