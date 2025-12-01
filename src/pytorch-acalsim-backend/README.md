# PyTorch ACALSim Backend

A custom TorchDynamo backend that compiles PyTorch models to run on the ACALSim accelerator simulator via QEMU-SST integration.

## Overview

This package provides a `torch.compile()` backend that:

1. Captures PyTorch FX graphs using TorchDynamo
2. Lowers them to ACALSim IR (intermediate representation)
3. Generates RISC-V bare-metal C code for the accelerator
4. Optionally runs simulation via QEMU-SST

## Installation

```bash
# Install in development mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from acalsim_backend import ACALSimBackend

# Create your model
model = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 10),
)

# Create the ACALSim backend
backend = ACALSimBackend(
    mode="codegen",    # Generate RISC-V code
    verbose=True,      # Print compilation info
    output_dir="./output",
)

# Compile the model
compiled_model = torch.compile(model, backend=backend)

# Run inference (also generates code)
x = torch.randn(4, 16)
output = compiled_model(x)
```

## Backend Modes

| Mode | Description |
|------|-------------|
| `trace` | Print FX graph, run eagerly |
| `compile` | Generate ACALSim IR, run eagerly |
| `codegen` | Generate RISC-V C code, run eagerly |
| `simulate` | Run on QEMU-SST (not yet implemented) |
| `eager` | No compilation, pure PyTorch |

## Generated Files

When using `codegen` mode, the following files are generated:

- `{name}.c` - Main C source with kernel implementation
- `{name}.h` - Header with tensor definitions
- `{name}_weights.c` - Weight data as C arrays
- `{name}_weights.bin` - Binary weight data
- `{name}.ld` - Linker script
- `Makefile` - Build configuration

## Building Generated Code

```bash
# Requires RISC-V toolchain
export CROSS_COMPILE=riscv64-unknown-elf-

cd output_dir
make
```

## Examples

See the `examples/` directory:

- `simple_mlp.py` - Basic MLP compilation
- `matmul_demo.py` - Matrix multiplication patterns
- `transformer_attention.py` - Attention mechanism

Run examples:
```bash
python examples/simple_mlp.py
python examples/matmul_demo.py
python examples/transformer_attention.py
```

## Architecture

```
PyTorch Model
     │
     ▼
┌────────────────┐
│ TorchDynamo    │  ← Captures FX Graph
└────────────────┘
     │
     ▼
┌────────────────┐
│ ACALSimBackend │  ← Custom backend
└────────────────┘
     │
     ▼
┌────────────────┐
│ ACALSimCompiler│  ← FX → ACALSim IR
└────────────────┘
     │
     ▼
┌────────────────┐
│ ACALSim IR     │  ← Intermediate representation
└────────────────┘
     │
     ▼
┌────────────────┐
│ RISCVCodeGen   │  ← IR → RISC-V C code
└────────────────┘
     │
     ▼
┌────────────────┐
│ QEMU-SST       │  ← Simulation (future)
└────────────────┘
```

## Supported Operations

Currently supported PyTorch operations:

- **Elementwise**: add, sub, mul, div, neg
- **Matrix**: matmul, mm, bmm, linear (GEMM)
- **Activations**: relu, sigmoid, tanh, gelu, softmax
- **Shape**: reshape, view, transpose, permute, flatten
- **Normalization**: batch_norm, layer_norm (partial)
- **Reduction**: sum, mean, max, min

## Integration with QEMU-SST

The generated code is designed to run on the qemu-acalsim-sst-baremetal infrastructure:

1. MMIO communication with accelerator components
2. Memory-mapped device registers at:
   - Echo Device: `0x10200000`
   - Compute Device: `0x10300000`

## License

Apache License 2.0 - See LICENSE file.
