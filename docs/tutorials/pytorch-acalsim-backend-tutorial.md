# PyTorch-ACALSim Backend Tutorial

## Complete Flow: From PyTorch Model to RISC-V Accelerator Simulation

This tutorial explains the complete software architecture and data flow for compiling PyTorch models to run on the ACALSim accelerator simulator via QEMU-SST integration.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Software Architecture](#2-software-architecture)
3. [Step-by-Step Flow with matmul_demo.py](#3-step-by-step-flow-with-matmul_demopy)
4. [Component Deep Dives](#4-component-deep-dives)
5. [SST Component Communication Protocol](#5-sst-component-communication-protocol)
6. [Generated Code Analysis](#6-generated-code-analysis)
7. [Running the Complete Pipeline](#7-running-the-complete-pipeline)

---

## 1. Overview

The PyTorch-ACALSim backend enables you to:

1. Write PyTorch models using familiar APIs
2. Compile them using `torch.compile()` with a custom backend
3. Generate RISC-V bare-metal C code
4. Simulate execution on the QEMU-SST accelerator simulator

### Why This Matters

- **Hardware-Software Co-design**: Test accelerator designs with real ML workloads
- **Cycle-accurate Simulation**: Get detailed performance metrics via SST
- **Rapid Prototyping**: Iterate on accelerator architecture without hardware

---

## 2. Software Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PyTorch Application                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  model = MatMulModel()                                           │   │
│  │  compiled = torch.compile(model, backend=ACALSimBackend)         │   │
│  │  output = compiled(input_tensor)                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Layer 1: TorchDynamo + Custom Backend                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │   TorchDynamo    │──│  ACALSimBackend  │──│   ACALSimCompiler    │  │
│  │  (Graph Capture) │  │  (backend.py)    │  │   (compiler.py)      │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
│           │                                            │                │
│           ▼                                            ▼                │
│    ┌─────────────┐                           ┌─────────────────┐       │
│    │  FX Graph   │                           │  ACALSim IR     │       │
│    │  (PyTorch)  │                           │  (ir.py)        │       │
│    └─────────────┘                           └─────────────────┘       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Layer 2: Code Generation                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    RISCVCodeGenerator (riscv_codegen.py)          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │  │
│  │  │  kernel.c   │  │  kernel.h   │  │  weights.c  │  │ Makefile│  │  │
│  │  │  (C code)   │  │  (headers)  │  │  (data)     │  │         │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Layer 3: RISC-V Compilation                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              riscv64-unknown-elf-gcc (Cross Compiler)             │  │
│  │                              │                                    │  │
│  │                              ▼                                    │  │
│  │                      kernel.elf (RISC-V Binary)                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Layer 4: QEMU-SST Simulation                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      QEMU (RISC-V Emulator)                         │ │
│  │  ┌─────────────────┐     ┌─────────────────────────────────────┐   │ │
│  │  │ RISC-V CPU Core │     │        SST Device (MMIO)            │   │ │
│  │  │  (kernel.elf)   │────▶│  Base: 0x10200000 (Echo Device)     │   │ │
│  │  │                 │     │  Base: 0x10300000 (Compute Device)  │   │ │
│  │  └─────────────────┘     └───────────────┬─────────────────────┘   │ │
│  └──────────────────────────────────────────┼─────────────────────────┘ │
│                                             │ Unix Socket               │
│                                             ▼                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    SST Simulation Framework                         │ │
│  │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐  │ │
│  │  │QEMUBinaryComponent│───│ACALSimDevice     │───│ Custom Accel │  │ │
│  │  │(Socket Server)   │    │Component         │    │ Components   │  │ │
│  │  └──────────────────┘    └──────────────────┘    └──────────────┘  │ │
│  │                                                                     │ │
│  │  Output: Cycle counts, Memory traces, Performance statistics        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Flow with matmul_demo.py

Let's trace through `examples/matmul_demo.py` to understand each stage:

### 3.1 Source Code: MatMulModel

```python
# examples/matmul_demo.py

class MatMulModel(nn.Module):
    """A model with matrix multiplication."""

    def __init__(self, M=8, K=16, N=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, N))  # [16, 8]
        self.bias = nn.Parameter(torch.randn(N))       # [8]

    def forward(self, x):
        # x: [M, K] @ weight: [K, N] -> [M, N]
        y = torch.matmul(x, self.weight)
        y = y + self.bias
        return y
```

### 3.2 Compilation Invocation

```python
# Create backend with codegen mode
backend = ACALSimBackend(
    mode="codegen",      # Generate RISC-V code
    verbose=True,        # Print compilation info
    output_dir="output_matmul"
)

# Compile the model
compiled = torch.compile(model, backend=backend)

# Run inference (triggers compilation)
with torch.no_grad():
    output = compiled(x)
```

### 3.3 What Happens at Each Stage

#### Stage 1: TorchDynamo Graph Capture

When `compiled(x)` is called, TorchDynamo:
1. Traces the Python code execution
2. Captures operations into an FX Graph
3. Passes the graph to our custom backend

**Captured FX Graph:**
```
Graph nodes:
  placeholder: l_x_ -> L_x_                    # Input tensor
  get_attr: l__self___weight -> L__self___weight  # Weight parameter
  call_function: y -> torch.matmul             # Matrix multiply
  get_attr: l__self___bias -> L__self___bias   # Bias parameter
  call_function: y_1 -> operator.add           # Add bias
  output: output -> output                     # Output
```

#### Stage 2: ACALSimBackend Processing

`acalsim_backend/backend.py:__call__()` receives the FX GraphModule:

```python
def __call__(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    # gm contains the captured FX graph
    # example_inputs contains shapes: [(8, 16)]

    if self.mode == "codegen":
        return self._codegen_mode(gm, example_inputs, graph_name)
```

#### Stage 3: ACALSimCompiler - FX to IR Conversion

`acalsim_backend/compiler.py:compile()` converts FX nodes to ACALSim IR:

```python
def compile(self, gm, example_inputs, graph_name):
    ir = ACALSimIR(name=graph_name)

    # Shape propagation using example inputs
    shape_info = self._propagate_shapes(gm, example_inputs)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Create input tensor descriptor
            desc = TensorDesc(name="input_0", shape=(8, 16), is_input=True)
            ir.add_tensor(desc)

        elif node.op == "get_attr":
            # Extract weight/bias parameters
            param = self._get_attr_value(gm, node.target)
            desc = TensorDesc(name="param_1", shape=(16, 8), is_param=True)
            ir.add_tensor(desc)
            ir.constants["param_1"] = param.numpy()  # Store weight data

        elif node.op == "call_function":
            # Create operation node
            self._compile_call_function(node, ir, ...)
```

**Generated ACALSim IR:**
```
=== ACALSim IR: graph_0 ===

Tensors:
  input_0: (8, 16) float32 [input]
  param_1: (16, 8) float32 [param]    # Weight
  t_2: (8, 8) float32                  # Matmul output
  param_3: (8,) float32 [param]        # Bias
  t_4: (8, 8) float32 [output]         # Final output

Operations:
  0: t_2 = MATMUL(input_0, param_1)
  1: t_4 = ADD(t_2, param_3)

Inputs: ['input_0']
Outputs: ['t_4']
```

#### Stage 4: RISCVCodeGenerator - IR to C Code

`codegen/riscv_codegen.py:generate()` produces bare-metal C:

```python
def generate(self, ir: ACALSimIR, name: str):
    # Generate main C file
    c_source = self._generate_c_source(ir, name)

    # Generate header with definitions
    header = self._generate_header(ir, name)

    # Generate weight data arrays
    weights_c = self._generate_weights_c(ir, name)

    # Generate Makefile and linker script
    makefile = self._generate_makefile(name)
    linker_script = self._generate_linker_script(ir)
```

---

## 4. Component Deep Dives

### 4.1 ACALSimBackend (backend.py)

**Purpose**: Entry point for `torch.compile()`, routes to appropriate mode.

**Key Implementation:**
```python
class ACALSimBackend:
    def __init__(self, mode="compile", verbose=False, output_dir=None):
        self.mode = mode
        self.compiler = ACALSimCompiler(verbose=verbose)
        self._compiled_cache = {}  # Cache compiled IRs

    def __call__(self, gm, example_inputs):
        """Called by TorchDynamo with captured graph."""
        graph_name = f"graph_{self._graph_counter}"

        if self.mode == "codegen":
            # 1. Compile FX graph to IR
            ir = self.compiler.compile(gm, example_inputs, graph_name)

            # 2. Generate RISC-V code
            from codegen.riscv_codegen import RISCVCodeGenerator
            codegen = RISCVCodeGenerator(output_dir=self.output_dir)
            code_files = codegen.generate(ir, graph_name)

            # 3. Return eager execution for now
            return gm.forward
```

### 4.2 ACALSimCompiler (compiler.py)

**Purpose**: Convert PyTorch FX graphs to ACALSim IR.

**Key Stages:**

1. **Shape Propagation**: Run example inputs to infer tensor shapes
2. **Node Processing**: Convert each FX node to IR operations
3. **Constant Extraction**: Extract weights/biases for embedding

```python
def _compile_call_function(self, node, ir, node_to_tensor, shape_info):
    """Convert FX call_function to ACALSim operation."""

    # Get PyTorch operation name
    op_name = node.target.__name__  # e.g., "matmul"

    # Map to ACALSim operation type
    acalsim_op_type = get_acalsim_op_type(op_name)  # -> MATMUL

    # Collect input tensor names
    input_names = []
    for arg in node.args:
        if hasattr(arg, 'name'):
            input_names.append(node_to_tensor[arg.name])

    # Create output tensor
    output_name = self._new_tensor_name("t")
    output_shape = shape_info[node.name]

    # Create IR operation
    op = ACALSimOp(
        op_type=acalsim_op_type,
        name=f"matmul_0",
        inputs=input_names,      # ["input_0", "param_1"]
        outputs=[output_name],   # ["t_2"]
        compute_cycles=self._estimate_compute_cycles(acalsim_op_type, output_shape)
    )
    ir.add_op(op)
```

### 4.3 ACALSimIR (ir.py)

**Purpose**: Intermediate representation for accelerator operations.

**Key Data Structures:**

```python
class ACALSimOpType(Enum):
    """Supported accelerator operations."""
    LOAD = auto()      # Memory load
    STORE = auto()     # Memory store
    ADD = auto()       # Element-wise add
    MUL = auto()       # Element-wise multiply
    MATMUL = auto()    # Matrix multiplication
    GEMM = auto()      # General matrix multiply
    RELU = auto()      # ReLU activation
    SOFTMAX = auto()   # Softmax
    # ... more ops

@dataclass
class TensorDesc:
    """Tensor metadata."""
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    is_input: bool = False
    is_output: bool = False
    is_param: bool = False
    memory_addr: Optional[int] = None  # Assigned during codegen

@dataclass
class ACALSimOp:
    """Single operation in the IR."""
    op_type: ACALSimOpType
    name: str
    inputs: List[str]      # Input tensor names
    outputs: List[str]     # Output tensor names
    attrs: Dict[str, Any]  # Operation attributes
    compute_cycles: int    # Estimated cycles
```

### 4.4 RISCVCodeGenerator (riscv_codegen.py)

**Purpose**: Generate bare-metal RISC-V C code from IR.

**Generated Files:**

| File | Purpose |
|------|---------|
| `graph_0.c` | Main kernel implementation |
| `graph_0.h` | Header with tensor definitions |
| `graph_0_weights.c` | Weight data as C arrays |
| `graph_0_weights.bin` | Binary weight data |
| `graph_0.ld` | Linker script for bare-metal |
| `Makefile` | Build configuration |

**Code Generation for MATMUL:**

```python
def _gen_matmul(self, op: ACALSimOp, ir: ACALSimIR) -> List[str]:
    """Generate matrix multiplication code."""
    a_tensor = ir.tensors[op.inputs[0]]
    b_tensor = ir.tensors[op.inputs[1]]

    M = a_tensor.shape[0]  # 8
    K = a_tensor.shape[1]  # 16
    N = b_tensor.shape[1]  # 8

    return [
        f"    // MATMUL: [{M}x{K}] @ [{K}x{N}] -> [{M}x{N}]",
        f"    for (int m = 0; m < {M}; m++) {{",
        f"        for (int n = 0; n < {N}; n++) {{",
        f"            float sum = 0.0f;",
        f"            for (int k = 0; k < {K}; k++) {{",
        f"                sum += {a_safe}[m * {K} + k] * {b_safe}[k * {N} + n];",
        f"            }}",
        f"            {out_safe}[m * {N} + n] = sum;",
        f"        }}",
        f"    }}",
    ]
```

---

## 5. SST Component Communication Protocol

### 5.1 Device Memory Map

The generated code communicates with SST accelerator components via Memory-Mapped I/O (MMIO):

```
┌─────────────────────────────────────────────────────────┐
│              RISC-V Physical Memory Map                  │
├─────────────────────────────────────────────────────────┤
│ 0x80000000 - 0x8FFFFFFF  │  RAM (program + data)        │
├─────────────────────────────────────────────────────────┤
│ 0x10200000 - 0x10200FFF  │  Echo Device (Device 0)      │
├─────────────────────────────────────────────────────────┤
│ 0x10300000 - 0x10300FFF  │  Compute Device (Device 1)   │
├─────────────────────────────────────────────────────────┤
│ 0x10400000+              │  Additional Devices          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Device Register Layout

Each device has a standard register interface:

| Offset | Register | R/W | Description |
|--------|----------|-----|-------------|
| 0x00 | DATA_IN | W | Write data to device |
| 0x04 | DATA_OUT | R | Read response data |
| 0x08 | STATUS | R | Device status flags |
| 0x0C | CONTROL | RW | Control operations |
| 0x10 | SIZE | RW | Data size register |
| 0x14 | RESULT | R | Operation result |

**Status Register Bits:**
- Bit 0: BUSY - Device processing
- Bit 1: DATA_READY - Response available
- Bit 2: ERROR - Operation failed

### 5.3 Generated MMIO Code

The generated C code includes MMIO helpers:

```c
// graph_0.c (generated)

// MMIO helper macros
#define MMIO_WRITE32(addr, val) (*(volatile uint32_t*)(addr) = (val))
#define MMIO_READ32(addr) (*(volatile uint32_t*)(addr))

// Device base addresses
#define ECHO_DEVICE_BASE    0x10200000
#define COMPUTE_DEVICE_BASE 0x10300000

// Register offsets
#define REG_CMD      0x00
#define REG_STATUS   0x04
#define REG_DATA_IN  0x08
#define REG_DATA_OUT 0x0C

// Wait for device to be ready
static inline void wait_device_ready(void) {
    while (MMIO_READ32(COMPUTE_DEVICE_BASE + REG_STATUS) != 0) {
        // Busy wait - SST simulates the latency
    }
}
```

### 5.4 QEMU-SST Communication Flow

When the bare-metal program writes to an MMIO address:

```
┌───────────────────────────────────────────────────────────────────────┐
│ Step 1: Bare-Metal MMIO Write                                         │
│                                                                        │
│   // C code in generated kernel                                        │
│   MMIO_WRITE32(COMPUTE_DEVICE_BASE + REG_DATA_IN, 0xDEADBEEF);        │
│                                                                        │
│   This compiles to RISC-V store instruction:                          │
│   sw a0, 0(a1)   // Store to address 0x10300008                       │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Step 2: QEMU Intercepts MMIO Access                                   │
│                                                                        │
│   // qemu-sst-device/sst-device.c                                     │
│   static void sst_device_write(void *opaque, hwaddr addr,             │
│                                uint64_t val, unsigned size) {         │
│       // Create binary request                                         │
│       struct MMIORequest req = {                                       │
│           .type = 1,        // WRITE                                   │
│           .size = size,     // 4 bytes                                 │
│           .addr = addr,     // 0x10300008                              │
│           .data = val       // 0xDEADBEEF                              │
│       };                                                               │
│       // Send to SST via Unix socket                                   │
│       send(socket_fd, &req, sizeof(req), 0);                          │
│   }                                                                    │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Unix Socket
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Step 3: SST QEMUBinaryComponent Receives Request                      │
│                                                                        │
│   // QEMUBinaryComponent.cc                                           │
│   void handleSocketData() {                                            │
│       MMIORequest req;                                                 │
│       recv(client_fd, &req, sizeof(req), 0);                          │
│                                                                        │
│       // Convert to SST event                                          │
│       auto* trans = new MemoryTransactionEvent(                        │
│           TransactionType::STORE,                                      │
│           req.addr,        // 0x10300008                               │
│           req.data,        // 0xDEADBEEF                               │
│           req.size,        // 4                                        │
│           next_req_id_++   // Unique ID for tracking                   │
│       );                                                               │
│                                                                        │
│       // Route to appropriate device link                              │
│       device_link_->send(trans);                                       │
│   }                                                                    │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ SST Link
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Step 4: ACALSimDeviceComponent Processes Transaction                  │
│                                                                        │
│   // ACALSimDeviceComponent.cc                                        │
│   void handleTransaction(MemoryTransactionEvent* trans) {              │
│       uint64_t offset = trans->address - base_addr_;  // 0x08          │
│                                                                        │
│       if (trans->type == STORE) {                                      │
│           writeRegister(offset, trans->data);                          │
│           // REG_DATA_IN: store data and mark busy                     │
│           if (offset == REG_DATA_IN) {                                 │
│               data_in_ = trans->data;                                  │
│               status_ |= STATUS_BUSY;                                  │
│               // Schedule completion after compute_latency cycles      │
│               scheduleCompletion(compute_latency_);                    │
│           }                                                            │
│       }                                                                │
│                                                                        │
│       // Send response                                                 │
│       auto* resp = new MemoryResponseEvent(trans->req_id, 0, true);   │
│       link_->send(resp);                                               │
│   }                                                                    │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ SST Link (response)
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Step 5: Response Propagates Back to QEMU                              │
│                                                                        │
│   QEMUBinaryComponent converts MemoryResponseEvent to MMIOResponse    │
│   Sends via Unix socket back to QEMU                                   │
│   QEMU's sst_device_write() completes                                  │
│   RISC-V CPU continues execution                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 5.5 Binary Protocol Structures

```c
// QEMU -> SST Request (24 bytes)
struct MMIORequest {
    uint8_t  type;        // 0=READ, 1=WRITE
    uint8_t  size;        // 1, 2, 4, or 8 bytes
    uint16_t reserved;
    uint64_t addr;        // Global MMIO address
    uint64_t data;        // Write data (for stores)
} __attribute__((packed));

// SST -> QEMU Response (16 bytes)
struct MMIOResponse {
    uint8_t  success;     // 0=error, 1=success
    uint8_t  reserved[7];
    uint64_t data;        // Read result (for loads)
} __attribute__((packed));
```

---

## 6. Generated Code Analysis

### 6.1 Complete Generated Kernel (graph_0.c)

```c
// Auto-generated by ACALSim PyTorch Backend
// Graph: graph_0

#include "graph_0.h"
#include <stddef.h>

// MMIO helper macros
#define MMIO_WRITE32(addr, val) (*(volatile uint32_t*)(addr) = (val))
#define MMIO_READ32(addr) (*(volatile uint32_t*)(addr))

// Wait for device to be ready
static inline void wait_device_ready(void) {
    while (MMIO_READ32(COMPUTE_DEVICE_BASE + REG_STATUS) != 0) {
        // Busy wait
    }
}

// Tensor buffers (intermediate results)
static float input_0[128];    // [8, 16] = 128 elements
static float t_2[64];         // [8, 8] = 64 elements (matmul output)
static float t_4[64];         // [8, 8] = 64 elements (final output)

// External weight data (from graph_0_weights.c)
extern const float param_1[];  // Weight [16, 8]
extern const float param_3[];  // Bias [8]

void graph_0_init(void) {
    // Initialize device
    MMIO_WRITE32(COMPUTE_DEVICE_BASE + REG_CMD, CMD_NOP);
    wait_device_ready();
}

void graph_0_forward(float* inputs, float* outputs) {
    // Copy input to input_0
    for (int i = 0; i < 128; i++) {
        input_0[i] = inputs[i];
    }

    // Op 0: t_2 = MATMUL(input_0, param_1)
    // MATMUL: [8x16] @ [16x8] -> [8x8]
    for (int m = 0; m < 8; m++) {
        for (int n = 0; n < 8; n++) {
            float sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                sum += input_0[m * 16 + k] * param_1[k * 8 + n];
            }
            t_2[m * 8 + n] = sum;
        }
    }

    // Op 1: t_4 = ADD(t_2, param_3)
    for (int i = 0; i < 64; i++) {
        t_4[i] = t_2[i] + param_3[i % 8];  // Broadcast bias
    }

    // Copy output from t_4
    for (int i = 0; i < 64; i++) {
        outputs[i] = t_4[i];
    }
}

void graph_0_cleanup(void) {
    // Cleanup - nothing to do for static buffers
}

// Main function for standalone execution
int main(void) {
    static float inputs[128];
    static float outputs[64];

    // Initialize with test data
    for (int i = 0; i < 128; i++) {
        inputs[i] = (float)(i % 10) / 10.0f;
    }

    graph_0_init();
    graph_0_forward(inputs, outputs);
    graph_0_cleanup();

    // Signal completion via MMIO
    MMIO_WRITE32(ECHO_DEVICE_BASE, 0xDEADBEEF);

    return 0;
}
```

### 6.2 Generated Weight Data (graph_0_weights.c)

```c
// Auto-generated weight data
#include "graph_0.h"

// Weight tensor: param_1 [16, 8] = 128 elements
const float param_1[128] = {
    0.12345678f, -0.23456789f, 0.34567890f, ...
    // ... (128 float values from PyTorch model)
};

// Bias tensor: param_3 [8] = 8 elements
const float param_3[8] = {
    0.01234567f, 0.02345678f, 0.03456789f, ...
};
```

### 6.3 Generated Linker Script (graph_0.ld)

```ld
/* Auto-generated linker script for ACALSim kernel */

ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x100000
}

SECTIONS
{
    .text : {
        _start = .;
        *(.text.init)
        *(.text .text.*)
    } > RAM

    .rodata : {
        *(.rodata .rodata.*)
    } > RAM

    .data : {
        *(.data .data.*)
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    .stack : {
        . = ALIGN(16);
        . = . + 0x4000;  /* 16KB stack */
        __stack_top = .;
    } > RAM
}
```

---

## 7. Running the Complete Pipeline

### 7.1 Step 1: Generate Code

```bash
cd src/pytorch-acalsim-backend

# Run the matmul demo
python examples/matmul_demo.py

# Output files in examples/output_matmul/
ls examples/output_matmul/
# graph_0.c
# graph_0.h
# graph_0_weights.c
# graph_0_weights.bin
# graph_0.ld
# Makefile
```

### 7.2 Step 2: Cross-Compile for RISC-V

```bash
cd examples/output_matmul

# Requires RISC-V toolchain
export CROSS_COMPILE=riscv64-unknown-elf-

# Build the ELF binary
make

# Generated files:
# graph_0.elf   - Executable
# graph_0.bin   - Raw binary
# graph_0.dump  - Disassembly
```

### 7.3 Step 3: Run on QEMU-SST Simulator

```bash
# From the qemu-acalsim-sst-baremetal directory
cd src/qemu-acalsim-sst-baremetal

# Start SST simulation (in one terminal)
sst tests/test_config.py

# Start QEMU with the kernel (in another terminal)
qemu-system-riscv64 \
    -M virt \
    -nographic \
    -bios none \
    -kernel path/to/graph_0.elf \
    -device sst-device,socket=/tmp/sst.sock

# SST output will show:
# - Cycle-accurate timing
# - Memory access patterns
# - Device register accesses
```

### 7.4 Expected SST Output

```
SST Simulation Started
[Cycle 0] QEMUBinaryComponent: Client connected
[Cycle 100] ACALSimDevice: WRITE to REG_DATA_IN = 0x3F800000
[Cycle 101] ACALSimDevice: Status -> BUSY
[Cycle 111] ACALSimDevice: Echo complete, latency=10 cycles
[Cycle 111] ACALSimDevice: Status -> DATA_READY
...
[Cycle 50000] ACALSimDevice: WRITE to ECHO_DEVICE = 0xDEADBEEF
[Cycle 50000] Simulation complete

Statistics:
  Total cycles: 50000
  Memory reads: 1024
  Memory writes: 512
  Device transactions: 256
```

---

## Summary

The PyTorch-ACALSim backend provides a complete flow from high-level PyTorch models to cycle-accurate hardware simulation:

1. **TorchDynamo** captures the computation graph
2. **ACALSimCompiler** converts to accelerator-specific IR
3. **RISCVCodeGenerator** produces bare-metal C code
4. **RISC-V toolchain** compiles to executable
5. **QEMU** runs the code with MMIO interception
6. **SST** simulates accelerator timing with cycle accuracy

This enables hardware-software co-design where:
- ML researchers can test models on simulated hardware
- Hardware designers can validate accelerator designs with real workloads
- Performance analysis is cycle-accurate and reproducible
