# Copyright 2023-2025 Playlab/ACAL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ACALSim Intermediate Representation (IR) for compiled graphs.

This module defines the IR that represents operations to be executed on
the ACALSim accelerator. The IR is generated from PyTorch FX graphs and
can be lowered to RISC-V bare-metal code.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ACALSimOpType(Enum):
    """Operation types supported by the ACALSim accelerator."""

    # Memory operations
    LOAD = auto()  # Load data from memory
    STORE = auto()  # Store data to memory
    ALLOC = auto()  # Allocate memory buffer

    # Arithmetic operations (element-wise)
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()

    # Matrix operations
    MATMUL = auto()  # Matrix multiplication
    GEMM = auto()  # General matrix multiply (alpha*A@B + beta*C)
    CONV2D = auto()  # 2D convolution

    # Reduction operations
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()

    # Activation functions
    RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()
    GELU = auto()

    # Shape operations
    RESHAPE = auto()
    TRANSPOSE = auto()
    PERMUTE = auto()
    FLATTEN = auto()
    UNSQUEEZE = auto()
    SQUEEZE = auto()

    # Normalization
    BATCH_NORM = auto()
    LAYER_NORM = auto()

    # Control flow
    NOP = auto()  # No operation
    SYNC = auto()  # Synchronization barrier

    # Custom/unknown
    CUSTOM = auto()


@dataclass
class TensorDesc:
    """Description of a tensor in the IR."""

    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"  # float32, float16, int32, int8
    is_input: bool = False
    is_output: bool = False
    is_param: bool = False  # Weights/biases
    memory_addr: Optional[int] = None  # Assigned during code generation

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        """Size in bytes."""
        dtype_sizes = {"float32": 4, "float16": 2, "int32": 4, "int8": 1}
        return self.numel * dtype_sizes.get(self.dtype, 4)


@dataclass
class ACALSimOp:
    """A single operation in the ACALSim IR."""

    op_type: ACALSimOpType
    name: str
    inputs: List[str]  # Names of input tensors
    outputs: List[str]  # Names of output tensors
    attrs: Dict[str, Any] = field(default_factory=dict)

    # Scheduling info (filled during compilation)
    compute_cycles: int = 1
    memory_cycles: int = 0

    def __repr__(self) -> str:
        inputs_str = ", ".join(self.inputs)
        outputs_str = ", ".join(self.outputs)
        return f"{outputs_str} = {self.op_type.name}({inputs_str})"


@dataclass
class ACALSimIR:
    """Complete IR representation of a compiled graph.

    This IR can be:
    1. Printed for debugging
    2. Optimized (fusion, scheduling)
    3. Lowered to RISC-V code
    """

    name: str
    tensors: Dict[str, TensorDesc] = field(default_factory=dict)
    ops: List[ACALSimOp] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    # Constants (weights, biases) - stored as numpy arrays
    constants: Dict[str, np.ndarray] = field(default_factory=dict)

    def add_tensor(self, desc: TensorDesc) -> None:
        """Add a tensor to the IR."""
        self.tensors[desc.name] = desc

    def add_op(self, op: ACALSimOp) -> None:
        """Add an operation to the IR."""
        self.ops.append(op)

    def get_tensor(self, name: str) -> Optional[TensorDesc]:
        """Get tensor by name."""
        return self.tensors.get(name)

    def print_ir(self) -> str:
        """Generate human-readable IR representation."""
        lines = [f"=== ACALSim IR: {self.name} ===", ""]

        # Print tensors
        lines.append("Tensors:")
        for name, tensor in self.tensors.items():
            flags = []
            if tensor.is_input:
                flags.append("input")
            if tensor.is_output:
                flags.append("output")
            if tensor.is_param:
                flags.append("param")
            flags_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  {name}: {tensor.shape} {tensor.dtype}{flags_str}")

        lines.append("")
        lines.append("Operations:")
        for i, op in enumerate(self.ops):
            lines.append(f"  {i}: {op}")
            if op.attrs:
                for k, v in op.attrs.items():
                    lines.append(f"       {k}={v}")

        lines.append("")
        lines.append(f"Inputs: {self.input_names}")
        lines.append(f"Outputs: {self.output_names}")

        return "\n".join(lines)

    def estimate_cycles(self) -> int:
        """Estimate total execution cycles."""
        return sum(op.compute_cycles + op.memory_cycles for op in self.ops)

    def estimate_memory(self) -> int:
        """Estimate total memory required in bytes."""
        return sum(t.size_bytes for t in self.tensors.values())


# Mapping from PyTorch FX ops to ACALSim ops
FX_TO_ACALSIM_OP = {
    # aten operations
    "add": ACALSimOpType.ADD,
    "sub": ACALSimOpType.SUB,
    "mul": ACALSimOpType.MUL,
    "div": ACALSimOpType.DIV,
    "neg": ACALSimOpType.NEG,
    "mm": ACALSimOpType.MATMUL,
    "bmm": ACALSimOpType.MATMUL,
    "matmul": ACALSimOpType.MATMUL,
    "linear": ACALSimOpType.GEMM,
    "addmm": ACALSimOpType.GEMM,
    "conv2d": ACALSimOpType.CONV2D,
    "sum": ACALSimOpType.SUM,
    "mean": ACALSimOpType.MEAN,
    "max": ACALSimOpType.MAX,
    "min": ACALSimOpType.MIN,
    "relu": ACALSimOpType.RELU,
    "sigmoid": ACALSimOpType.SIGMOID,
    "tanh": ACALSimOpType.TANH,
    "softmax": ACALSimOpType.SOFTMAX,
    "gelu": ACALSimOpType.GELU,
    "reshape": ACALSimOpType.RESHAPE,
    "view": ACALSimOpType.RESHAPE,
    "transpose": ACALSimOpType.TRANSPOSE,
    "permute": ACALSimOpType.PERMUTE,
    "flatten": ACALSimOpType.FLATTEN,
    "unsqueeze": ACALSimOpType.UNSQUEEZE,
    "squeeze": ACALSimOpType.SQUEEZE,
    "batch_norm": ACALSimOpType.BATCH_NORM,
    "layer_norm": ACALSimOpType.LAYER_NORM,
}


def get_acalsim_op_type(fx_op_name: str) -> ACALSimOpType:
    """Convert FX operation name to ACALSim operation type."""
    # Handle aten:: prefix
    if fx_op_name.startswith("aten."):
        fx_op_name = fx_op_name[5:]
    if fx_op_name.startswith("aten::"):
        fx_op_name = fx_op_name[6:]

    # Remove any suffix like .default
    if "." in fx_op_name:
        fx_op_name = fx_op_name.split(".")[0]

    return FX_TO_ACALSIM_OP.get(fx_op_name, ACALSimOpType.CUSTOM)
