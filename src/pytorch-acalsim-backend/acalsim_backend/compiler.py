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
"""Compiler that converts PyTorch FX graphs to ACALSim IR.

This module provides the main compilation pipeline:
1. FX Graph Analysis - extract tensor shapes and operations
2. IR Generation - convert FX nodes to ACALSim ops
3. Optimization - fuse ops, schedule for accelerator
4. Code Generation - produce RISC-V assembly
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.fx as fx
import numpy as np

from .ir import (
    ACALSimIR,
    ACALSimOp,
    ACALSimOpType,
    TensorDesc,
    get_acalsim_op_type,
)

logger = logging.getLogger(__name__)


class ACALSimCompiler:
    """Compiles PyTorch FX graphs to ACALSim IR.

    This compiler:
    1. Walks the FX graph to extract operations
    2. Infers tensor shapes from example inputs
    3. Generates ACALSim IR operations
    4. Extracts constants (weights/biases) for embedding
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._tensor_counter = 0
        self._op_counter = 0

    def compile(
        self,
        gm: fx.GraphModule,
        example_inputs: List[torch.Tensor],
        graph_name: str = "graph",
    ) -> ACALSimIR:
        """Compile an FX GraphModule to ACALSim IR.

        Args:
            gm: The FX GraphModule captured by TorchDynamo
            example_inputs: Example input tensors (for shape inference)
            graph_name: Name for this compiled graph

        Returns:
            ACALSimIR containing the lowered operations
        """
        ir = ACALSimIR(name=graph_name)

        # Reset counters
        self._tensor_counter = 0
        self._op_counter = 0

        # Map from FX node names to IR tensor names
        node_to_tensor: Dict[str, str] = {}

        # First pass: run shape propagation
        shape_info = self._propagate_shapes(gm, example_inputs)

        if self.verbose:
            print(f"\n=== Compiling graph: {graph_name} ===")
            try:
                gm.graph.print_tabular()
            except ImportError:
                # tabulate not installed, print simple node list
                print("Graph nodes:")
                for node in gm.graph.nodes:
                    print(f"  {node.op}: {node.name} -> {node.target}")

        # Process each node in topological order
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                # Input tensor
                tensor_name = self._new_tensor_name("input")
                shape = shape_info.get(node.name, (1,))
                dtype = self._get_dtype(shape_info, node.name)

                desc = TensorDesc(
                    name=tensor_name,
                    shape=shape,
                    dtype=dtype,
                    is_input=True,
                )
                ir.add_tensor(desc)
                ir.input_names.append(tensor_name)
                node_to_tensor[node.name] = tensor_name

                if self.verbose:
                    print(f"  Input: {node.name} -> {tensor_name} {shape}")

            elif node.op == "get_attr":
                # Model parameter (weight, bias)
                tensor_name = self._new_tensor_name("param")
                param = self._get_attr_value(gm, node.target)

                if param is not None:
                    shape = tuple(param.shape)
                    dtype = self._torch_dtype_to_str(param.dtype)

                    desc = TensorDesc(
                        name=tensor_name,
                        shape=shape,
                        dtype=dtype,
                        is_param=True,
                    )
                    ir.add_tensor(desc)

                    # Store the constant value
                    ir.constants[tensor_name] = param.detach().cpu().numpy()
                    node_to_tensor[node.name] = tensor_name

                    if self.verbose:
                        print(f"  Param: {node.name} -> {tensor_name} {shape}")

            elif node.op == "call_function":
                # Operation node
                self._compile_call_function(
                    node, ir, node_to_tensor, shape_info
                )

            elif node.op == "call_method":
                # Method call (e.g., tensor.view())
                self._compile_call_method(node, ir, node_to_tensor, shape_info)

            elif node.op == "call_module":
                # Module call (e.g., self.linear)
                self._compile_call_module(
                    gm, node, ir, node_to_tensor, shape_info
                )

            elif node.op == "output":
                # Mark output tensors
                output_args = node.args[0]
                if isinstance(output_args, (tuple, list)):
                    for arg in output_args:
                        if hasattr(arg, "name") and arg.name in node_to_tensor:
                            tensor_name = node_to_tensor[arg.name]
                            ir.output_names.append(tensor_name)
                            if tensor_name in ir.tensors:
                                ir.tensors[tensor_name].is_output = True
                elif hasattr(output_args, "name"):
                    tensor_name = node_to_tensor.get(output_args.name)
                    if tensor_name:
                        ir.output_names.append(tensor_name)
                        if tensor_name in ir.tensors:
                            ir.tensors[tensor_name].is_output = True

        if self.verbose:
            print("\n" + ir.print_ir())

        return ir

    def _propagate_shapes(
        self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]
    ) -> Dict[str, Tuple[int, ...]]:
        """Run the graph with example inputs to get tensor shapes."""
        shape_info = {}

        # Run shape propagation using torch.fx.passes
        try:
            from torch.fx.passes.shape_prop import ShapeProp

            sp = ShapeProp(gm)
            sp.propagate(*example_inputs)

            for node in gm.graph.nodes:
                if hasattr(node, "meta") and "tensor_meta" in node.meta:
                    meta = node.meta["tensor_meta"]
                    if hasattr(meta, "shape"):
                        shape_info[node.name] = tuple(meta.shape)
                    elif isinstance(meta, torch.Tensor):
                        shape_info[node.name] = tuple(meta.shape)

        except Exception as e:
            logger.warning(f"Shape propagation failed: {e}, using fallback")
            # Fallback: just record input shapes
            for i, node in enumerate(gm.graph.nodes):
                if node.op == "placeholder" and i < len(example_inputs):
                    shape_info[node.name] = tuple(example_inputs[i].shape)

        return shape_info

    def _compile_call_function(
        self,
        node: fx.Node,
        ir: ACALSimIR,
        node_to_tensor: Dict[str, str],
        shape_info: Dict[str, Tuple[int, ...]],
    ) -> None:
        """Compile a call_function node to ACALSim IR."""
        # Get operation name
        target = node.target
        if hasattr(target, "__name__"):
            op_name = target.__name__
        elif hasattr(target, "name"):
            op_name = target.name()
        else:
            op_name = str(target)

        # Map to ACALSim op type
        acalsim_op_type = get_acalsim_op_type(op_name)

        # Collect input tensor names
        input_names = []
        for arg in node.args:
            if hasattr(arg, "name") and arg.name in node_to_tensor:
                input_names.append(node_to_tensor[arg.name])
            elif isinstance(arg, (int, float)):
                # Scalar constant - create a constant tensor
                const_name = self._new_tensor_name("const")
                desc = TensorDesc(
                    name=const_name,
                    shape=(1,),
                    dtype="float32",
                    is_param=True,
                )
                ir.add_tensor(desc)
                ir.constants[const_name] = np.array([arg], dtype=np.float32)
                input_names.append(const_name)

        # Create output tensor
        output_name = self._new_tensor_name("t")
        output_shape = shape_info.get(node.name, (1,))

        desc = TensorDesc(
            name=output_name,
            shape=output_shape,
            dtype="float32",
        )
        ir.add_tensor(desc)
        node_to_tensor[node.name] = output_name

        # Extract attributes from kwargs
        attrs = {}
        for key, value in node.kwargs.items():
            if isinstance(value, (int, float, str, bool, tuple, list)):
                attrs[key] = value

        # Estimate compute cycles based on operation type
        compute_cycles = self._estimate_compute_cycles(
            acalsim_op_type, output_shape
        )

        # Create the operation
        op = ACALSimOp(
            op_type=acalsim_op_type,
            name=f"{op_name}_{self._op_counter}",
            inputs=input_names,
            outputs=[output_name],
            attrs=attrs,
            compute_cycles=compute_cycles,
        )
        self._op_counter += 1
        ir.add_op(op)

        if self.verbose:
            print(f"  Op: {op}")

    def _compile_call_method(
        self,
        node: fx.Node,
        ir: ACALSimIR,
        node_to_tensor: Dict[str, str],
        shape_info: Dict[str, Tuple[int, ...]],
    ) -> None:
        """Compile a call_method node to ACALSim IR."""
        method_name = node.target
        acalsim_op_type = get_acalsim_op_type(method_name)

        # First arg is the tensor being called on
        input_names = []
        if node.args and hasattr(node.args[0], "name"):
            if node.args[0].name in node_to_tensor:
                input_names.append(node_to_tensor[node.args[0].name])

        # Create output tensor
        output_name = self._new_tensor_name("t")
        output_shape = shape_info.get(node.name, (1,))

        desc = TensorDesc(
            name=output_name,
            shape=output_shape,
            dtype="float32",
        )
        ir.add_tensor(desc)
        node_to_tensor[node.name] = output_name

        # Extract shape arguments for reshape/view operations
        attrs = {}
        if method_name in ("view", "reshape"):
            shape_args = node.args[1:]
            if shape_args:
                attrs["new_shape"] = tuple(
                    arg if isinstance(arg, int) else -1 for arg in shape_args
                )

        op = ACALSimOp(
            op_type=acalsim_op_type,
            name=f"{method_name}_{self._op_counter}",
            inputs=input_names,
            outputs=[output_name],
            attrs=attrs,
        )
        self._op_counter += 1
        ir.add_op(op)

        if self.verbose:
            print(f"  Method: {op}")

    def _compile_call_module(
        self,
        gm: fx.GraphModule,
        node: fx.Node,
        ir: ACALSimIR,
        node_to_tensor: Dict[str, str],
        shape_info: Dict[str, Tuple[int, ...]],
    ) -> None:
        """Compile a call_module node to ACALSim IR."""
        # Get the actual module
        module = gm.get_submodule(node.target)
        module_type = type(module).__name__

        # Map module types to operations
        module_op_map = {
            "Linear": ACALSimOpType.GEMM,
            "Conv2d": ACALSimOpType.CONV2D,
            "BatchNorm2d": ACALSimOpType.BATCH_NORM,
            "LayerNorm": ACALSimOpType.LAYER_NORM,
            "ReLU": ACALSimOpType.RELU,
            "GELU": ACALSimOpType.GELU,
            "Sigmoid": ACALSimOpType.SIGMOID,
            "Tanh": ACALSimOpType.TANH,
            "Softmax": ACALSimOpType.SOFTMAX,
            "Flatten": ACALSimOpType.FLATTEN,
        }

        acalsim_op_type = module_op_map.get(module_type, ACALSimOpType.CUSTOM)

        # Collect input tensor names
        input_names = []
        for arg in node.args:
            if hasattr(arg, "name") and arg.name in node_to_tensor:
                input_names.append(node_to_tensor[arg.name])

        # Add module parameters as inputs
        for param_name, param in module.named_parameters():
            const_name = self._new_tensor_name(f"{node.target}_{param_name}")
            desc = TensorDesc(
                name=const_name,
                shape=tuple(param.shape),
                dtype=self._torch_dtype_to_str(param.dtype),
                is_param=True,
            )
            ir.add_tensor(desc)
            ir.constants[const_name] = param.detach().cpu().numpy()
            input_names.append(const_name)

        # Create output tensor
        output_name = self._new_tensor_name("t")
        output_shape = shape_info.get(node.name, (1,))

        desc = TensorDesc(
            name=output_name,
            shape=output_shape,
            dtype="float32",
        )
        ir.add_tensor(desc)
        node_to_tensor[node.name] = output_name

        # Extract module attributes
        attrs = {"module_type": module_type}
        if hasattr(module, "in_features"):
            attrs["in_features"] = module.in_features
        if hasattr(module, "out_features"):
            attrs["out_features"] = module.out_features

        compute_cycles = self._estimate_compute_cycles(
            acalsim_op_type, output_shape
        )

        op = ACALSimOp(
            op_type=acalsim_op_type,
            name=f"{module_type}_{self._op_counter}",
            inputs=input_names,
            outputs=[output_name],
            attrs=attrs,
            compute_cycles=compute_cycles,
        )
        self._op_counter += 1
        ir.add_op(op)

        if self.verbose:
            print(f"  Module: {op}")

    def _new_tensor_name(self, prefix: str = "t") -> str:
        """Generate a new unique tensor name."""
        name = f"{prefix}_{self._tensor_counter}"
        self._tensor_counter += 1
        return name

    def _get_attr_value(
        self, gm: fx.GraphModule, target: str
    ) -> Optional[torch.Tensor]:
        """Get the value of a module attribute."""
        try:
            parts = target.split(".")
            obj = gm
            for part in parts:
                obj = getattr(obj, part)
            if isinstance(obj, torch.Tensor):
                return obj
            elif isinstance(obj, torch.nn.Parameter):
                return obj.data
        except AttributeError:
            pass
        return None

    def _get_dtype(
        self, shape_info: Dict[str, Any], node_name: str
    ) -> str:
        """Get dtype string for a node."""
        # Default to float32
        return "float32"

    def _torch_dtype_to_str(self, dtype: torch.dtype) -> str:
        """Convert torch dtype to string."""
        dtype_map = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.int32: "int32",
            torch.int64: "int32",  # Downcast
            torch.int8: "int8",
            torch.bool: "int8",
        }
        return dtype_map.get(dtype, "float32")

    def _estimate_compute_cycles(
        self, op_type: ACALSimOpType, shape: Tuple[int, ...]
    ) -> int:
        """Estimate compute cycles for an operation."""
        numel = 1
        for dim in shape:
            numel *= dim

        # Simple cycle estimation based on operation type
        cycle_factors = {
            ACALSimOpType.ADD: 1,
            ACALSimOpType.SUB: 1,
            ACALSimOpType.MUL: 1,
            ACALSimOpType.DIV: 4,
            ACALSimOpType.MATMUL: 2,  # Per output element
            ACALSimOpType.GEMM: 2,
            ACALSimOpType.CONV2D: 4,
            ACALSimOpType.RELU: 1,
            ACALSimOpType.SIGMOID: 8,
            ACALSimOpType.TANH: 8,
            ACALSimOpType.SOFTMAX: 16,
            ACALSimOpType.GELU: 12,
        }

        factor = cycle_factors.get(op_type, 1)
        return numel * factor
