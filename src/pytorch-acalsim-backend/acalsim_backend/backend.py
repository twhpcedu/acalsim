# Copyright 2023-2026 Playlab/ACAL
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
"""TorchDynamo backend for ACALSim accelerator simulation.

This module provides the custom backend for torch.compile() that:
1. Captures FX graphs from TorchDynamo
2. Compiles them to ACALSim IR
3. Generates RISC-V code for the accelerator
4. Provides execution on the simulator or fallback to CPU
"""

from typing import Any, Callable, Dict, List, Optional, Union
import logging
import os
import tempfile

import torch
import torch.fx as fx

from .compiler import ACALSimCompiler
from .ir import ACALSimIR

logger = logging.getLogger(__name__)


class ACALSimBackend:
    """Custom TorchDynamo backend for ACALSim.

    This backend captures PyTorch operations and compiles them for the
    ACALSim accelerator simulator. It can operate in multiple modes:

    Modes:
        - "trace": Just trace and print the captured graph (debugging)
        - "compile": Compile to ACALSim IR only
        - "codegen": Generate RISC-V code
        - "simulate": Run on QEMU-SST simulator (requires setup)
        - "eager": Fallback to eager PyTorch execution

    Usage:
        backend = ACALSimBackend(mode="compile", verbose=True)
        compiled_model = torch.compile(model, backend=backend)
    """

    def __init__(
        self,
        mode: str = "compile",
        verbose: bool = False,
        output_dir: Optional[str] = None,
        qemu_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the ACALSim backend.

        Args:
            mode: Operating mode ("trace", "compile", "codegen", "simulate", "eager")
            verbose: Print detailed compilation info
            output_dir: Directory for generated code (default: tempdir)
            qemu_config: Configuration for QEMU-SST simulation
        """
        self.mode = mode
        self.verbose = verbose
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="acalsim_")
        self.qemu_config = qemu_config or {}

        self.compiler = ACALSimCompiler(verbose=verbose)

        # Cache for compiled graphs
        self._compiled_cache: Dict[str, Any] = {}
        self._graph_counter = 0

    def __call__(
        self,
        gm: fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> Callable[..., Any]:
        """Main entry point called by TorchDynamo.

        Args:
            gm: The captured FX GraphModule
            example_inputs: Example input tensors for shape inference

        Returns:
            A callable that executes the compiled graph
        """
        graph_name = f"graph_{self._graph_counter}"
        self._graph_counter += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ACALSim Backend: Captured {graph_name}")
            print(f"{'='*60}")
            print(f"Number of nodes: {len(gm.graph.nodes)}")
            print(f"Input shapes: {[tuple(x.shape) for x in example_inputs]}")

        if self.mode == "trace":
            return self._trace_mode(gm, example_inputs, graph_name)
        elif self.mode == "compile":
            return self._compile_mode(gm, example_inputs, graph_name)
        elif self.mode == "codegen":
            return self._codegen_mode(gm, example_inputs, graph_name)
        elif self.mode == "simulate":
            return self._simulate_mode(gm, example_inputs, graph_name)
        else:  # eager
            return self._eager_mode(gm, example_inputs, graph_name)

    def _trace_mode(
        self,
        gm: fx.GraphModule,
        example_inputs: List[torch.Tensor],
        graph_name: str,
    ) -> Callable[..., Any]:
        """Trace mode: just print the graph and run eagerly."""
        print(f"\n=== FX Graph for {graph_name} ===")
        try:
            gm.graph.print_tabular()
        except ImportError:
            # tabulate not installed, print simple node list
            print("Graph nodes:")
            for node in gm.graph.nodes:
                print(f"  {node.op}: {node.name} -> {node.target}")
        print("\nGraph code:")
        print(gm.code)

        # Return the original forward function
        return gm.forward

    def _compile_mode(
        self,
        gm: fx.GraphModule,
        example_inputs: List[torch.Tensor],
        graph_name: str,
    ) -> Callable[..., Any]:
        """Compile mode: generate IR and print analysis."""
        # Compile to ACALSim IR
        ir = self.compiler.compile(gm, example_inputs, graph_name)

        # Cache the IR
        self._compiled_cache[graph_name] = ir

        if self.verbose:
            print(f"\nEstimated cycles: {ir.estimate_cycles()}")
            print(f"Estimated memory: {ir.estimate_memory()} bytes")

        # Save IR to file
        ir_file = os.path.join(self.output_dir, f"{graph_name}.ir")
        with open(ir_file, "w") as f:
            f.write(ir.print_ir())

        if self.verbose:
            print(f"IR saved to: {ir_file}")

        # Return eager execution (we compiled but execute on CPU)
        return gm.forward

    def _codegen_mode(
        self,
        gm: fx.GraphModule,
        example_inputs: List[torch.Tensor],
        graph_name: str,
    ) -> Callable[..., Any]:
        """Codegen mode: generate RISC-V code."""
        # First compile to IR
        ir = self.compiler.compile(gm, example_inputs, graph_name)
        self._compiled_cache[graph_name] = ir

        # Import code generator (lazy import to avoid circular deps)
        from codegen.riscv_codegen import RISCVCodeGenerator

        codegen = RISCVCodeGenerator(output_dir=self.output_dir)
        code_files = codegen.generate(ir, graph_name)

        if self.verbose:
            print(f"\nGenerated RISC-V code files:")
            for f in code_files:
                print(f"  {f}")

        # Return eager execution
        return gm.forward

    def _simulate_mode(
        self,
        gm: fx.GraphModule,
        example_inputs: List[torch.Tensor],
        graph_name: str,
    ) -> Callable[..., Any]:
        """Simulate mode: run on QEMU-SST (not yet implemented)."""
        # This would:
        # 1. Generate code
        # 2. Compile with RISC-V toolchain
        # 3. Launch QEMU-SST simulation
        # 4. Collect results

        logger.warning("Simulate mode not yet implemented, using eager mode")
        return self._compile_mode(gm, example_inputs, graph_name)

    def _eager_mode(
        self,
        gm: fx.GraphModule,
        example_inputs: List[torch.Tensor],
        graph_name: str,
    ) -> Callable[..., Any]:
        """Eager mode: just run on CPU."""
        if self.verbose:
            print("Using eager execution (no compilation)")
        return gm.forward

    def get_compiled_ir(self, graph_name: str) -> Optional[ACALSimIR]:
        """Get the compiled IR for a graph."""
        return self._compiled_cache.get(graph_name)

    def get_all_compiled_irs(self) -> Dict[str, ACALSimIR]:
        """Get all compiled IRs."""
        return self._compiled_cache.copy()


# Default backend instance for convenience
_default_backend: Optional[ACALSimBackend] = None


def acalsim_backend(
    gm: fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> Callable[..., Any]:
    """Default ACALSim backend function for torch.compile().

    This function provides a simple way to use the ACALSim backend:

        compiled_model = torch.compile(model, backend=acalsim_backend)

    For more control, create an ACALSimBackend instance directly.
    """
    global _default_backend
    if _default_backend is None:
        _default_backend = ACALSimBackend(
            mode="compile",
            verbose=True,
        )
    return _default_backend(gm, example_inputs)


def make_acalsim_backend(
    mode: str = "compile",
    verbose: bool = False,
    output_dir: Optional[str] = None,
) -> Callable[[fx.GraphModule, List[torch.Tensor]], Callable[..., Any]]:
    """Factory function to create a configured ACALSim backend.

    Usage:
        backend = make_acalsim_backend(mode="codegen", verbose=True)
        compiled_model = torch.compile(model, backend=backend)

    Args:
        mode: Operating mode
        verbose: Print compilation details
        output_dir: Directory for generated files

    Returns:
        A backend function compatible with torch.compile()
    """
    backend_instance = ACALSimBackend(
        mode=mode,
        verbose=verbose,
        output_dir=output_dir,
    )
    return backend_instance
