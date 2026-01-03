#!/usr/bin/env python3
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
"""Example: Compile a simple MLP to ACALSim IR.

This example demonstrates how to use the ACALSim backend with torch.compile()
to capture and compile a simple Multi-Layer Perceptron (MLP) model.
"""

import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from acalsim_backend import ACALSimBackend, make_acalsim_backend


class SimpleMLP(nn.Module):
    """A simple 3-layer MLP for demonstration."""

    def __init__(self, input_size=16, hidden_size=32, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    print("=" * 60)
    print("ACALSim Backend Demo: Simple MLP")
    print("=" * 60)

    # Create model
    model = SimpleMLP(input_size=16, hidden_size=32, output_size=10)
    model.eval()

    # Create example input
    batch_size = 4
    x = torch.randn(batch_size, 16)

    print(f"\nModel: {model}")
    print(f"Input shape: {x.shape}")

    # Mode 1: Trace mode - just print the FX graph
    print("\n" + "=" * 60)
    print("Mode 1: TRACE - Print FX Graph")
    print("=" * 60)

    backend_trace = ACALSimBackend(mode="trace", verbose=True)
    compiled_trace = torch.compile(model, backend=backend_trace)

    with torch.no_grad():
        output_trace = compiled_trace(x)

    print(f"Output shape: {output_trace.shape}")

    # Mode 2: Compile mode - generate ACALSim IR
    print("\n" + "=" * 60)
    print("Mode 2: COMPILE - Generate ACALSim IR")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), "output_mlp")
    os.makedirs(output_dir, exist_ok=True)

    backend_compile = ACALSimBackend(
        mode="compile",
        verbose=True,
        output_dir=output_dir,
    )

    # Need fresh model for new compilation
    model2 = SimpleMLP(input_size=16, hidden_size=32, output_size=10)
    model2.eval()

    compiled_model = torch.compile(model2, backend=backend_compile)

    with torch.no_grad():
        output_compile = compiled_model(x)

    print(f"\nOutput shape: {output_compile.shape}")
    print(f"IR files saved to: {output_dir}")

    # Mode 3: Codegen mode - generate RISC-V code
    print("\n" + "=" * 60)
    print("Mode 3: CODEGEN - Generate RISC-V Code")
    print("=" * 60)

    codegen_dir = os.path.join(os.path.dirname(__file__), "output_mlp_riscv")
    os.makedirs(codegen_dir, exist_ok=True)

    backend_codegen = ACALSimBackend(
        mode="codegen",
        verbose=True,
        output_dir=codegen_dir,
    )

    model3 = SimpleMLP(input_size=16, hidden_size=32, output_size=10)
    model3.eval()

    compiled_codegen = torch.compile(model3, backend=backend_codegen)

    with torch.no_grad():
        output_codegen = compiled_codegen(x)

    print(f"\nRISC-V code saved to: {codegen_dir}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(codegen_dir)):
        filepath = os.path.join(codegen_dir, f)
        size = os.path.getsize(filepath)
        print(f"  {f}: {size} bytes")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
