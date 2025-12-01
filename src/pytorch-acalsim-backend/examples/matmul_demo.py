#!/usr/bin/env python3
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
"""Example: Compile matrix multiplication operations to ACALSim IR.

This example demonstrates how the ACALSim backend captures matrix operations
and generates corresponding IR and RISC-V code.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from acalsim_backend import ACALSimBackend


class MatMulModel(nn.Module):
    """A model with various matrix multiplication patterns."""

    def __init__(self, M=8, K=16, N=8):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        # Store weight as parameter
        self.weight = nn.Parameter(torch.randn(K, N))
        self.bias = nn.Parameter(torch.randn(N))

    def forward(self, x):
        # x: [M, K] @ weight: [K, N] -> [M, N]
        y = torch.matmul(x, self.weight)
        y = y + self.bias
        return y


class ChainedMatMul(nn.Module):
    """Multiple chained matrix multiplications."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(16, 32))
        self.w2 = nn.Parameter(torch.randn(32, 16))
        self.w3 = nn.Parameter(torch.randn(16, 8))

    def forward(self, x):
        # Chain: [B, 16] -> [B, 32] -> [B, 16] -> [B, 8]
        x = torch.matmul(x, self.w1)
        x = torch.relu(x)
        x = torch.matmul(x, self.w2)
        x = torch.relu(x)
        x = torch.matmul(x, self.w3)
        return x


def demo_simple_matmul():
    """Demo simple matrix multiplication."""
    print("\n" + "=" * 60)
    print("Demo 1: Simple Matrix Multiplication")
    print("=" * 60)

    model = MatMulModel(M=8, K=16, N=8)
    model.eval()

    x = torch.randn(8, 16)

    output_dir = os.path.join(os.path.dirname(__file__), "output_matmul")
    os.makedirs(output_dir, exist_ok=True)

    backend = ACALSimBackend(mode="codegen", verbose=True, output_dir=output_dir)
    compiled = torch.compile(model, backend=backend)

    with torch.no_grad():
        output = compiled(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Files saved to: {output_dir}")


def demo_chained_matmul():
    """Demo chained matrix multiplications."""
    print("\n" + "=" * 60)
    print("Demo 2: Chained Matrix Multiplications")
    print("=" * 60)

    model = ChainedMatMul()
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 16)

    output_dir = os.path.join(os.path.dirname(__file__), "output_chained")
    os.makedirs(output_dir, exist_ok=True)

    backend = ACALSimBackend(mode="codegen", verbose=True, output_dir=output_dir)
    compiled = torch.compile(model, backend=backend)

    with torch.no_grad():
        output = compiled(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Files saved to: {output_dir}")


def demo_batched_matmul():
    """Demo batched matrix multiplication using torch function."""
    print("\n" + "=" * 60)
    print("Demo 3: Batched Matrix Multiplication (bmm)")
    print("=" * 60)

    # Define a simple function instead of a class
    def batched_matmul(a, b):
        return torch.bmm(a, b)

    # Create example inputs
    batch = 2
    M, K, N = 4, 8, 4
    a = torch.randn(batch, M, K)
    b = torch.randn(batch, K, N)

    output_dir = os.path.join(os.path.dirname(__file__), "output_bmm")
    os.makedirs(output_dir, exist_ok=True)

    backend = ACALSimBackend(mode="compile", verbose=True, output_dir=output_dir)
    compiled = torch.compile(batched_matmul, backend=backend)

    with torch.no_grad():
        output = compiled(a, b)

    print(f"\nInput A shape: {a.shape}")
    print(f"Input B shape: {b.shape}")
    print(f"Output shape: {output.shape}")


def main():
    print("=" * 60)
    print("ACALSim Backend Demo: Matrix Multiplication Operations")
    print("=" * 60)

    demo_simple_matmul()
    demo_chained_matmul()
    demo_batched_matmul()

    print("\n" + "=" * 60)
    print("All Demos Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
