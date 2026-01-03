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
"""Example: Compile simplified attention mechanism to ACALSim IR.

This example shows how the ACALSim backend handles attention-like operations
that are common in transformer architectures.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from acalsim_backend import ACALSimBackend


class SimpleAttention(nn.Module):
    """Simplified single-head attention for demonstration."""

    def __init__(self, embed_dim=32, head_dim=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)

        self.q_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, head_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Compute attention scores: Q @ K^T / sqrt(d)
        # [batch, seq_len, head_dim] @ [batch, head_dim, seq_len] -> [batch, seq_len, seq_len]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Softmax
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        # [batch, seq_len, seq_len] @ [batch, seq_len, head_dim] -> [batch, seq_len, head_dim]
        out = torch.matmul(attn, v)

        # Project back
        out = self.out_proj(out)

        return out


class SimpleFeedForward(nn.Module):
    """Simple feed-forward network (FFN) as in transformers."""

    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class MiniTransformerBlock(nn.Module):
    """A minimal transformer block combining attention and FFN."""

    def __init__(self, embed_dim=32, head_dim=8, ffn_dim=64):
        super().__init__()
        self.attention = SimpleAttention(embed_dim, head_dim)
        self.ffn = SimpleFeedForward(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def demo_attention():
    """Demo single-head attention."""
    print("\n" + "=" * 60)
    print("Demo 1: Simple Single-Head Attention")
    print("=" * 60)

    model = SimpleAttention(embed_dim=32, head_dim=8)
    model.eval()

    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, 32)

    output_dir = os.path.join(os.path.dirname(__file__), "output_attention")
    os.makedirs(output_dir, exist_ok=True)

    backend = ACALSimBackend(mode="codegen", verbose=True, output_dir=output_dir)
    compiled = torch.compile(model, backend=backend)

    with torch.no_grad():
        output = compiled(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Files saved to: {output_dir}")


def demo_ffn():
    """Demo feed-forward network."""
    print("\n" + "=" * 60)
    print("Demo 2: Feed-Forward Network")
    print("=" * 60)

    model = SimpleFeedForward(embed_dim=32, hidden_dim=64)
    model.eval()

    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, 32)

    output_dir = os.path.join(os.path.dirname(__file__), "output_ffn")
    os.makedirs(output_dir, exist_ok=True)

    backend = ACALSimBackend(mode="codegen", verbose=True, output_dir=output_dir)
    compiled = torch.compile(model, backend=backend)

    with torch.no_grad():
        output = compiled(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def demo_transformer_block():
    """Demo full transformer block."""
    print("\n" + "=" * 60)
    print("Demo 3: Mini Transformer Block")
    print("=" * 60)

    model = MiniTransformerBlock(embed_dim=32, head_dim=8, ffn_dim=64)
    model.eval()

    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, 32)

    output_dir = os.path.join(os.path.dirname(__file__), "output_transformer")
    os.makedirs(output_dir, exist_ok=True)

    backend = ACALSimBackend(mode="compile", verbose=True, output_dir=output_dir)
    compiled = torch.compile(model, backend=backend)

    with torch.no_grad():
        output = compiled(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Files saved to: {output_dir}")

    # Show IR statistics
    ir = backend.get_compiled_ir("graph_0")
    if ir:
        print(f"\nIR Statistics:")
        print(f"  Operations: {len(ir.ops)}")
        print(f"  Tensors: {len(ir.tensors)}")
        print(f"  Estimated cycles: {ir.estimate_cycles()}")
        print(f"  Estimated memory: {ir.estimate_memory()} bytes")


def main():
    print("=" * 60)
    print("ACALSim Backend Demo: Transformer Components")
    print("=" * 60)

    demo_attention()
    demo_ffn()
    demo_transformer_block()

    print("\n" + "=" * 60)
    print("All Demos Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
