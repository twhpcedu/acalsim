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
"""Tests for ACALSim backend module."""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from acalsim_backend import ACALSimBackend


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return torch.relu(self.fc(x))


class TestACALSimBackend:
    """Tests for ACALSimBackend."""

    def test_trace_mode(self):
        """Test trace mode captures graph and runs."""
        model = SimpleModel()
        model.eval()
        x = torch.randn(2, 4)

        backend = ACALSimBackend(mode="trace", verbose=False)
        compiled = torch.compile(model, backend=backend)

        with torch.no_grad():
            output = compiled(x)

        assert output.shape == (2, 2)

    def test_compile_mode(self):
        """Test compile mode generates IR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            model.eval()
            x = torch.randn(2, 4)

            backend = ACALSimBackend(
                mode="compile", verbose=False, output_dir=tmpdir
            )
            compiled = torch.compile(model, backend=backend)

            with torch.no_grad():
                output = compiled(x)

            assert output.shape == (2, 2)

            # Check IR file was created
            files = os.listdir(tmpdir)
            ir_files = [f for f in files if f.endswith(".ir")]
            assert len(ir_files) > 0

    def test_eager_mode(self):
        """Test eager mode runs without compilation."""
        model = SimpleModel()
        model.eval()
        x = torch.randn(2, 4)

        backend = ACALSimBackend(mode="eager", verbose=False)
        compiled = torch.compile(model, backend=backend)

        with torch.no_grad():
            output = compiled(x)

        assert output.shape == (2, 2)

    def test_get_compiled_ir(self):
        """Test retrieving compiled IR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            model.eval()
            x = torch.randn(2, 4)

            backend = ACALSimBackend(
                mode="compile", verbose=False, output_dir=tmpdir
            )
            compiled = torch.compile(model, backend=backend)

            with torch.no_grad():
                _ = compiled(x)

            # Get compiled IR
            ir = backend.get_compiled_ir("graph_0")
            assert ir is not None
            assert len(ir.ops) > 0

    def test_codegen_mode(self):
        """Test codegen mode generates RISC-V code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            model.eval()
            x = torch.randn(2, 4)

            backend = ACALSimBackend(
                mode="codegen", verbose=False, output_dir=tmpdir
            )
            compiled = torch.compile(model, backend=backend)

            with torch.no_grad():
                output = compiled(x)

            assert output.shape == (2, 2)

            # Check code files were created
            files = os.listdir(tmpdir)
            c_files = [f for f in files if f.endswith(".c")]
            h_files = [f for f in files if f.endswith(".h")]

            assert len(c_files) > 0
            assert len(h_files) > 0


class TestFunctionalCompilation:
    """Test compilation of functional operations."""

    def test_matmul(self):
        """Test matrix multiplication compilation."""

        def matmul_fn(a, b):
            return torch.matmul(a, b)

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ACALSimBackend(
                mode="compile", verbose=False, output_dir=tmpdir
            )
            compiled = torch.compile(matmul_fn, backend=backend)

            a = torch.randn(4, 8)
            b = torch.randn(8, 4)

            with torch.no_grad():
                output = compiled(a, b)

            assert output.shape == (4, 4)

    def test_elementwise_ops(self):
        """Test elementwise operations."""

        def ops_fn(x, y):
            return x + y * 2 - 1

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ACALSimBackend(
                mode="compile", verbose=False, output_dir=tmpdir
            )
            compiled = torch.compile(ops_fn, backend=backend)

            x = torch.randn(4, 4)
            y = torch.randn(4, 4)

            with torch.no_grad():
                output = compiled(x, y)

            assert output.shape == (4, 4)

    def test_activations(self):
        """Test activation functions."""

        def act_fn(x):
            x = torch.relu(x)
            x = torch.sigmoid(x)
            return x

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ACALSimBackend(
                mode="compile", verbose=False, output_dir=tmpdir
            )
            compiled = torch.compile(act_fn, backend=backend)

            x = torch.randn(4, 4)

            with torch.no_grad():
                output = compiled(x)

            assert output.shape == (4, 4)
