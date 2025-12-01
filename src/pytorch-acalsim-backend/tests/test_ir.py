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
"""Tests for ACALSim IR module."""

import pytest
from acalsim_backend.ir import (
    ACALSimIR,
    ACALSimOp,
    ACALSimOpType,
    TensorDesc,
    get_acalsim_op_type,
)


class TestTensorDesc:
    """Tests for TensorDesc."""

    def test_numel(self):
        """Test number of elements calculation."""
        desc = TensorDesc(name="t0", shape=(2, 3, 4))
        assert desc.numel == 24

    def test_numel_scalar(self):
        """Test scalar tensor."""
        desc = TensorDesc(name="t0", shape=(1,))
        assert desc.numel == 1

    def test_size_bytes_float32(self):
        """Test byte size for float32."""
        desc = TensorDesc(name="t0", shape=(10,), dtype="float32")
        assert desc.size_bytes == 40

    def test_size_bytes_float16(self):
        """Test byte size for float16."""
        desc = TensorDesc(name="t0", shape=(10,), dtype="float16")
        assert desc.size_bytes == 20


class TestACALSimOp:
    """Tests for ACALSimOp."""

    def test_repr(self):
        """Test operation string representation."""
        op = ACALSimOp(
            op_type=ACALSimOpType.ADD,
            name="add_0",
            inputs=["a", "b"],
            outputs=["c"],
        )
        assert "ADD" in repr(op)
        assert "a" in repr(op)
        assert "c" in repr(op)


class TestACALSimIR:
    """Tests for ACALSimIR."""

    def test_add_tensor(self):
        """Test adding tensors to IR."""
        ir = ACALSimIR(name="test")
        desc = TensorDesc(name="t0", shape=(2, 3))
        ir.add_tensor(desc)

        assert "t0" in ir.tensors
        assert ir.tensors["t0"].shape == (2, 3)

    def test_add_op(self):
        """Test adding operations to IR."""
        ir = ACALSimIR(name="test")
        op = ACALSimOp(
            op_type=ACALSimOpType.RELU,
            name="relu_0",
            inputs=["t0"],
            outputs=["t1"],
        )
        ir.add_op(op)

        assert len(ir.ops) == 1
        assert ir.ops[0].op_type == ACALSimOpType.RELU

    def test_estimate_memory(self):
        """Test memory estimation."""
        ir = ACALSimIR(name="test")
        ir.add_tensor(TensorDesc(name="t0", shape=(10,), dtype="float32"))
        ir.add_tensor(TensorDesc(name="t1", shape=(10,), dtype="float32"))

        assert ir.estimate_memory() == 80  # 2 * 10 * 4 bytes

    def test_estimate_cycles(self):
        """Test cycle estimation."""
        ir = ACALSimIR(name="test")
        ir.add_op(
            ACALSimOp(
                op_type=ACALSimOpType.ADD,
                name="add_0",
                inputs=["a"],
                outputs=["b"],
                compute_cycles=100,
            )
        )
        ir.add_op(
            ACALSimOp(
                op_type=ACALSimOpType.RELU,
                name="relu_0",
                inputs=["b"],
                outputs=["c"],
                compute_cycles=50,
            )
        )

        assert ir.estimate_cycles() == 150

    def test_print_ir(self):
        """Test IR printing."""
        ir = ACALSimIR(name="test_graph")
        ir.add_tensor(TensorDesc(name="input", shape=(2, 3), is_input=True))
        ir.add_tensor(TensorDesc(name="output", shape=(2, 3), is_output=True))
        ir.input_names = ["input"]
        ir.output_names = ["output"]

        ir_str = ir.print_ir()
        assert "test_graph" in ir_str
        assert "input" in ir_str
        assert "output" in ir_str


class TestGetAcalsimOpType:
    """Tests for operation type mapping."""

    def test_basic_ops(self):
        """Test basic operation mapping."""
        assert get_acalsim_op_type("add") == ACALSimOpType.ADD
        assert get_acalsim_op_type("matmul") == ACALSimOpType.MATMUL
        assert get_acalsim_op_type("relu") == ACALSimOpType.RELU

    def test_aten_prefix(self):
        """Test operations with aten prefix."""
        assert get_acalsim_op_type("aten.add") == ACALSimOpType.ADD
        assert get_acalsim_op_type("aten::matmul") == ACALSimOpType.MATMUL

    def test_suffix_removal(self):
        """Test operations with suffix."""
        assert get_acalsim_op_type("add.Tensor") == ACALSimOpType.ADD
        assert get_acalsim_op_type("relu.default") == ACALSimOpType.RELU

    def test_unknown_op(self):
        """Test unknown operations map to CUSTOM."""
        assert get_acalsim_op_type("unknown_op") == ACALSimOpType.CUSTOM
