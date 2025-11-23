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
"""
Test script for device GEMM operator
Run this on host (Docker Ubuntu)
"""
import torch
import sys

sys.path.insert(0, '/home/user/projects/device_gemm')

from device_gemm_operator import DeviceLinear, DeviceGEMM

print("=" * 60)
print("Device GEMM Operator Test")
print("=" * 60)
print()

# Test 1: Simple matrix multiplication
print("Test 1: Simple GEMM operation")
print("-" * 60)

A = torch.randn(4, 8)
B = torch.randn(8, 16)

print(f"Matrix A: {A.shape}")
print(f"Matrix B: {B.shape}")

try:
	result = DeviceGEMM.apply(A, B)
	print(f"Result: {result.shape}")
	print(f"✓ Device GEMM executed successfully!")

	# Verify correctness
	expected = torch.matmul(A, B)
	diff = torch.abs(result - expected).max().item()
	print(f"Max difference from CPU: {diff:.6f}")

	if diff < 1e-5:
		print("✓ Result matches CPU computation!")
	else:
		print("⚠ Result differs from CPU")

except Exception as e:
	print(f"✗ Device GEMM failed: {e}")
	import traceback
	traceback.print_exc()

print()

# Test 2: DeviceLinear layer
print("Test 2: DeviceLinear layer")
print("-" * 60)

layer = DeviceLinear(128, 256)
x = torch.randn(32, 128)  # batch_size=32, in_features=128

print(f"Input: {x.shape}")
print(f"Layer: Linear({layer.in_features} -> {layer.out_features})")

try:
	output = layer(x)
	print(f"Output: {output.shape}")
	print(f"✓ DeviceLinear layer executed successfully!")
except Exception as e:
	print(f"✗ DeviceLinear failed: {e}")
	import traceback
	traceback.print_exc()

print()
print("=" * 60)
print("Test completed")
print("=" * 60)
