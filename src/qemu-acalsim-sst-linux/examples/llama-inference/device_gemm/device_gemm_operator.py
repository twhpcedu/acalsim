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
Custom PyTorch GEMM operator that offloads to QEMU device
"""
import torch
import socket
import numpy as np
from operator_protocol import Message, OpType, GEMMOperation


class DeviceGEMM(torch.autograd.Function):
	"""Custom GEMM operation that executes on device (QEMU+SST)"""

	device_socket = None  # Shared socket connection
	op_counter = 0

	@staticmethod
	def forward(ctx, A, B):
		"""
        Forward pass: Send GEMM operation to device
        A: (M, K) matrix
        B: (K, N) matrix
        Returns: (M, N) result
        """
		assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"

		M, K = A.shape
		K2, N = B.shape

		# Create operation descriptor
		op = GEMMOperation(M, N, K, list(A.shape), list(B.shape))
		DeviceGEMM.op_counter += 1
		op.op_id = DeviceGEMM.op_counter

		# Send to device
		result = DeviceGEMM._execute_on_device(op, A, B)

		# Save for backward pass
		ctx.save_for_backward(A, B)

		return result

	@staticmethod
	def _execute_on_device(op, A, B):
		"""Execute GEMM on device via socket"""
		try:
			sock = DeviceGEMM._get_socket()

			# Pack and send operation metadata
			msg = Message.pack(OpType.GEMM, op.to_dict())
			sock.sendall(msg)

			# Send matrix data (simplified - send shapes and data)
			A_bytes = A.detach().cpu().numpy().tobytes()
			B_bytes = B.detach().cpu().numpy().tobytes()

			# Send A matrix
			sock.sendall(struct.pack('>I', len(A_bytes)))
			sock.sendall(A_bytes)

			# Send B matrix
			sock.sendall(struct.pack('>I', len(B_bytes)))
			sock.sendall(B_bytes)

			# Receive result
			result_size = struct.unpack('>I', DeviceGEMM._recv_exact(sock, 4))[0]
			result_bytes = DeviceGEMM._recv_exact(sock, result_size)

			# Reconstruct result tensor
			result_np = np.frombuffer(result_bytes, dtype=np.float32).reshape(op.M, op.N)
			result = torch.from_numpy(result_np)

			return result

		except Exception as e:
			print(f"Device execution failed: {e}")
			print("Falling back to CPU execution")
			return torch.matmul(A, B)

	@staticmethod
	def _get_socket():
		"""Get or create socket connection to device"""
		if DeviceGEMM.device_socket is None:
			DeviceGEMM.device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			DeviceGEMM.device_socket.connect(('localhost', 9999))
		return DeviceGEMM.device_socket

	@staticmethod
	def _recv_exact(sock, n):
		"""Receive exactly n bytes from socket"""
		data = b''
		while len(data) < n:
			chunk = sock.recv(n - len(data))
			if not chunk:
				raise ConnectionError("Socket connection broken")
			data += chunk
		return data

	@staticmethod
	def backward(ctx, grad_output):
		"""Backward pass (not implemented for now)"""
		A, B = ctx.saved_tensors
		grad_A = torch.matmul(grad_output, B.t())
		grad_B = torch.matmul(A.t(), grad_output)
		return grad_A, grad_B


class DeviceLinear(torch.nn.Module):
	"""
    Linear layer that uses device GEMM
    Drop-in replacement for torch.nn.Linear
    """

	def __init__(self, in_features, out_features, bias=True):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
		if bias:
			self.bias = torch.nn.Parameter(torch.randn(out_features))
		else:
			self.register_parameter('bias', None)

	def forward(self, x):
		# x: (batch, in_features)
		# weight: (out_features, in_features)
		# result: (batch, out_features)
		result = DeviceGEMM.apply(x, self.weight.t())
		if self.bias is not None:
			result = result + self.bias
		return result


# Add missing import
import struct
