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
"""
Communication protocol for Host <-> QEMU <-> SST
"""
import struct
import json


class OpType:
	GEMM = 0x01
	CONV = 0x02
	MATMUL = 0x03


class Message:
	"""
    Message format:
    [4 bytes] Magic number (0xDEADBEEF)
    [4 bytes] Message length
    [1 byte]  Operation type
    [N bytes] JSON payload
    """
	MAGIC = 0xDEADBEEF

	@staticmethod
	def pack(op_type, payload):
		"""Pack a message for transmission"""
		json_data = json.dumps(payload).encode('utf-8')
		header = struct.pack('>IIB', Message.MAGIC, len(json_data), op_type)
		return header + json_data

	@staticmethod
	def unpack(data):
		"""Unpack a received message"""
		if len(data) < 9:
			raise ValueError("Message too short")

		magic, length, op_type = struct.unpack('>IIB', data[: 9])
		if magic != Message.MAGIC:
			raise ValueError(f"Invalid magic number: {hex(magic)}")

		payload = json.loads(data[9 : 9 + length].decode('utf-8'))
		return op_type, payload


class GEMMOperation:
	"""GEMM operation payload"""

	def __init__(self, M, N, K, A_shape, B_shape, dtype='float32'):
		self.M = M
		self.N = N
		self.K = K
		self.A_shape = A_shape
		self.B_shape = B_shape
		self.dtype = dtype
		self.op_id = None  # Set by dispatcher

	def to_dict(self):
		return {
		    'op_id': self.op_id,
		    'M': self.M,
		    'N': self.N,
		    'K': self.K,
		    'A_shape': self.A_shape,
		    'B_shape': self.B_shape,
		    'dtype': self.dtype
		}

	@staticmethod
	def from_dict(data):
		op = GEMMOperation(
		    data['M'], data['N'], data['K'], data['A_shape'], data['B_shape'], data['dtype']
		)
		op.op_id = data['op_id']
		return op
