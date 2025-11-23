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
QEMU Device-side operator receiver and SST dispatcher
Runs inside QEMU, receives operators from host, dispatches to SST
"""
import socket
import struct
import json
import numpy as np
import os
import sys

# Import protocol (copy from host)
sys.path.insert(0, '/mnt/shared')
from operator_protocol import Message, OpType, GEMMOperation


class SSTKernelLauncher:
	"""Interface to SST accelerator"""

	def __init__(self, sst_socket_path='/tmp/qemu-sst-llama.sock'):
		self.sst_socket = None
		self.sst_socket_path = sst_socket_path
		self.connect_to_sst()

	def connect_to_sst(self):
		"""Connect to SST simulator via VirtIO socket"""
		try:
			# SST connection happens via VirtIO device
			# For now, simulate with file-based communication
			print(f"Connecting to SST at {self.sst_socket_path}...")
			# In real implementation, this would use VirtIO-SST device
			self.sst_connected = os.path.exists(self.sst_socket_path)
			if self.sst_connected:
				print("✓ SST connection established")
			else:
				print("⚠ SST not available, will use CPU fallback")
		except Exception as e:
			print(f"SST connection failed: {e}")
			self.sst_connected = False

	def launch_gemm_kernel(self, op, A_data, B_data):
		"""
        Launch GEMM kernel on SST
        Returns result matrix as numpy array
        """
		if self.sst_connected:
			print(f"Launching GEMM kernel on SST: {op.M}x{op.K} @ {op.K}x{op.N}")
			# TODO: Actual SST kernel launch via VirtIO
			# For now, simulate with CPU computation
			result = self._sst_gemm_simulate(A_data, B_data)
		else:
			print("SST not available, executing on CPU")
			result = np.matmul(A_data, B_data)

		return result

	def _sst_gemm_simulate(self, A, B):
		"""Simulate SST GEMM execution (replace with real VirtIO call)"""
		# This is where you would:
		# 1. Format data for SST
		# 2. Send via VirtIO-SST device
		# 3. Wait for completion
		# 4. Receive result

		# For now, use CPU
		return np.matmul(A, B)


class QEMUDeviceServer:
	"""Device-side server that receives operators from host"""

	def __init__(self, socket_path='/tmp/qemu-device-ops.sock'):
		self.socket_path = socket_path
		self.sst_launcher = SSTKernelLauncher()

		# Remove existing socket
		if os.path.exists(socket_path):
			os.remove(socket_path)

		# Create server socket
		self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		self.server.bind(socket_path)
		self.server.listen(1)
		print(f"✓ QEMU device server listening on {socket_path}")

	def run(self):
		"""Main server loop"""
		print("Waiting for host connection...")

		while True:
			conn, addr = self.server.accept()
			print("✓ Host connected")

			try:
				self.handle_connection(conn)
			except Exception as e:
				print(f"Error handling connection: {e}")
				import traceback
				traceback.print_exc()
			finally:
				conn.close()
				print("Host disconnected")

	def handle_connection(self, conn):
		"""Handle incoming operator requests"""
		while True:
			try:
				# Receive message header
				header = self._recv_exact(conn, 9)
				if not header:
					break

				op_type, payload = Message.unpack(header)

				if op_type == OpType.GEMM:
					self.handle_gemm(conn, payload)
				else:
					print(f"Unknown operation type: {op_type}")

			except ConnectionError:
				break
			except Exception as e:
				print(f"Error processing request: {e}")
				import traceback
				traceback.print_exc()
				break

	def handle_gemm(self, conn, payload):
		"""Handle GEMM operation"""
		op = GEMMOperation.from_dict(payload)
		print(f"\n{'='*60}")
		print(f"Received GEMM operation #{op.op_id}")
		print(f"  Matrix A: {op.A_shape}")
		print(f"  Matrix B: {op.B_shape}")
		print(f"  Output: ({op.M}, {op.N})")

		# Receive matrix A
		A_size = struct.unpack('>I', self._recv_exact(conn, 4))[0]
		A_bytes = self._recv_exact(conn, A_size)
		A_data = np.frombuffer(A_bytes, dtype=np.float32).reshape(op.A_shape)

		# Receive matrix B
		B_size = struct.unpack('>I', self._recv_exact(conn, 4))[0]
		B_bytes = self._recv_exact(conn, B_size)
		B_data = np.frombuffer(B_bytes, dtype=np.float32).reshape(op.B_shape)

		print(f"  Matrices received, launching kernel...")

		# Launch on SST
		result = self.sst_launcher.launch_gemm_kernel(op, A_data, B_data)

		print(f"  Kernel completed, sending result back to host")

		# Send result back
		result_bytes = result.astype(np.float32).tobytes()
		conn.sendall(struct.pack('>I', len(result_bytes)))
		conn.sendall(result_bytes)

		print(f"✓ GEMM operation #{op.op_id} completed")
		print(f"{'='*60}\n")

	def _recv_exact(self, sock, n):
		"""Receive exactly n bytes"""
		data = b''
		while len(data) < n:
			chunk = sock.recv(n - len(data))
			if not chunk:
				return None
			data += chunk
		return data


if __name__ == '__main__':
	print("=" * 60)
	print("QEMU Device Server - Operator Receiver & SST Dispatcher")
	print("=" * 60)
	print()

	server = QEMUDeviceServer()
	server.run()
