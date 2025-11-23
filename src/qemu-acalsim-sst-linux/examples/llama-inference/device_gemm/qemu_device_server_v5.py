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

#!/usr/bin/env python3
"""QEMU Device Server v5 - ALL sockets in shared folder"""
import socket, struct, json, numpy as np, os, sys, time

sys.path.insert(0, "/mnt/shared")
from operator_protocol import Message, OpType, GEMMOperation


class SSTKernelLauncher:

	def __init__(self):
		self.bridge_socket = "/mnt/shared/device_gemm/sst-bridge.sock"
		self.sst = None
		self.job_counter = 0
		self.connect_to_bridge()

	def connect_to_bridge(self):
		if not os.path.exists(self.bridge_socket):
			print(f"SST bridge not found at {self.bridge_socket}")
			self.sst = None
			return
		try:
			self.sst = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
			self.sst.connect(self.bridge_socket)
			print(f"Connected to SST bridge")
		except Exception as e:
			print(f"Failed to connect: {e}")
			self.sst = None

	def launch_gemm_kernel(self, op, A_data, B_data):
		self.job_counter += 1
		if self.sst:
			print(f"Launching GEMM on SST: {op.M}x{op.K} @ {op.K}x{op.N}")
			result = self._submit_to_sst(op, A_data, B_data)
		else:
			print("SST bridge not connected, using CPU fallback")
			result = np.matmul(A_data, B_data)
		return result

	def _submit_to_sst(self, op, A, B):
		compute_units = op.M * op.N * op.K
		payload = struct.pack("<Q", compute_units)
		req = struct.pack("<IQI", 2, self.job_counter, len(payload)) + payload
		try:
			self.sst.sendall(req)
			resp_hdr = self._recv_exact(16)
			if not resp_hdr:
				print("No response from SST, using CPU fallback")
				return np.matmul(A, B)
			status, resp_id, resp_size = struct.unpack("<IQI", resp_hdr)
			if status == 0:
				print(f"SST completed job {resp_id}")
				return np.matmul(A, B)
			else:
				print(f"SST error: status={status}")
				return np.matmul(A, B)
		except Exception as e:
			print(f"SST error: {e}")
			return np.matmul(A, B)

	def _recv_exact(self, n):
		data = b""
		while len(data) < n:
			chunk = self.sst.recv(n - len(data))
			if not chunk: return None
			data += chunk
		return data


class QEMUDeviceServer:

	def __init__(self):
		# Put socket in shared folder so Docker can access it!
		self.socket_path = "/mnt/shared/device_gemm/qemu-device-ops.sock"
		self.sst_launcher = SSTKernelLauncher()
		if os.path.exists(self.socket_path):
			os.remove(self.socket_path)
		self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		self.server.bind(self.socket_path)
		self.server.listen(1)
		print(f"Listening on {self.socket_path}")
		print("(Accessible from Docker at /home/user/projects/device_gemm/qemu-device-ops.sock)")

	def run(self):
		print("Waiting for host connection...")
		while True:
			conn, addr = self.server.accept()
			print("Host connected")
			try:
				self.handle_connection(conn)
			except Exception as e:
				print(f"Error: {e}")
			finally:
				conn.close()

	def handle_connection(self, conn):
		while True:
			try:
				header = self._recv_exact(conn, 9)
				if not header: break
				op_type, payload = Message.unpack(header)
				if op_type == OpType.GEMM:
					self.handle_gemm(conn, payload)
			except:
				break

	def handle_gemm(self, conn, payload):
		op = GEMMOperation.from_dict(payload)
		print(f"\\nGEMM #{op.op_id}: {op.A_shape} @ {op.B_shape}")
		A_size = struct.unpack(">I", self._recv_exact(conn, 4))[0]
		A_data = np.frombuffer(self._recv_exact(conn, A_size), dtype=np.float32).reshape(op.A_shape)
		B_size = struct.unpack(">I", self._recv_exact(conn, 4))[0]
		B_data = np.frombuffer(self._recv_exact(conn, B_size), dtype=np.float32).reshape(op.B_shape)
		result = self.sst_launcher.launch_gemm_kernel(op, A_data, B_data)
		conn.sendall(struct.pack(">I", len(result.tobytes())) + result.astype(np.float32).tobytes())
		print(f"GEMM #{op.op_id} completed\\n")

	def _recv_exact(self, sock, n):
		data = b""
		while len(data) < n:
			chunk = sock.recv(n - len(data))
			if not chunk: return None
			data += chunk
		return data


if __name__ == "__main__":
	print("=" * 60)
	print("QEMU Device Server v5")
	print("=" * 60)
	server = QEMUDeviceServer()
	server.run()
