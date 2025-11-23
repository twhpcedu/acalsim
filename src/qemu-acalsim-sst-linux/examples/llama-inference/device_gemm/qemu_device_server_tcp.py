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
"""QEMU Device Server - TCP version (works across Docker/QEMU boundary)"""
import socket, struct, json, numpy as np, os, sys

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
			print(f"SST bridge not found")
			self.sst = None
			return
		try:
			self.sst = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
			self.sst.connect(self.bridge_socket)
			print("Connected to SST bridge")
		except Exception as e:
			print(f"Failed to connect: {e}")
			self.sst = None

	def launch_gemm_kernel(self, op, A, B):
		self.job_counter += 1
		if self.sst:
			print(f"SST GEMM: {op.M}x{op.K} @ {op.K}x{op.N}")
			return self._submit_to_sst(op, A, B)
		print("CPU fallback")
		return np.matmul(A, B)

	def _submit_to_sst(self, op, A, B):
		payload = struct.pack("<Q", op.M * op.N * op.K)
		req = struct.pack("<IQI", 2, self.job_counter, len(payload)) + payload
		try:
			self.sst.sendall(req)
			hdr = self._recv_exact(16)
			if hdr:
				status, _, _ = struct.unpack("<IQI", hdr)
				if status == 0:
					print(f"SST OK")
					return np.matmul(A, B)
		except:
			pass
		return np.matmul(A, B)

	def _recv_exact(self, n):
		d = b""
		while len(d) < n:
			c = self.sst.recv(n - len(d))
			if not c: return None
			d += c
		return d


class QEMUDeviceServer:

	def __init__(self, port=9999):
		self.port = port
		self.sst = SSTKernelLauncher()
		self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.server.bind(("0.0.0.0", port))
		self.server.listen(1)
		print(f"Listening on TCP port {port}")
		print("(Accessible from Docker at localhost:9999)")

	def run(self):
		print("Waiting for host...")
		while True:
			conn, addr = self.server.accept()
			print(f"Host connected from {addr}")
			try:
				self.handle_connection(conn)
			except:
				pass
			finally:
				conn.close()

	def handle_connection(self, conn):
		while True:
			try:
				hdr = self._recv_exact(conn, 9)
				if not hdr: break
				op_type, payload = Message.unpack(hdr)
				if op_type == OpType.GEMM:
					self.handle_gemm(conn, payload)
			except:
				break

	def handle_gemm(self, conn, payload):
		op = GEMMOperation.from_dict(payload)
		print(f"GEMM #{op.op_id}: {op.A_shape} @ {op.B_shape}")
		A_size = struct.unpack(">I", self._recv_exact(conn, 4))[0]
		A = np.frombuffer(self._recv_exact(conn, A_size), dtype=np.float32).reshape(op.A_shape)
		B_size = struct.unpack(">I", self._recv_exact(conn, 4))[0]
		B = np.frombuffer(self._recv_exact(conn, B_size), dtype=np.float32).reshape(op.B_shape)
		result = self.sst.launch_gemm_kernel(op, A, B)
		conn.sendall(struct.pack(">I", len(result.tobytes())) + result.astype(np.float32).tobytes())
		print(f"Done #{op.op_id}")

	def _recv_exact(self, s, n):
		d = b""
		while len(d) < n:
			c = s.recv(n - len(d))
			if not c: return None
			d += c
		return d


if __name__ == "__main__":
	print("=" * 60)
	print("QEMU Device Server - TCP")
	print("=" * 60)
	QEMUDeviceServer().run()
