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
"""QEMU Device Server - TCP version with debug logging"""
import socket, struct, json, numpy as np, os, sys
import traceback

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
			print(f"⚠ SST bridge not found at {self.bridge_socket}")
			print("  Will use CPU fallback")
			self.sst = None
			return
		try:
			self.sst = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
			self.sst.connect(self.bridge_socket)
			print(f"✓ Connected to SST bridge")
		except Exception as e:
			print(f"✗ Failed to connect to SST bridge: {e}")
			self.sst = None

	def launch_gemm_kernel(self, op, A, B):
		self.job_counter += 1
		if self.sst:
			print(f"  → Sending to SST: {op.M}x{op.K} @ {op.K}x{op.N}")
			try:
				return self._submit_to_sst(op, A, B)
			except Exception as e:
				print(f"  ✗ SST error: {e}")
				print("  → Falling back to CPU")
				return np.matmul(A, B)
		else:
			print(f"  → CPU fallback: {op.M}x{op.K} @ {op.K}x{op.N}")
			return np.matmul(A, B)

	def _submit_to_sst(self, op, A, B):
		# Send compute request to SST bridge
		payload = struct.pack("<Q", op.M * op.N * op.K)
		req = struct.pack("<IQI", 2, self.job_counter, len(payload)) + payload
		self.sst.sendall(req)

		# Receive response
		hdr = self._recv_exact(16)
		if hdr:
			status, req_id, psize = struct.unpack("<IQI", hdr)
			print(f"  ← SST response: status={status}, req_id={req_id}")
			if status == 0:
				# SST succeeded, compute result on CPU
				# (Real SST would return actual result)
				return np.matmul(A, B)
		print(f"  ⚠ No SST response, using CPU")
		return np.matmul(A, B)

	def _recv_exact(self, n):
		d = b""
		while len(d) < n:
			c = self.sst.recv(n - len(d))
			if not c:
				print(f"  ✗ SST connection closed")
				return None
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
		print(f"✓ Listening on TCP port {port}")
		print("  (Accessible from Docker at localhost:9999)")

	def run(self):
		print("\nWaiting for host connection...")
		print("=" * 60)
		while True:
			conn, addr = self.server.accept()
			print(f"\n✓ Host connected from {addr}")
			try:
				self.handle_connection(conn)
			except Exception as e:
				print(f"✗ Connection error: {e}")
				traceback.print_exc()
			finally:
				conn.close()
				print(f"✗ Host disconnected\n")
				print("=" * 60)
				print("Waiting for next connection...")

	def handle_connection(self, conn):
		while True:
			try:
				# Receive message header (9 bytes: magic(4) + length(4) + op_type(1))
				print(f"\n[Waiting for message header...]")
				hdr = self._recv_exact(conn, 9)
				if not hdr:
					print("✗ Connection closed by host")
					break

				print(f"  Received header: {len(hdr)} bytes")

				# Parse header to get payload length
				magic, length, op_type = struct.unpack(">IIB", hdr)
				print(f"  Magic: {hex(magic)}, Length: {length}, OpType: {op_type}")

				if magic != Message.MAGIC:
					print(f"  ✗ Invalid magic: expected {hex(Message.MAGIC)}, got {hex(magic)}")
					break

				# Receive JSON payload
				print(f"  Receiving payload: {length} bytes...")
				payload_bytes = self._recv_exact(conn, length)
				if not payload_bytes:
					print("  ✗ Failed to receive payload")
					break

				# Parse JSON payload
				payload = json.loads(payload_bytes.decode("utf-8"))
				print(f"  Payload: {payload}")

				if op_type == OpType.GEMM:
					self.handle_gemm(conn, payload)
				else:
					print(f"  ✗ Unknown operation type: {op_type}")

			except ConnectionError as e:
				print(f"✗ Connection error: {e}")
				break
			except Exception as e:
				print(f"✗ Protocol error: {e}")
				traceback.print_exc()
				break

	def handle_gemm(self, conn, payload):
		try:
			op = GEMMOperation.from_dict(payload)
			print("\n" + "=" * 60)
			print(f"GEMM Operation #{op.op_id}")
			print(f"  Matrix A: {op.A_shape}")
			print(f"  Matrix B: {op.B_shape}")
			print(f"  Output: ({op.M}, {op.N})")

			# Receive matrix A
			print(f"\n  Receiving matrix A...")
			A_size_bytes = self._recv_exact(conn, 4)
			if not A_size_bytes:
				print("  ✗ Failed to receive A size")
				return
			A_size = struct.unpack(">I", A_size_bytes)[0]
			print(f"    Size: {A_size} bytes")

			A_bytes = self._recv_exact(conn, A_size)
			if not A_bytes:
				print("  ✗ Failed to receive A data")
				return
			A = np.frombuffer(A_bytes, dtype=np.float32).reshape(op.A_shape)
			print(f"    ✓ Received A: {A.shape}")

			# Receive matrix B
			print(f"\n  Receiving matrix B...")
			B_size_bytes = self._recv_exact(conn, 4)
			if not B_size_bytes:
				print("  ✗ Failed to receive B size")
				return
			B_size = struct.unpack(">I", B_size_bytes)[0]
			print(f"    Size: {B_size} bytes")

			B_bytes = self._recv_exact(conn, B_size)
			if not B_bytes:
				print("  ✗ Failed to receive B data")
				return
			B = np.frombuffer(B_bytes, dtype=np.float32).reshape(op.B_shape)
			print(f"    ✓ Received B: {B.shape}")

			# Launch kernel
			print(f"\n  Launching GEMM kernel...")
			result = self.sst.launch_gemm_kernel(op, A, B)
			print(f"    ✓ Kernel complete: {result.shape}")

			# Send result
			print(f"\n  Sending result...")
			result_bytes = result.astype(np.float32).tobytes()
			response = struct.pack(">I", len(result_bytes)) + result_bytes
			print(f"    Sending {len(response)} bytes")
			conn.sendall(response)
			print(f"    ✓ Result sent")

			print(f"\n✓ GEMM #{op.op_id} completed successfully")
			print("=" * 60)

		except Exception as e:
			print(f"\n✗ GEMM handler error: {e}")
			traceback.print_exc()

	def _recv_exact(self, sock, n):
		"""Receive exactly n bytes"""
		data = b""
		while len(data) < n:
			chunk = sock.recv(n - len(data))
			if not chunk:
				print(f"    ✗ Socket closed (received {len(data)}/{n} bytes)")
				return None
			data += chunk
		return data


if __name__ == "__main__":
	print("=" * 60)
	print("QEMU Device Server - TCP (Debug Mode)")
	print("=" * 60)
	print()
	QEMUDeviceServer().run()
