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
"""QEMU Device Server - VirtIO-SST version"""
import socket, struct, json, numpy as np, os, sys
import traceback

sys.path.insert(0, "/mnt/shared")
from operator_protocol import Message, OpType, GEMMOperation

# Import VirtIO-SST wrapper
try:
	from virtio_sst_wrapper_fixed import VirtIOSST, SSTRequestType, SSTStatus
	VIRTIO_SST_AVAILABLE = True
except ImportError:
	print("⚠ VirtIO-SST wrapper not found, using CPU fallback only")
	VIRTIO_SST_AVAILABLE = False


class SSTKernelLauncher:
	"""SST kernel launcher using VirtIO-SST device"""

	def __init__(self):
		self.sst = None
		self.job_counter = 0
		self.use_virtio = VIRTIO_SST_AVAILABLE
		self.connect_to_sst()

	def connect_to_sst(self):
		"""Connect to SST via /dev/sst0"""
		if not self.use_virtio:
			print("  ⚠ VirtIO-SST not available, using CPU fallback")
			return

		try:
			# Check if /dev/sst0 exists
			if not os.path.exists("/dev/sst0"):
				print("  ⚠ /dev/sst0 not found")
				print("    Load module with: sudo modprobe virtio-sst")
				self.use_virtio = False
				return

			# Open VirtIO-SST device
			self.sst = VirtIOSST()
			self.sst.open()

			# Test with ping
			if self.sst.ping():
				print("  ✓ Connected to SST via VirtIO-SST (/dev/sst0)")
			else:
				print("  ✗ SST ping failed")
				self.sst.close()
				self.sst = None
				self.use_virtio = False

		except Exception as e:
			print(f"  ✗ Failed to connect to VirtIO-SST: {e}")
			self.sst = None
			self.use_virtio = False

	def launch_gemm_kernel(self, op, A, B):
		"""Launch GEMM kernel on SST or CPU"""
		self.job_counter += 1

		if self.use_virtio and self.sst:
			print(f"  → Sending to SST via VirtIO: {op.M}x{op.K} @ {op.K}x{op.N}")
			try:
				return self._submit_to_sst(op, A, B)
			except Exception as e:
				print(f"  ✗ SST error: {e}")
				traceback.print_exc()
				print("  → Falling back to CPU")
				return np.matmul(A, B)
		else:
			print(f"  → CPU fallback: {op.M}x{op.K} @ {op.K}x{op.N}")
			return np.matmul(A, B)

	def _submit_to_sst(self, op, A, B):
		"""Submit GEMM to SST and get result"""
		# Calculate compute units (FLOPs)
		compute_units = op.M * op.N * (2 * op.K)  # Each element: K multiplies + K adds

		print(f"    Compute units: {compute_units}")

		# Send compute request to SST
		result = self.sst.compute(compute_units=compute_units, latency_model=0)

		if result["status"] == "ok":
			cycles = result.get("cycles", 0)
			timestamp = result.get("timestamp", 0)
			print(f"  ← SST response: {cycles} cycles, timestamp={timestamp}")

			# For now, compute result on CPU (SST returns timing, not data)
			# In a real implementation, SST would return the actual matrix result
			C = np.matmul(A, B)

			# Store SST metadata (could be returned to PyTorch)
			# self.last_sst_cycles = cycles
			# self.last_sst_timestamp = timestamp

			return C
		else:
			print(f"  ✗ SST compute failed: {result}")
			return np.matmul(A, B)

	def __del__(self):
		"""Cleanup"""
		if self.sst:
			self.sst.close()


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

			# Execute GEMM on SST/CPU
			print(f"\n  Computing {op.M}x{op.K} @ {op.K}x{op.N} = {op.M}x{op.N}...")
			C = self.sst.launch_gemm_kernel(op, A, B)

			# Send result back
			print(f"\n  Sending result...")
			result_bytes = C.astype(np.float32).tobytes()
			print(f"    Result size: {len(result_bytes)} bytes")
			conn.sendall(struct.pack(">I", len(result_bytes)))
			conn.sendall(result_bytes)
			print(f"    ✓ Result sent")
			print("=" * 60)

		except Exception as e:
			print(f"✗ GEMM handler error: {e}")
			traceback.print_exc()

	def _recv_exact(self, conn, n):
		"""Receive exactly n bytes"""
		data = b""
		while len(data) < n:
			chunk = conn.recv(n - len(data))
			if not chunk:
				return None
			data += chunk
		return data


if __name__ == "__main__":
	print("=" * 60)
	print("QEMU Device Server - VirtIO-SST Version")
	print("=" * 60)
	print()

	server = QEMUDeviceServer(port=9999)

	try:
		server.run()
	except KeyboardInterrupt:
		print("\n\n✓ Server stopped by user")
	except Exception as e:
		print(f"\n✗ Server error: {e}")
		traceback.print_exc()
