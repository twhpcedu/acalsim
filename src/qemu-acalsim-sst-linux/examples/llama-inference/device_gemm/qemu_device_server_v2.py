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
QEMU Device-side operator receiver and SST dispatcher (v2)
Uses shared file system to communicate with SST
"""
import socket
import struct
import json
import numpy as np
import os
import sys
import time

# Import protocol
sys.path.insert(0, '/mnt/shared')
from operator_protocol import Message, OpType, GEMMOperation


class SSTKernelLauncher:
	"""Interface to SST accelerator via shared filesystem"""

	def __init__(self):
		self.sst_job_dir = '/mnt/shared/sst_jobs'
		self.sst_result_dir = '/mnt/shared/sst_results'
		self.job_counter = 0
		self.setup_directories()

	def setup_directories(self):
		"""Create directories for SST communication"""
		os.makedirs(self.sst_job_dir, exist_ok=True)
		os.makedirs(self.sst_result_dir, exist_ok=True)

		# Check if SST is running by looking for marker file
		sst_marker = '/mnt/shared/.sst_running'
		self.sst_connected = os.path.exists(sst_marker)

		if self.sst_connected:
			print(f"✓ SST connection established (via {sst_marker})")
		else:
			print(f"⚠ SST not detected, will use CPU fallback")
			print(f"   (SST should create {sst_marker} when running)")

	def launch_gemm_kernel(self, op, A_data, B_data):
		"""
        Launch GEMM kernel on SST via shared filesystem
        """
		self.job_counter += 1
		job_id = f"gemm_{self.job_counter}_{int(time.time()*1000)}"

		if self.sst_connected:
			print(f"Launching GEMM kernel on SST: {op.M}x{op.K} @ {op.K}x{op.N}")
			result = self._submit_to_sst(job_id, op, A_data, B_data)
		else:
			print("SST not available, executing on CPU")
			result = np.matmul(A_data, B_data)

		return result

	def _submit_to_sst(self, job_id, op, A, B):
		"""
        Submit job to SST via shared filesystem
        
        Job format:
        sst_jobs/
          ├── {job_id}.json      # Operation metadata
          ├── {job_id}_A.npy     # Matrix A
          └── {job_id}_B.npy     # Matrix B
        
        Result format:
        sst_results/
          └── {job_id}_result.npy
        """
		# Write job files
		job_meta = {
		    'job_id': job_id,
		    'op_type': 'GEMM',
		    'operation': op.to_dict(),
		    'timestamp': time.time()
		}

		job_file = os.path.join(self.sst_job_dir, f"{job_id}.json")
		A_file = os.path.join(self.sst_job_dir, f"{job_id}_A.npy")
		B_file = os.path.join(self.sst_job_dir, f"{job_id}_B.npy")
		result_file = os.path.join(self.sst_result_dir, f"{job_id}_result.npy")

		# Save matrices
		np.save(A_file, A)
		np.save(B_file, B)

		# Save metadata (this signals SST that job is ready)
		with open(job_file, 'w') as f:
			json.dump(job_meta, f, indent=2)

		print(f"  Job submitted: {job_id}")
		print(f"  Waiting for SST to process...")

		# Wait for result (with timeout)
		timeout = 30  # seconds
		start_time = time.time()

		while not os.path.exists(result_file):
			if time.time() - start_time > timeout:
				print(f"  ⚠ Timeout waiting for SST, falling back to CPU")
				return np.matmul(A, B)
			time.sleep(0.1)

		# Load result
		result = np.load(result_file)

		# Cleanup
		os.remove(job_file)
		os.remove(A_file)
		os.remove(B_file)
		os.remove(result_file)

		print(f"  ✓ SST result received")
		return result


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
	print("QEMU Device Server v2 - Filesystem-based SST")
	print("=" * 60)
	print()

	server = QEMUDeviceServer()
	server.run()
