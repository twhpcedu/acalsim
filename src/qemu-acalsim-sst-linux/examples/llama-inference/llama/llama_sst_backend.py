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
SST Accelerator Backend for LLAMA 2 Inference

Copyright 2023-2025 Playlab/ACAL
Licensed under the Apache License, Version 2.0
"""

import os
import struct


class SSTAcceleratorBackend:
	"""Interface to SST accelerators via /dev/sst0"""

	# SST protocol constants
	SST_REQ_NOOP = 0
	SST_REQ_ECHO = 1
	SST_REQ_COMPUTE = 2
	SST_REQ_GET_INFO = 3
	SST_REQ_RESET = 4

	SST_STATUS_OK = 0
	SST_STATUS_ERROR = 1

	def __init__(self):
		self.device_path = "/dev/sst0"
		self.fd = None
		self.device_available = False
		self.stats = {
		    "attention_ops": 0,
		    "ffn_ops": 0,
		    "embedding_ops": 0,
		    "total_cycles": 0,
		    "total_requests": 0
		}

	def check_device(self):
		"""Check if SST device is available"""
		self.device_available = os.path.exists(self.device_path)
		return self.device_available

	def open_device(self):
		"""Open SST device"""
		if not self.device_available:
			return False

		if self.fd is None:
			try:
				self.fd = os.open(self.device_path, os.O_RDWR)
				return True
			except OSError as e:
				print(f"WARNING: Failed to open {self.device_path}: {e}")
				self.device_available = False
				return False
		return True

	def close_device(self):
		"""Close SST device"""
		if self.fd is not None:
			try:
				os.close(self.fd)
			except:
				pass
			self.fd = None

	def send_compute_request(self, compute_units, latency_model=0):
		"""
        Send compute request to SST

        Args:
            compute_units: Number of compute units for this operation
            latency_model: Latency model selector (0=attention, 1=ffn, 2=embedding)

        Returns:
            cycles: Simulated cycles, or 0 if device unavailable
        """
		if not self.open_device():
			# Estimate cycles if device not available
			latency_per_unit = [1000, 500, 100][latency_model]  # ns
			return compute_units * latency_per_unit

		# Pack SSTRequest
		# struct SSTRequest {
		#     uint32_t type;
		#     uint32_t request_id;
		#     uint64_t user_data;
		#     union {
		#         struct { uint64_t compute_units; uint64_t latency_model; } compute;
		#         ...
		#     } payload;
		# };
		request_id = self.stats["total_requests"]
		self.stats["total_requests"] += 1

		request = struct.pack(
		    '<II Q QQ 4040x',  # Pad to 4096 bytes
		    self.SST_REQ_COMPUTE,  # type
		    request_id,  # request_id
		    0,  # user_data
		    compute_units,  # payload.compute.compute_units
		    latency_model  # payload.compute.latency_model
		)

		try:
			# Send request
			os.write(self.fd, request)

			# Receive response
			response = os.read(self.fd, 4096)

			# Parse SSTResponse
			# struct SSTResponse {
			#     uint32_t status;
			#     uint32_t request_id;
			#     uint64_t user_data;
			#     union {
			#         struct { uint64_t cycles; } compute;
			#         ...
			#     } payload;
			# };
			status, resp_id, user_data, cycles = struct.unpack('<II Q Q', response[: 24])

			if status == self.SST_STATUS_OK:
				return cycles
			else:
				print(f"WARNING: SST request {resp_id} returned error status {status}")
				return 0

		except Exception as e:
			print(f"WARNING: SST communication error: {e}")
			self.device_available = False
			return 0

	def simulate_attention_op(self, batch_size=1, seq_len=128):
		"""Simulate attention operation"""
		compute_units = batch_size * seq_len
		cycles = self.send_compute_request(compute_units, latency_model=0)

		self.stats["attention_ops"] += 1
		self.stats["total_cycles"] += cycles

		return cycles

	def simulate_ffn_op(self, batch_size=1, hidden_size=4096):
		"""Simulate feed-forward network operation"""
		compute_units = batch_size * hidden_size
		cycles = self.send_compute_request(compute_units, latency_model=1)

		self.stats["ffn_ops"] += 1
		self.stats["total_cycles"] += cycles

		return cycles

	def simulate_embedding_op(self, num_tokens=1):
		"""Simulate embedding lookup operation"""
		compute_units = num_tokens
		cycles = self.send_compute_request(compute_units, latency_model=2)

		self.stats["embedding_ops"] += 1
		self.stats["total_cycles"] += cycles

		return cycles

	def instrument_model(self, model):
		"""
        Instrument PyTorch model to track operations via SST

        This is a simple hook-based approach. For production, consider
        using torch.fx or custom CUDA kernels.
        """
		# Hook attention layers
		try:
			for name, module in model.named_modules():
				if "self_attn" in name or "attention" in name.lower():
					self._hook_attention_layer(module)
				elif "mlp" in name or "ffn" in name.lower():
					self._hook_ffn_layer(module)
		except Exception as e:
			print(f"WARNING: Failed to instrument model: {e}")

		return model

	def _hook_attention_layer(self, module):
		"""Add hook to attention layer"""

		def forward_hook(module, input, output):
			if len(input) > 0:
				hidden_states = input[0]
				batch_size = hidden_states.size(0)
				seq_len = hidden_states.size(1) if len(hidden_states.size()) > 1 else 1
				self.simulate_attention_op(batch_size, seq_len)
			return output

		module.register_forward_hook(forward_hook)

	def _hook_ffn_layer(self, module):
		"""Add hook to FFN layer"""

		def forward_hook(module, input, output):
			if len(input) > 0:
				hidden_states = input[0]
				batch_size = hidden_states.size(0)
				hidden_size = hidden_states.size(-1)
				self.simulate_ffn_op(batch_size, hidden_size)
			return output

		module.register_forward_hook(forward_hook)

	def reset_stats(self):
		"""Reset statistics"""
		self.stats = {
		    "attention_ops": 0,
		    "ffn_ops": 0,
		    "embedding_ops": 0,
		    "total_cycles": 0,
		    "total_requests": self.stats["total_requests"]  # Keep request counter
		}

	def print_stats(self):
		"""Print statistics"""
		print("\n" + "=" * 60)
		print("SST Accelerator Statistics:")
		print("=" * 60)
		print(f"  Attention operations:  {self.stats['attention_ops']:>10,}")
		print(f"  FFN operations:        {self.stats['ffn_ops']:>10,}")
		print(f"  Embedding operations:  {self.stats['embedding_ops']:>10,}")
		print(f"  Total requests:        {self.stats['total_requests']:>10,}")
		print(f"  Total simulated cycles: {self.stats['total_cycles']:>10,}")

		# Estimate wall-clock time at 2GHz
		if self.stats['total_cycles'] > 0:
			time_sec = self.stats['total_cycles'] / 2e9  # 2GHz = 2e9 cycles/sec
			if time_sec < 1e-3:
				time_us = time_sec * 1e6
				print(f"  Estimated time @ 2GHz:  {time_us:>10.2f} µs")
			elif time_sec < 1:
				time_ms = time_sec * 1e3
				print(f"  Estimated time @ 2GHz:  {time_ms:>10.2f} ms")
			else:
				print(f"  Estimated time @ 2GHz:  {time_sec:>10.2f} s")

		print("=" * 60)

	def close(self):
		"""Cleanup"""
		self.close_device()

	def __del__(self):
		self.close()
