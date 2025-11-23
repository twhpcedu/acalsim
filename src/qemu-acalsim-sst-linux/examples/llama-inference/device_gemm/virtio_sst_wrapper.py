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
Python wrapper for /dev/sst0 VirtIO-SST device
Provides high-level interface for SST communication
"""
import struct
import os


# SST Protocol Constants (from sst-protocol.h)
class SSTRequestType:
	NOOP = 0
	ECHO = 1
	COMPUTE = 2
	READ = 3
	WRITE = 4
	RESET = 5
	GET_INFO = 6
	CONFIGURE = 7


class SSTStatus:
	OK = 0
	ERROR = 1
	BUSY = 2
	INVALID = 3
	TIMEOUT = 4
	NO_DEVICE = 5


# Request structure (simplified Python representation)
# Full C structure is more complex, but we'll use binary packing
class SSTRequest:

	def __init__(self, req_type, user_data=0):
		self.type = req_type
		self.request_id = 0  # Assigned by driver
		self.flags = 0
		self.user_data = user_data
		self.payload_size = 0
		self.payload = b''

	def pack(self):
		"""Pack request into binary format for /dev/sst0"""
		# Simplified format: type(4) + request_id(8) + flags(4) + user_data(8) + payload_size(4)
		header = struct.pack(
		    '<IQIQI', self.type, self.request_id, self.flags, self.user_data, len(self.payload)
		)
		return header + self.payload


class SSTResponse:

	def __init__(self):
		self.status = 0
		self.request_id = 0
		self.user_data = 0
		self.payload_size = 0
		self.payload = b''

	@staticmethod
	def unpack(data):
		"""Unpack response from /dev/sst0"""
		resp = SSTResponse()
		if len(data) < 28:  # Minimum header size
			return None

		# Parse header
		(resp.status, resp.request_id, resp.user_data, resp.payload_size) = \
                                                      struct.unpack('<IQQI', data[:28])

		# Extract payload if present
		if resp.payload_size > 0 and len(data) >= 28 + resp.payload_size:
			resp.payload = data[28 : 28 + resp.payload_size]

		return resp


class VirtIOSST:
	"""High-level interface to VirtIO-SST device"""

	def __init__(self, device_path='/dev/sst0'):
		self.device_path = device_path
		self.fd = None

	def open(self):
		"""Open SST device"""
		if not os.path.exists(self.device_path):
			raise FileNotFoundError(
			    f"Device {self.device_path} not found. "
			    "Run setup_virtio_sst.sh first."
			)

		self.fd = os.open(self.device_path, os.O_RDWR)
		print(f"✓ Opened {self.device_path}")

	def close(self):
		"""Close SST device"""
		if self.fd is not None:
			os.close(self.fd)
			self.fd = None

	def send_request(self, req):
		"""Send request to SST and wait for response"""
		if self.fd is None:
			raise RuntimeError("Device not opened")

		# Pack and write request
		data = req.pack()
		written = os.write(self.fd, data)

		if written != len(data):
			raise RuntimeError(f"Write failed: {written}/{len(data)} bytes")

		# Read response (blocking)
		resp_data = os.read(self.fd, 4096)  # Max response size

		# Unpack response
		resp = SSTResponse.unpack(resp_data)

		if resp is None:
			raise RuntimeError("Failed to parse response")

		return resp

	def ping(self):
		"""Test connectivity with NOOP request"""
		req = SSTRequest(SSTRequestType.NOOP, user_data=0x12345678)
		resp = self.send_request(req)

		if resp.status == SSTStatus.OK:
			print(f"✓ SST device ping successful (request_id={resp.request_id})")
			return True
		else:
			print(f"✗ SST device ping failed (status={resp.status})")
			return False

	def echo(self, data):
		"""Send echo request"""
		req = SSTRequest(SSTRequestType.ECHO)
		req.payload = data if isinstance(data, bytes) else data.encode()
		resp = self.send_request(req)

		if resp.status == SSTStatus.OK:
			return resp.payload
		else:
			raise RuntimeError(f"Echo failed: status={resp.status}")

	def compute(self, compute_units, latency_model=0):
		"""Submit compute request to SST"""
		req = SSTRequest(SSTRequestType.COMPUTE)
		# Pack compute payload: compute_units(8) + latency_model(4)
		req.payload = struct.pack('<QI', compute_units, latency_model)

		resp = self.send_request(req)

		if resp.status == SSTStatus.OK and len(resp.payload) >= 8:
			cycles = struct.unpack('<Q', resp.payload[: 8])[0]
			return {'status': 'ok', 'cycles': cycles}
		else:
			return {'status': 'error', 'code': resp.status}

	def get_info(self):
		"""Get SST device information"""
		req = SSTRequest(SSTRequestType.GET_INFO)
		resp = self.send_request(req)

		if resp.status == SSTStatus.OK:
			# Parse device info from payload
			return {'status': 'ok', 'info': resp.payload}
		else:
			return {'status': 'error', 'code': resp.status}

	def __enter__(self):
		self.open()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()


# Example usage
if __name__ == '__main__':
	print("=" * 60)
	print("VirtIO-SST Device Test")
	print("=" * 60)
	print()

	try:
		with VirtIOSST() as sst:
			# Test 1: Ping
			print("Test 1: Ping SST device")
			sst.ping()
			print()

			# Test 2: Echo
			print("Test 2: Echo test")
			msg = b"Hello SST!"
			echo_resp = sst.echo(msg)
			print(f"  Sent: {msg}")
			print(f"  Received: {echo_resp}")
			print(f"  Match: {msg == echo_resp}")
			print()

			# Test 3: Compute
			print("Test 3: Compute request")
			result = sst.compute(compute_units=1000, latency_model=0)
			print(f"  Compute units: 1000")
			print(f"  Result: {result}")
			print()

			# Test 4: Get Info
			print("Test 4: Device info")
			info = sst.get_info()
			print(f"  Info: {info}")
			print()

			print("✓ All tests passed!")

	except FileNotFoundError as e:
		print(f"✗ {e}")
		print()
		print("Setup required:")
		print("  1. Run QEMU with: -device virtio-sst-device,socket=/tmp/qemu-sst-llama.sock")
		print("  2. Inside QEMU, run: bash /mnt/shared/device_gemm/setup_virtio_sst.sh")
		print("  3. Verify /dev/sst0 exists")
	except Exception as e:
		print(f"✗ Error: {e}")
		import traceback
		traceback.print_exc()

	print()
	print("=" * 60)
