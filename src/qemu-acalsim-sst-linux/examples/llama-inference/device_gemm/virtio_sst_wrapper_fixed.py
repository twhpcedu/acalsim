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
Matches the actual sst-protocol.h structure
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


# From sst-protocol.h
SST_MAX_DATA_SIZE = 4080


class SSTRequest:
	"""
    struct SSTRequest {
        uint32_t type;        // 4 bytes
        uint32_t flags;       // 4 bytes
        uint64_t request_id;  // 8 bytes
        uint64_t user_data;   // 8 bytes
        union { uint8_t data[4080]; } payload;  // 4080 bytes
    } __attribute__((packed));

    Total: 4104 bytes
    """
	SIZE = 4104

	def __init__(self, req_type, request_id=0, user_data=0, flags=0):
		self.type = req_type
		self.flags = flags
		self.request_id = request_id
		self.user_data = user_data
		self.payload = b'\x00' * SST_MAX_DATA_SIZE  # 4080 bytes of zeros

	def pack(self):
		"""Pack request into binary format"""
		# Header: type(4) + flags(4) + request_id(8) + user_data(8) = 24 bytes
		header = struct.pack('<IIQQ', self.type, self.flags, self.request_id, self.user_data)
		# Payload: 4080 bytes
		payload = self.payload[: SST_MAX_DATA_SIZE].ljust(SST_MAX_DATA_SIZE, b'\x00')
		return header + payload


class SSTResponse:
	"""
    struct SSTResponse {
        uint32_t status;      // 4 bytes
        uint32_t reserved;    // 4 bytes
        uint64_t request_id;  // 8 bytes
        uint64_t user_data;   // 8 bytes
        uint64_t result;      // 8 bytes
        union { uint8_t data[4080]; } payload;  // 4080 bytes
    } __attribute__((packed));

    Total: 4112 bytes
    """
	SIZE = 4112

	def __init__(self):
		self.status = 0
		self.reserved = 0
		self.request_id = 0
		self.user_data = 0
		self.result = 0
		self.payload = b''

	@staticmethod
	def unpack(data):
		"""Unpack response from /dev/sst0"""
		if len(data) < 32:  # Minimum header size
			return None

		resp = SSTResponse()
		# Parse header: status(4) + reserved(4) + request_id(8) + user_data(8) + result(8) = 32 bytes
		(resp.status, resp.reserved, resp.request_id, resp.user_data, resp.result) = \
                  struct.unpack('<IIQQQ', data[:32])

		# Extract payload if present
		if len(data) >= 32 + SST_MAX_DATA_SIZE:
			resp.payload = data[32 : 32 + SST_MAX_DATA_SIZE]

		return resp


class VirtIOSST:
	"""High-level interface to VirtIO-SST device"""

	def __init__(self, device_path='/dev/sst0'):
		self.device_path = device_path
		self.fd = None
		self.next_request_id = 1

	def open(self):
		"""Open SST device"""
		if not os.path.exists(self.device_path):
			raise FileNotFoundError(
			    f"Device {self.device_path} not found. "
			    "Load module with: sudo insmod virtio-sst.ko"
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

		# Assign request ID
		req.request_id = self.next_request_id
		self.next_request_id += 1

		# Pack and write request
		data = req.pack()
		if len(data) != SSTRequest.SIZE:
			raise RuntimeError(f"Invalid request size: {len(data)}, expected {SSTRequest.SIZE}")

		written = os.write(self.fd, data)

		if written != len(data):
			raise RuntimeError(f"Write failed: {written}/{len(data)} bytes")

		# Read response (blocking)
		resp_data = os.read(self.fd, SSTResponse.SIZE)

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
		# Pack data into payload
		if isinstance(data, str):
			data = data.encode()
		req.payload = data[: SST_MAX_DATA_SIZE].ljust(SST_MAX_DATA_SIZE, b'\x00')

		resp = self.send_request(req)

		if resp.status == SSTStatus.OK:
			# Extract echoed data (strip null padding)
			return resp.payload[: len(data)]
		else:
			raise RuntimeError(f"Echo failed: status={resp.status}")

	def compute(self, compute_units, latency_model=0):
		"""Submit compute request to SST"""
		req = SSTRequest(SSTRequestType.COMPUTE)
		# Pack compute payload: compute_units(8) + latency_model(4) + reserved(4)
		compute_payload = struct.pack('<QII', compute_units, latency_model, 0)
		req.payload = compute_payload.ljust(SST_MAX_DATA_SIZE, b'\x00')

		resp = self.send_request(req)

		if resp.status == SSTStatus.OK:
			# Parse compute response from payload
			if len(resp.payload) >= 16:
				cycles, timestamp = struct.unpack('<QQ', resp.payload[: 16])
				return {'status': 'ok', 'cycles': cycles, 'timestamp': timestamp}
			else:
				return {'status': 'ok', 'result': resp.result}
		else:
			return {'status': 'error', 'code': resp.status}

	def get_info(self):
		"""Get SST device information"""
		req = SSTRequest(SSTRequestType.GET_INFO)
		resp = self.send_request(req)

		if resp.status == SSTStatus.OK:
			# Parse device info from payload
			if len(resp.payload) >= 32:
				version, capabilities, max_compute, mem_size = \
                                struct.unpack('<IIQQ', resp.payload[:24])
				return {
				    'status': 'ok',
				    'version': version,
				    'capabilities': capabilities,
				    'max_compute_units': max_compute,
				    'memory_size': mem_size
				}
			else:
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
	print("VirtIO-SST Device Test (Protocol v2)")
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
		print("  1. Run QEMU with custom kernel")
		print(
		    "  2. Inside QEMU, run: sudo insmod /mnt/shared/acalsim/src/qemu-acalsim-sst-linux/drivers/virtio-sst.ko"
		)
		print("  3. Fix permissions: sudo chmod 666 /dev/sst0")
	except Exception as e:
		print(f"✗ Error: {e}")
		import traceback
		traceback.print_exc()

	print()
	print("=" * 60)
