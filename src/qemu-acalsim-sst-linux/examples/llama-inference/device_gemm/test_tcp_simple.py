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
"""Simple TCP connection test"""
import socket
import struct

print("Connecting to localhost:9999...")
try:
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.connect(("localhost", 9999))
	print("✓ Connected!")

	# Send a simple test message
	test_msg = b"HELLO"
	print(f"Sending: {test_msg}")
	sock.sendall(test_msg)

	# Try to receive response
	print("Waiting for response...")
	resp = sock.recv(1024)
	print(f"Received: {resp}")

	sock.close()
	print("✓ Test complete")

except Exception as e:
	print(f"✗ Error: {e}")
	import traceback
	traceback.print_exc()
