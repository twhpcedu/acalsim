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
"""User-space SST Bridge"""
import socket, struct, os


class SSTBridge:

	def __init__(self):
		self.qemu_sock = "/tmp/sst-bridge.sock"
		self.sst_sock = "/tmp/qemu-sst-llama.sock"

	def connect_sst(self):
		if not os.path.exists(self.sst_sock):
			print(f"SST not found")
			return None
		try:
			s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
			s.connect(self.sst_sock)
			print("Connected to SST")
			return s
		except:
			return None

	def run(self):
		print("=" * 60)
		print("User-Space SST Bridge")
		print("=" * 60)

		sst = self.connect_sst()
		if os.path.exists(self.qemu_sock): os.remove(self.qemu_sock)

		srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		srv.bind(self.qemu_sock)
		srv.listen(5)
		print(f"Listening on {self.qemu_sock}")

		while True:
			conn, _ = srv.accept()
			print("QEMU connected")
			while True:
				try:
					hdr = self.recv_exact(conn, 16)
					if not hdr: break
					req_type, req_id, psize = struct.unpack("<IQI", hdr)
					payload = self.recv_exact(conn, psize) if psize > 0 else b""

					print(f"  Req: type={req_type} id={req_id}")

					if sst:
						sst_req = struct.pack(
						    "<IQIQI", req_type, req_id, 0, 0, len(payload)
						) + payload
						sst.sendall(sst_req)
						sst_resp = self.recv_exact(sst, 4096)
						if sst_resp and len(sst_resp) >= 28:
							status, rid, _, rsize = struct.unpack("<IQQI", sst_resp[: 28])
							rpayload = sst_resp[28 : 28 + rsize]
							print(f"  Resp: status={status}")
							conn.sendall(struct.pack("<IQI", status, rid, len(rpayload)) + rpayload)
						else:
							conn.sendall(struct.pack("<IQI", 1, req_id, 0))
					else:
						conn.sendall(struct.pack("<IQI", 5, req_id, 0))
				except:
					break
			conn.close()

	def recv_exact(self, s, n):
		d = b""
		while len(d) < n:
			c = s.recv(n - len(d))
			if not c: return None
			d += c
		return d


if __name__ == "__main__":
	SSTBridge().run()
