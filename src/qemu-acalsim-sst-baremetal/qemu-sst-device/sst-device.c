/*
 * SST Device for QEMU - Phase 2C
 *
 * Copyright 2023-2026 Playlab/ACAL
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * QEMU SST Device - Binary MMIO Interface
 *
 * This QEMU device provides memory-mapped I/O interface for communication
 * with SST simulator via Unix socket and binary protocol.
 *
 * Memory Map (4KB region):
 *   0x00: DATA_IN     - Write data to SST device
 *   0x04: DATA_OUT    - Read data from SST device
 *   0x08: STATUS      - Device status (BUSY, DATA_READY, ERROR)
 *   0x0C: CONTROL     - Control register (START, RESET)
 *
 * Protocol: Binary MMIORequest/MMIOResponse structs over Unix socket
 *
 * Usage in QEMU:
 *   qemu-system-riscv32 -device sst-device,socket=/tmp/qemu-sst-mmio.sock
 */

#include "hw/qdev-properties.h"
#include "hw/sysbus.h"
#include "qapi/error.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/osdep.h"

#define TYPE_SST_DEVICE "sst-device"
#define SST_DEVICE(obj) OBJECT_CHECK(SSTDeviceState, (obj), TYPE_SST_DEVICE)

#define SST_DEVICE_SIZE 0x1000  // 4KB MMIO region

// Register offsets
#define SST_DATA_IN_OFFSET  0x00
#define SST_DATA_OUT_OFFSET 0x04
#define SST_STATUS_OFFSET   0x08
#define SST_CONTROL_OFFSET  0x0C

// Status bits
#define STATUS_BUSY       (1 << 0)
#define STATUS_DATA_READY (1 << 1)
#define STATUS_ERROR      (1 << 2)

// Control bits
#define CONTROL_START (1 << 0)
#define CONTROL_RESET (1 << 1)

// Binary protocol structures (must match QEMUBinaryComponent)
struct MMIORequest {
	uint8_t  type;  // 0 = READ, 1 = WRITE
	uint8_t  size;  // 1, 2, 4, or 8 bytes
	uint16_t reserved;
	uint64_t addr;  // MMIO address
	uint64_t data;  // Write data
} __attribute__((packed));

struct MMIOResponse {
	uint8_t  success;  // 0 = error, 1 = success
	uint8_t  reserved[7];
	uint64_t data;  // Read data
} __attribute__((packed));

typedef struct {
	SysBusDevice parent_obj;

	MemoryRegion iomem;

	// Socket connection to SST
	int   socket_fd;
	char* socket_path;
	bool  connected;

	// Device base address (for N-device routing)
	uint64_t base_address;

	// Device registers
	uint32_t data_in;
	uint32_t data_out;
	uint32_t status;
	uint32_t control;

	// Statistics
	uint64_t total_reads;
	uint64_t total_writes;
} SSTDeviceState;

/*
 * Send MMIO request to SST and wait for response
 */
static bool sst_send_request(SSTDeviceState* s, struct MMIORequest* req, struct MMIOResponse* resp) {
	if (!s->connected || s->socket_fd < 0) {
		qemu_log("SST Device: Not connected to SST\n");
		return false;
	}

	// Send request
	ssize_t sent = write(s->socket_fd, req, sizeof(*req));
	if (sent != sizeof(*req)) {
		qemu_log("SST Device: Failed to send request (sent %zd bytes)\n", sent);
		s->status |= STATUS_ERROR;
		return false;
	}

	// Wait for response
	ssize_t received = read(s->socket_fd, resp, sizeof(*resp));
	if (received != sizeof(*resp)) {
		qemu_log("SST Device: Failed to receive response (got %zd bytes)\n", received);
		s->status |= STATUS_ERROR;
		return false;
	}

	return resp->success != 0;
}

/*
 * MMIO Read Handler
 */
static uint64_t sst_device_read(void* opaque, hwaddr addr, unsigned size) {
	SSTDeviceState* s     = SST_DEVICE(opaque);
	uint64_t        value = 0;

	switch (addr) {
		case SST_DATA_IN_OFFSET: value = s->data_in; break;

		case SST_DATA_OUT_OFFSET:
			// Reading DATA_OUT triggers SST read transaction
			if (s->connected) {
				struct MMIORequest  req = {.type     = 0,  // READ
				                           .size     = size,
				                           .reserved = 0,
				                           .addr     = s->base_address + addr,  // Send global address
				                           .data     = 0};
				struct MMIOResponse resp;

				s->status |= STATUS_BUSY;
				if (sst_send_request(s, &req, &resp)) {
					s->data_out = (uint32_t)resp.data;
					s->status |= STATUS_DATA_READY;
					s->total_reads++;
				}
				s->status &= ~STATUS_BUSY;
			}
			value = s->data_out;
			break;

		case SST_STATUS_OFFSET: value = s->status; break;

		case SST_CONTROL_OFFSET: value = s->control; break;

		default: qemu_log("SST Device: Read from unknown offset 0x%lx\n", addr); break;
	}

	qemu_log("SST Device: Read  addr=0x%04lx size=%u value=0x%08lx\n", addr, size, value);

	return value;
}

/*
 * MMIO Write Handler
 */
static void sst_device_write(void* opaque, hwaddr addr, uint64_t val, unsigned size) {
	SSTDeviceState* s = SST_DEVICE(opaque);

	qemu_log("SST Device: Write addr=0x%04lx size=%u value=0x%08lx\n", addr, size, val);

	switch (addr) {
		case SST_DATA_IN_OFFSET:
			s->data_in = (uint32_t)val;
			// Writing DATA_IN doesn't automatically trigger transaction
			// Wait for CONTROL_START
			break;

		case SST_DATA_OUT_OFFSET:
			// DATA_OUT is read-only, ignore writes
			qemu_log("SST Device: Warning - Write to read-only DATA_OUT register\n");
			break;

		case SST_STATUS_OFFSET:
			// STATUS is mostly read-only, but can clear error bit
			if (val & STATUS_ERROR) { s->status &= ~STATUS_ERROR; }
			break;

		case SST_CONTROL_OFFSET:
			s->control = (uint32_t)val;

			if (val & CONTROL_RESET) {
				// Reset device
				s->data_in  = 0;
				s->data_out = 0;
				s->status   = 0;
				s->control  = 0;
				qemu_log("SST Device: Reset\n");
			} else if (val & CONTROL_START) {
				// Start transaction - send DATA_IN to SST
				if (s->connected) {
					struct MMIORequest  req = {.type     = 1,  // WRITE
					                           .size     = 4,
					                           .reserved = 0,
					                           .addr     = s->base_address + SST_DATA_IN_OFFSET,  // Send global address
					                           .data     = s->data_in};
					struct MMIOResponse resp;

					s->status |= STATUS_BUSY;
					s->status &= ~STATUS_DATA_READY;

					if (sst_send_request(s, &req, &resp)) {
						s->data_out = (uint32_t)resp.data;
						s->status |= STATUS_DATA_READY;
						s->total_writes++;
					}
					s->status &= ~STATUS_BUSY;
				} else {
					s->status |= STATUS_ERROR;
					qemu_log("SST Device: Not connected - cannot start transaction\n");
				}
			}
			break;

		default: qemu_log("SST Device: Write to unknown offset 0x%lx\n", addr); break;
	}
}

static const MemoryRegionOps sst_device_ops = {
    .read       = sst_device_read,
    .write      = sst_device_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid =
        {
            .min_access_size = 1,
            .max_access_size = 8,
        },
};

/*
 * Device Initialization
 */
static void sst_device_realize(DeviceState* dev, Error** errp) {
	SSTDeviceState*    s = SST_DEVICE(dev);
	struct sockaddr_un addr;

	qemu_log("SST Device: Initializing with socket: %s, base_address: 0x%016lx\n", s->socket_path, s->base_address);

	// Create MMIO region
	memory_region_init_io(&s->iomem, OBJECT(s), &sst_device_ops, s, TYPE_SST_DEVICE, SST_DEVICE_SIZE);
	sysbus_init_mmio(SYS_BUS_DEVICE(s), &s->iomem);

	// Initialize registers
	s->data_in      = 0;
	s->data_out     = 0;
	s->status       = 0;
	s->control      = 0;
	s->total_reads  = 0;
	s->total_writes = 0;

	// Connect to SST via Unix socket
	s->socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (s->socket_fd < 0) {
		error_setg(errp, "Failed to create socket");
		return;
	}

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, s->socket_path, sizeof(addr.sun_path) - 1);

	// Try to connect to SST server
	if (connect(s->socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		qemu_log("SST Device: Warning - Failed to connect to SST socket: %s\n", s->socket_path);
		qemu_log("SST Device: Device will be created but non-functional\n");
		qemu_log("SST Device: Make sure SST is running with socket server\n");
		close(s->socket_fd);
		s->socket_fd = -1;
		s->connected = false;
		// Don't fail initialization - allow QEMU to start
		return;
	}

	s->connected = true;
	qemu_log("SST Device: Connected to SST at %s\n", s->socket_path);
}

/*
 * Device Cleanup
 */
static void sst_device_unrealize(DeviceState* dev) {
	SSTDeviceState* s = SST_DEVICE(dev);

	qemu_log("SST Device: Shutting down\n");
	qemu_log("SST Device: Statistics - Reads: %lu, Writes: %lu\n", s->total_reads, s->total_writes);

	if (s->socket_fd >= 0) {
		close(s->socket_fd);
		s->socket_fd = -1;
	}

	s->connected = false;
}

/*
 * Device Properties
 */
static Property sst_device_properties[] = {
    DEFINE_PROP_STRING("socket", SSTDeviceState, socket_path),
    DEFINE_PROP_UINT64("base_address", SSTDeviceState, base_address, 0x10200000),
    DEFINE_PROP_END_OF_LIST(),
};

/*
 * Device Class Initialization
 */
static void sst_device_class_init(ObjectClass* klass, void* data) {
	DeviceClass* dc = DEVICE_CLASS(klass);

	dc->realize   = sst_device_realize;
	dc->unrealize = sst_device_unrealize;
	device_class_set_props(dc, sst_device_properties);
	set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

/*
 * Device Type Information
 */
static const TypeInfo sst_device_info = {
    .name          = TYPE_SST_DEVICE,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(SSTDeviceState),
    .class_init    = sst_device_class_init,
};

/*
 * Module Registration
 */
static void sst_device_register_types(void) { type_register_static(&sst_device_info); }

type_init(sst_device_register_types)
