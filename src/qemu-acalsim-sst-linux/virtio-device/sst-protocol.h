/*
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
 * VirtIO SST Protocol
 *
 * This header defines the protocol for communication between:
 * - Linux kernel driver (virtio-sst.ko)
 * - QEMU VirtIO SST device (virtio-sst.c)
 * - SST simulator components
 *
 * Communication flow:
 * 1. Kernel driver prepares SSTRequest in virtqueue buffer
 * 2. Driver kicks virtqueue to notify QEMU device
 * 3. QEMU device forwards request to SST via Unix socket
 * 4. SST processes request and returns response
 * 5. QEMU places SSTResponse in virtqueue buffer
 * 6. QEMU triggers interrupt to notify driver
 * 7. Driver processes response and completes I/O operation
 */

#ifndef SST_PROTOCOL_H
#define SST_PROTOCOL_H

#ifdef __KERNEL__
#include <linux/types.h>
/* Define stdint-style types for kernel space */
typedef u8  uint8_t;
typedef u16 uint16_t;
typedef u32 uint32_t;
typedef u64 uint64_t;
#else
#include <stdint.h>
#endif

/*
 * VirtIO Device and Vendor IDs
 */
#define VIRTIO_ID_SST        0x1042  // Device ID for VirtIO SST
#define VIRTIO_VENDOR_ID_SST 0x1AF4  // Red Hat vendor ID

/*
 * VirtQueue Indices
 */
#define SST_VQ_REQUEST  0  // Request queue (driver -> device)
#define SST_VQ_RESPONSE 1  // Response queue (device -> driver)
#define SST_VQ_EVENT    2  // Event queue (device -> driver, async)
#define SST_VQ_MAX      3

/*
 * Request Types
 */
enum SSTRequestType {
	SST_REQ_NOOP      = 0,  // No operation (test connectivity)
	SST_REQ_ECHO      = 1,  // Echo request (returns same data)
	SST_REQ_COMPUTE   = 2,  // Compute request (SST simulation)
	SST_REQ_READ      = 3,  // Read from SST device memory
	SST_REQ_WRITE     = 4,  // Write to SST device memory
	SST_REQ_RESET     = 5,  // Reset device state
	SST_REQ_GET_INFO  = 6,  // Get device information
	SST_REQ_CONFIGURE = 7,  // Configure device parameters
};

/*
 * Response Status Codes
 */
enum SSTStatus {
	SST_STATUS_OK        = 0,  // Request completed successfully
	SST_STATUS_ERROR     = 1,  // Generic error
	SST_STATUS_BUSY      = 2,  // Device busy, retry later
	SST_STATUS_INVALID   = 3,  // Invalid request
	SST_STATUS_TIMEOUT   = 4,  // Request timed out
	SST_STATUS_NO_DEVICE = 5,  // SST device not connected
};

/*
 * Device Features (capability bits)
 */
#define SST_FEATURE_ECHO        (1ULL << 0)  // Echo requests supported
#define SST_FEATURE_COMPUTE     (1ULL << 1)  // Compute operations supported
#define SST_FEATURE_MEMORY      (1ULL << 2)  // Memory read/write supported
#define SST_FEATURE_EVENTS      (1ULL << 3)  // Async events supported
#define SST_FEATURE_MULTI_QUEUE (1ULL << 4)  // Multiple queue pairs supported
#define SST_FEATURE_RESET       (1ULL << 5)  // Device reset supported

/*
 * Request Structure
 *
 * Sent from kernel driver to QEMU device via request virtqueue.
 * Maximum size: 4KB (single page for simplicity)
 */
#define SST_MAX_DATA_SIZE 4080

struct SSTRequest {
	uint32_t type;        // Request type (enum SSTRequestType)
	uint32_t flags;       // Request flags (reserved)
	uint64_t request_id;  // Unique request identifier
	uint64_t user_data;   // User-defined data (opaque to device)

	union {
		// COMPUTE request payload
		struct {
			uint64_t compute_units;  // Amount of computation
			uint32_t latency_model;  // Latency model to use
			uint32_t reserved;
		} compute;

		// READ/WRITE request payload
		struct {
			uint64_t addr;  // Target address
			uint32_t size;  // Transfer size
			uint32_t reserved;
		} memory;

		// CONFIGURE request payload
		struct {
			uint32_t param_id;  // Parameter ID
			uint32_t reserved;
			uint64_t value;  // Parameter value
		} config;

		// Generic data buffer
		uint8_t data[SST_MAX_DATA_SIZE];
	} payload;
} __attribute__((packed));

/*
 * Response Structure
 *
 * Returned from QEMU device to kernel driver via response virtqueue.
 */
struct SSTResponse {
	uint32_t status;  // Response status (enum SSTStatus)
	uint32_t reserved;
	uint64_t request_id;  // Matching request_id from SSTRequest
	uint64_t user_data;   // Echo of user_data from request
	uint64_t result;      // Operation result (type-dependent)

	union {
		// COMPUTE response
		struct {
			uint64_t cycles;     // Simulated cycles
			uint64_t timestamp;  // Simulation timestamp
		} compute;

		// GET_INFO response
		struct {
			uint32_t version;            // Device version
			uint32_t capabilities;       // Capability flags
			uint64_t max_compute_units;  // Maximum compute units
			uint64_t memory_size;        // Device memory size
		} info;

		// Generic data buffer (for READ operations)
		uint8_t data[SST_MAX_DATA_SIZE];
	} payload;
} __attribute__((packed));

/*
 * Event Structure
 *
 * Asynchronous events sent from QEMU device to kernel driver.
 * Used for notifications that don't correspond to specific requests.
 */
struct SSTEvent {
	uint32_t type;  // Event type
	uint32_t reserved;
	uint64_t timestamp;  // Event timestamp
	uint64_t data;       // Event-specific data
} __attribute__((packed));

/*
 * Configuration Space
 *
 * Read-only device configuration accessible via VirtIO config space.
 */
struct SSTConfig {
	uint32_t version;     // Protocol version
	uint32_t device_id;   // SST device ID
	uint64_t features;    // Supported features
	uint64_t max_queues;  // Maximum number of queue pairs
} __attribute__((packed));

/*
 * Protocol Version
 */
#define SST_PROTOCOL_VERSION 0x00010000  // Version 1.0.0

/*
 * Helper Functions (inline for both kernel and userspace)
 */
static inline const char* sst_request_type_str(uint32_t type) {
	switch (type) {
		case SST_REQ_NOOP: return "NOOP";
		case SST_REQ_ECHO: return "ECHO";
		case SST_REQ_COMPUTE: return "COMPUTE";
		case SST_REQ_READ: return "READ";
		case SST_REQ_WRITE: return "WRITE";
		case SST_REQ_RESET: return "RESET";
		case SST_REQ_GET_INFO: return "GET_INFO";
		case SST_REQ_CONFIGURE: return "CONFIGURE";
		default: return "UNKNOWN";
	}
}

static inline const char* sst_status_str(uint32_t status) {
	switch (status) {
		case SST_STATUS_OK: return "OK";
		case SST_STATUS_ERROR: return "ERROR";
		case SST_STATUS_BUSY: return "BUSY";
		case SST_STATUS_INVALID: return "INVALID";
		case SST_STATUS_TIMEOUT: return "TIMEOUT";
		case SST_STATUS_NO_DEVICE: return "NO_DEVICE";
		default: return "UNKNOWN";
	}
}

#endif /* SST_PROTOCOL_H */
