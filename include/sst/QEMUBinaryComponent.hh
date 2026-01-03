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

#ifndef QEMU_BINARY_COMPONENT_HH
#define QEMU_BINARY_COMPONENT_HH

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sys/types.h>

#include <map>
#include <string>
#include <vector>

// Include event definitions from ACALSim device
#include "sst/ACALSimDeviceComponent.hh"

namespace ACALSim {
namespace QEMUBinary {

// Re-use event classes from QEMUIntegration
using ACALSim::QEMUIntegration::MemoryResponseEvent;
using ACALSim::QEMUIntegration::MemoryTransactionEvent;
using ACALSim::QEMUIntegration::TransactionType;

// Binary MMIO protocol structures
struct MMIORequest {
	uint8_t  type;  // 0 = READ, 1 = WRITE
	uint8_t  size;  // 1, 2, 4, or 8 bytes
	uint16_t reserved;
	uint64_t addr;  // MMIO address
	uint64_t data;  // Write data (ignored for READ)
} __attribute__((packed));

struct MMIOResponse {
	uint8_t  success;  // 0 = error, 1 = success
	uint8_t  reserved[7];
	uint64_t data;  // Read data or write acknowledgment
} __attribute__((packed));

// QEMU state
enum class QEMUState {
	IDLE,            // Not started
	LAUNCHING,       // Starting QEMU process
	RUNNING,         // QEMU running, processing MMIO
	WAITING_DEVICE,  // Waiting for SST device response
	COMPLETED,       // QEMU exited successfully
	ERROR            // Error occurred
};

// Pending MMIO request (waiting for SST device response)
struct PendingMMIORequest {
	MMIORequest request;     // Original MMIO request from QEMU
	uint64_t    sst_req_id;  // SST request ID
};

// Device information for routing (N-device mode)
struct DeviceInfo {
	uint64_t    base_addr;     // Device base address
	uint64_t    size;          // Device memory size
	SST::Link*  link;          // Link to device
	std::string name;          // Device name (for debugging)
	uint64_t    num_requests;  // Statistics: requests routed to this device

	// N-socket support: per-device socket connections
	std::string socket_path;   // Unix socket path (e.g., /tmp/qemu-sst-device0.sock)
	int         server_fd;     // Server socket file descriptor
	int         client_fd;     // Client connection file descriptor
	bool        socket_ready;  // True when client connected
};

/*
 * QEMUBinaryComponent - Phase 2C/2D
 *
 * SST component that manages a QEMU subprocess and bridges communication
 * via binary MMIO protocol instead of text-based serial protocol.
 *
 * Features:
 * - Binary protocol: No text parsing overhead
 * - MMIO-based: Direct memory-mapped I/O instead of UART
 * - N-device support: Route transactions to multiple devices based on address
 * - Backward compatible: Single device mode for legacy configurations
 *
 * Improvements over Phase 2B (QEMURealComponent):
 * - 10x throughput improvement (~10,000 tx/sec)
 * - 10x lower latency (~100μs/tx)
 * - 90% reduction in CPU usage
 * - Cleaner architecture: Standard QEMU device model
 */
class QEMUBinaryComponent : public SST::Component {
public:
	// SST registration macro
	SST_ELI_REGISTER_COMPONENT(QEMUBinaryComponent, "qemubinary", "QEMUBinary", SST_ELI_ELEMENT_VERSION(1, 0, 0),
	                           "QEMU Binary MMIO Component - Phase 2C with binary protocol",
	                           COMPONENT_CATEGORY_PROCESSOR)

	// Parameter documentation
	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency", "1GHz"}, {"verbose", "Verbosity level (0-3)", "1"},
	                        {"binary_path", "Path to RISC-V ELF binary", ""},
	                        {"qemu_path", "Path to qemu-system-riscv32", "qemu-system-riscv32"},
	                        {"socket_path", "Path to Unix socket for MMIO", "/tmp/qemu-sst-mmio.sock"},
	                        {"device_base", "SST device base address (hex) - legacy single device", "0x20000000"},
	                        {"num_devices", "Number of devices to support (N-device mode)", "1"},
	                        {"device%d_base", "Base address for device %d", "0x10000000"},
	                        {"device%d_size", "Memory size for device %d", "4096"},
	                        {"device%d_name", "Name for device %d (optional)", "device%d"})

	// Port documentation
	SST_ELI_DOCUMENT_PORTS({"device_port",
	                        "Port to memory/device subsystem (legacy single device)",
	                        {"MemoryTransactionEvent"}},
	                       {"device_port_%d", "Port to device %d (N-device mode)", {"MemoryTransactionEvent"}})

	// Constructor and destructor
	QEMUBinaryComponent(SST::ComponentId_t id, SST::Params& params);
	~QEMUBinaryComponent();

	// SST lifecycle methods
	void setup() override;
	void finish() override;

private:
	// Clock handler
	bool clockTick(SST::Cycle_t cycle);

	// Event handlers
	void handleDeviceResponse(SST::Event* ev);

	// QEMU process management
	void launchQEMU();
	void monitorQEMU();
	void terminateQEMU();
	bool isQEMURunning();

	// Socket communication
	void setupSocket();
	void handleMMIORequest();                                                // Legacy single-device mode
	void handleMMIORequest(DeviceInfo* device);                              // N-device mode
	void sendMMIOResponse(bool success, uint64_t data);                      // Legacy
	void sendMMIOResponse(DeviceInfo* device, bool success, uint64_t data);  // N-device

	// Per-device socket management (N-device mode)
	void setupDeviceSocket(DeviceInfo* device, int index);
	void acceptDeviceConnection(DeviceInfo* device);
	void pollDeviceSockets();

	// SST device communication
	void sendDeviceRequest(uint8_t type, uint64_t addr, uint64_t data, uint8_t size);

	// Device routing (N-device mode)
	DeviceInfo* findDeviceForAddress(uint64_t address);
	void        routeToDevice(DeviceInfo* device, uint8_t type, uint64_t addr, uint64_t data, uint8_t size);

	// State management
	void setState(QEMUState new_state);

	// Member variables - Output
	SST::Output out_;

	// Member variables - Timing
	SST::Cycle_t        current_cycle_;
	SST::TimeConverter* tc_;

	// Member variables - QEMU process
	pid_t       qemu_pid_;
	std::string binary_path_;
	std::string qemu_path_;
	std::string socket_path_;
	uint64_t    device_base_;
	QEMUState   state_;

	// Member variables - Socket communication
	int  server_fd_;  // Server socket (SST listens)
	int  client_fd_;  // Client socket (QEMU connects)
	bool socket_ready_;

	// Member variables - SST communication
	SST::Link*                             device_link_;       // Legacy single device (backward compatibility)
	std::vector<DeviceInfo>                devices_;           // N-device mode: multiple devices
	bool                                   use_multi_device_;  // True if num_devices > 1
	int                                    num_devices_;       // Number of devices configured
	std::map<uint64_t, PendingMMIORequest> pending_requests_;
	uint64_t                               next_req_id_;

	// Member variables - Statistics
	uint64_t total_reads_;
	uint64_t total_writes_;
	uint64_t total_bytes_;
	uint64_t successful_transactions_;
	uint64_t failed_transactions_;
};

}  // namespace QEMUBinary
}  // namespace ACALSim

#endif  // QEMU_BINARY_COMPONENT_HH
