/*
 * Copyright 2023-2025 Playlab/ACAL
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

#ifndef ACALSIM_VIRTIO_DEVICE_COMPONENT_HH
#define ACALSIM_VIRTIO_DEVICE_COMPONENT_HH

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <string>

namespace ACALSim {
namespace VirtIO {

/**
 * @brief SST Component for VirtIO SST Device Integration
 *
 * This component provides socket-based communication with QEMU's VirtIO SST device,
 * enabling Linux integration. Unlike MMIO-based components, this does not require
 * SST links to QEMU - all communication happens via Unix domain socket.
 *
 * Protocol: SST Request/Response protocol (defined in sst-protocol.h)
 * Operations: NOOP, ECHO, COMPUTE, etc.
 */
class ACALSimVirtIODeviceComponent : public SST::Component {
public:
	/**
	 * @brief Constructor
	 * @param id Component ID
	 * @param params Component parameters
	 */
	ACALSimVirtIODeviceComponent(SST::ComponentId_t id, SST::Params& params);

	/**
	 * @brief Destructor
	 */
	~ACALSimVirtIODeviceComponent();

	/**
	 * @brief SST setup phase
	 */
	void setup() override;

	/**
	 * @brief SST finish phase
	 */
	void finish() override;

	/**
	 * @brief Clock handler
	 * @param cycle Current cycle
	 * @return false to continue, true to stop
	 */
	bool clockTick(SST::Cycle_t cycle);

	// SST ELI Registration
	SST_ELI_REGISTER_COMPONENT(ACALSimVirtIODeviceComponent, "acalsim", "VirtIODevice",
	                           SST_ELI_ELEMENT_VERSION(1, 0, 0), "ACALSim VirtIO Device for Linux Integration",
	                           COMPONENT_CATEGORY_UNCATEGORIZED)

	// Parameter documentation
	SST_ELI_DOCUMENT_PARAMS({"socket_path", "Unix domain socket path", "/tmp/qemu-sst-linux.sock"},
	                        {"device_id", "Device identifier", "0"}, {"verbose", "Verbosity level (0-3)", "1"},
	                        {"clock", "Clock frequency", "1GHz"})

	// Statistics
	SST_ELI_DOCUMENT_STATISTICS({"total_requests", "Total requests processed", "requests", 1},
	                            {"noop_requests", "NOOP requests", "requests", 1},
	                            {"echo_requests", "ECHO requests", "requests", 1},
	                            {"compute_requests", "COMPUTE requests", "requests", 1})

private:
	// Socket management
	void initSocket();
	void cleanupSocket();
	void checkForConnections();
	void handleSocketData();

	// Request processing
	void processRequest(const uint8_t* data, size_t len);
	void sendResponse(const uint8_t* data, size_t len);

	// Configuration
	std::string socket_path_;
	uint32_t    device_id_;
	int         verbose_;

	// Socket state
	int  server_fd_;
	int  client_fd_;
	bool client_connected_;

	// SST infrastructure
	SST::Output                                        out_;
	SST::Clock::Handler<ACALSimVirtIODeviceComponent>* clock_handler_;

	// Statistics
	SST::Cycle_t current_cycle_;
	uint64_t     total_requests_;
	uint64_t     noop_requests_;
	uint64_t     echo_requests_;
	uint64_t     compute_requests_;
};

}  // namespace VirtIO
}  // namespace ACALSim

#endif  // ACALSIM_VIRTIO_DEVICE_COMPONENT_HH
