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

#ifndef __ACALSIM_COMPUTE_DEVICE_COMPONENT_HH__
#define __ACALSIM_COMPUTE_DEVICE_COMPONENT_HH__

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include "ACALSimDeviceComponent.hh"  // For event types

namespace ACALSim {
namespace QEMUIntegration {

/**
 * @brief Inter-device communication event
 *
 * This event is sent between devices for peer-to-peer communication.
 */
class DeviceMessageEvent : public SST::Event {
public:
	/**
	 * @brief Message types for inter-device communication
	 */
	enum MessageType : uint8_t {
		DATA_REQUEST     = 0,  ///< Request data from peer
		DATA_RESPONSE    = 1,  ///< Respond with data
		COMPUTE_REQUEST  = 2,  ///< Request computation from peer
		COMPUTE_RESPONSE = 3   ///< Respond with computation result
	};

	/**
	 * @brief Default constructor for serialization
	 */
	DeviceMessageEvent() : type_(DATA_REQUEST), data_(0), param_(0) {}

	/**
	 * @brief Constructor for device message
	 * @param type Message type
	 * @param data Data payload
	 * @param param Additional parameter
	 */
	DeviceMessageEvent(MessageType type, uint32_t data, uint32_t param) : type_(type), data_(data), param_(param) {}

	// Getters
	MessageType getType() const { return type_; }
	uint32_t    getData() const { return data_; }
	uint32_t    getParam() const { return param_; }

	// SST Event interface
	void serialize_order(SST::Core::Serialization::serializer& ser) override {
		Event::serialize_order(ser);
		ser & type_;
		ser & data_;
		ser & param_;
	}

	ImplementSerializable(ACALSim::QEMUIntegration::DeviceMessageEvent);

private:
	MessageType type_;   ///< Message type
	uint32_t    data_;   ///< Data payload
	uint32_t    param_;  ///< Additional parameter
};

/**
 * @brief ACALSim Compute Device SST Component
 *
 * This component models a computational accelerator device that:
 * - Receives memory transaction requests from QEMU via SST Link
 * - Performs arithmetic/logic operations
 * - Can communicate with peer devices (e.g., echo device)
 * - Sends responses back to QEMU
 *
 * Device Register Map (4KB at base address, typically 0x10300000):
 * - 0x00: OPERAND_A (W)    - First operand
 * - 0x04: OPERAND_B (W)    - Second operand
 * - 0x08: OPERATION (W)    - Operation code (0=ADD, 1=SUB, 2=MUL, 3=DIV)
 * - 0x0C: RESULT (R)       - Computation result
 * - 0x10: STATUS (R)       - Device status (bit 0=busy, bit 1=ready, bit 2=error)
 * - 0x14: CONTROL (RW)     - Control register (bit 0=reset, bit 1=trigger)
 * - 0x18: PEER_DATA_OUT (W) - Send data to peer device
 * - 0x1C: PEER_DATA_IN (R)  - Receive data from peer device
 */
class ACALSimComputeDeviceComponent : public SST::Component {
public:
	/**
	 * @brief SST ELI registration
	 */
	SST_ELI_REGISTER_COMPONENT(ACALSimComputeDeviceComponent, "acalsim", "ComputeDevice",
	                           SST_ELI_ELEMENT_VERSION(1, 0, 0),
	                           "ACALSim computational device for QEMU integration with peer communication",
	                           COMPONENT_CATEGORY_UNCATEGORIZED)

	/**
	 * @brief Parameter documentation
	 */
	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency", "1GHz"}, {"base_addr", "Device base address", "0x10300000"},
	                        {"size", "Device memory size", "4096"}, {"verbose", "Verbosity level", "1"},
	                        {"compute_latency", "Computation latency in cycles", "100"})

	/**
	 * @brief Port documentation
	 */
	SST_ELI_DOCUMENT_PORTS({"cpu_port", "Port for CPU (QEMU) communication", {"ACALSim.MemoryTransaction"}},
	                       {"peer_port", "Port for peer device communication", {"ACALSim.DeviceMessage"}})

	/**
	 * @brief Constructor
	 * @param id Component ID
	 * @param params Component parameters
	 */
	ACALSimComputeDeviceComponent(SST::ComponentId_t id, SST::Params& params);

	/**
	 * @brief Destructor
	 */
	~ACALSimComputeDeviceComponent() override;

	// SST Component interface
	void setup() override;
	void finish() override;

	/**
	 * @brief Clock tick handler
	 * @param cycle Current cycle
	 * @return true to continue, false to stop
	 */
	bool clockTick(SST::Cycle_t cycle);

	/**
	 * @brief Handle incoming memory transaction from QEMU
	 * @param ev Memory transaction event
	 */
	void handleMemoryTransaction(SST::Event* ev);

	/**
	 * @brief Handle incoming message from peer device
	 * @param ev Device message event
	 */
	void handlePeerMessage(SST::Event* ev);

private:
	// Device registers (offset from base address)
	static constexpr uint64_t REG_OPERAND_A     = 0x00;  ///< Operand A
	static constexpr uint64_t REG_OPERAND_B     = 0x04;  ///< Operand B
	static constexpr uint64_t REG_OPERATION     = 0x08;  ///< Operation code
	static constexpr uint64_t REG_RESULT        = 0x0C;  ///< Computation result
	static constexpr uint64_t REG_STATUS        = 0x10;  ///< Status register
	static constexpr uint64_t REG_CONTROL       = 0x14;  ///< Control register
	static constexpr uint64_t REG_PEER_DATA_OUT = 0x18;  ///< Peer output data
	static constexpr uint64_t REG_PEER_DATA_IN  = 0x1C;  ///< Peer input data

	// Operation codes
	enum Operation : uint32_t {
		OP_ADD = 0,  ///< Addition
		OP_SUB = 1,  ///< Subtraction
		OP_MUL = 2,  ///< Multiplication
		OP_DIV = 3   ///< Division
	};

	// Status register bits
	static constexpr uint32_t STATUS_BUSY       = 0x01;  ///< Bit 0: Device busy
	static constexpr uint32_t STATUS_READY      = 0x02;  ///< Bit 1: Result ready
	static constexpr uint32_t STATUS_ERROR      = 0x04;  ///< Bit 2: Error (e.g., div by zero)
	static constexpr uint32_t STATUS_PEER_READY = 0x08;  ///< Bit 3: Peer data ready

	// Control register bits
	static constexpr uint32_t CONTROL_RESET   = 0x01;  ///< Bit 0: Reset device
	static constexpr uint32_t CONTROL_TRIGGER = 0x02;  ///< Bit 1: Trigger computation

	/**
	 * @brief Process load operation
	 * @param addr Address offset from base
	 * @param size Transaction size
	 * @param req_id Request ID
	 */
	void processLoad(uint64_t addr, uint32_t size, uint64_t req_id);

	/**
	 * @brief Process store operation
	 * @param addr Address offset from base
	 * @param data Data to write
	 * @param size Transaction size
	 * @param req_id Request ID
	 */
	void processStore(uint64_t addr, uint32_t data, uint32_t size, uint64_t req_id);

	/**
	 * @brief Read device register
	 * @param offset Register offset
	 * @return Register value
	 */
	uint32_t readRegister(uint64_t offset);

	/**
	 * @brief Write device register
	 * @param offset Register offset
	 * @param value Value to write
	 */
	void writeRegister(uint64_t offset, uint32_t value);

	/**
	 * @brief Complete pending computation
	 */
	void completeComputation();

	/**
	 * @brief Trigger computation operation
	 */
	void triggerComputation();

	/**
	 * @brief Reset device to initial state
	 */
	void resetDevice();

	/**
	 * @brief Send data to peer device
	 * @param data Data to send
	 */
	void sendToPeer(uint32_t data);

	// SST infrastructure
	SST::Output         out_;            ///< Output handler
	SST::Link*          cpu_link_;       ///< Link to CPU (QEMU)
	SST::Link*          peer_link_;      ///< Link to peer device
	SST::TimeConverter* tc_;             ///< Time converter
	SST::Cycle_t        current_cycle_;  ///< Current cycle

	// Device configuration
	uint64_t base_addr_;        ///< Base address in memory map
	uint64_t size_;             ///< Device memory size
	uint64_t compute_latency_;  ///< Computation latency

	// Device state - registers
	uint32_t operand_a_;      ///< OPERAND_A register
	uint32_t operand_b_;      ///< OPERAND_B register
	uint32_t operation_;      ///< OPERATION register
	uint32_t result_;         ///< RESULT register
	uint32_t status_;         ///< STATUS register
	uint32_t control_;        ///< CONTROL register
	uint32_t peer_data_out_;  ///< PEER_DATA_OUT register
	uint32_t peer_data_in_;   ///< PEER_DATA_IN register

	// Computation state
	bool     compute_pending_;         ///< Computation in progress
	uint64_t compute_complete_cycle_;  ///< Cycle when computation completes
	uint64_t pending_req_id_;          ///< Request ID of pending operation

	// Statistics
	uint64_t total_loads_;         ///< Total load operations
	uint64_t total_stores_;        ///< Total store operations
	uint64_t total_computations_;  ///< Total computations performed
	uint64_t total_peer_msgs_;     ///< Total peer messages sent
};

}  // namespace QEMUIntegration
}  // namespace ACALSim

#endif  // __ACALSIM_COMPUTE_DEVICE_COMPONENT_HH__
