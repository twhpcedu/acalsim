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

#ifndef __ACALSIM_DEVICE_COMPONENT_HH__
#define __ACALSIM_DEVICE_COMPONENT_HH__

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

namespace ACALSim {
namespace QEMUIntegration {

/**
 * @brief Memory transaction types
 */
enum class TransactionType : uint8_t {
	LOAD  = 0,  ///< Read from device
	STORE = 1   ///< Write to device
};

/**
 * @brief Memory transaction event sent from QEMU to Device
 *
 * This event represents a load or store operation from QEMU
 * to the memory-mapped device region.
 */
class MemoryTransactionEvent : public SST::Event {
public:
	/**
	 * @brief Default constructor for serialization
	 */
	MemoryTransactionEvent() : type_(TransactionType::LOAD), address_(0), data_(0), size_(0), req_id_(0) {}

	/**
	 * @brief Constructor for memory transaction
	 * @param type Transaction type (LOAD or STORE)
	 * @param addr Memory address
	 * @param data Data value (for STORE operations)
	 * @param size Transaction size in bytes (1, 2, or 4)
	 * @param req_id Unique request ID for matching responses
	 */
	MemoryTransactionEvent(TransactionType type, uint64_t addr, uint32_t data, uint32_t size, uint64_t req_id)
	    : type_(type), address_(addr), data_(data), size_(size), req_id_(req_id) {}

	// Getters
	TransactionType getType() const { return type_; }
	uint64_t        getAddress() const { return address_; }
	uint32_t        getData() const { return data_; }
	uint32_t        getSize() const { return size_; }
	uint64_t        getReqId() const { return req_id_; }

	// SST Event interface
	void serialize_order(SST::Core::Serialization::serializer& ser) override {
		Event::serialize_order(ser);
		ser & type_;
		ser & address_;
		ser & data_;
		ser & size_;
		ser & req_id_;
	}

	ImplementSerializable(ACALSim::QEMUIntegration::MemoryTransactionEvent);

private:
	TransactionType type_;     ///< LOAD or STORE
	uint64_t        address_;  ///< Memory address
	uint32_t        data_;     ///< Data (for stores)
	uint32_t        size_;     ///< Size in bytes
	uint64_t        req_id_;   ///< Request ID
};

/**
 * @brief Memory response event sent from Device back to QEMU
 *
 * This event contains the response to a memory transaction.
 */
class MemoryResponseEvent : public SST::Event {
public:
	/**
	 * @brief Default constructor for serialization
	 */
	MemoryResponseEvent() : req_id_(0), data_(0), success_(false) {}

	/**
	 * @brief Constructor for memory response
	 * @param req_id Request ID matching the original transaction
	 * @param data Response data (for LOAD operations)
	 * @param success Transaction success status
	 */
	MemoryResponseEvent(uint64_t req_id, uint32_t data, bool success)
	    : req_id_(req_id), data_(data), success_(success) {}

	// Getters
	uint64_t getReqId() const { return req_id_; }
	uint32_t getData() const { return data_; }
	bool     getSuccess() const { return success_; }

	// SST Event interface
	void serialize_order(SST::Core::Serialization::serializer& ser) override {
		Event::serialize_order(ser);
		ser & req_id_;
		ser & data_;
		ser & success_;
	}

	ImplementSerializable(ACALSim::QEMUIntegration::MemoryResponseEvent);

private:
	uint64_t req_id_;   ///< Request ID
	uint32_t data_;     ///< Response data
	bool     success_;  ///< Success status
};

/**
 * @brief ACALSim Device SST Component
 *
 * This component models a simple memory-mapped echo device that:
 * - Receives memory transaction requests from QEMU via SST Link
 * - Processes transactions in cycle-accurate manner
 * - Sends responses back to QEMU
 *
 * Device Register Map (4KB at base address):
 * - 0x00: DATA_IN (W)  - Write data to device
 * - 0x04: DATA_OUT (R) - Read echoed data
 * - 0x08: STATUS (R)   - Device status (bit 0=busy, bit 1=ready)
 * - 0x0C: CONTROL (RW) - Control register (bit 0=reset)
 */
class ACALSimDeviceComponent : public SST::Component {
public:
	/**
	 * @brief SST ELI registration
	 */
	SST_ELI_REGISTER_COMPONENT(ACALSimDeviceComponent, "acalsim", "QEMUDevice", SST_ELI_ELEMENT_VERSION(1, 0, 0),
	                           "ACALSim memory-mapped device for QEMU integration", COMPONENT_CATEGORY_UNCATEGORIZED)

	/**
	 * @brief Parameter documentation
	 */
	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency", "1GHz"}, {"base_addr", "Device base address", "0x10000000"},
	                        {"size", "Device memory size", "4096"}, {"verbose", "Verbosity level", "1"},
	                        {"echo_latency", "Echo operation latency in cycles", "10"})

	/**
	 * @brief Port documentation
	 */
	SST_ELI_DOCUMENT_PORTS({"cpu_port", "Port for CPU (QEMU) communication", {"ACALSim.MemoryTransaction"}})

	/**
	 * @brief Constructor
	 * @param id Component ID
	 * @param params Component parameters
	 */
	ACALSimDeviceComponent(SST::ComponentId_t id, SST::Params& params);

	/**
	 * @brief Destructor
	 */
	~ACALSimDeviceComponent() override;

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

private:
	// Device registers (offset from base address)
	static constexpr uint64_t REG_DATA_IN  = 0x00;  ///< Data input register
	static constexpr uint64_t REG_DATA_OUT = 0x04;  ///< Data output register
	static constexpr uint64_t REG_STATUS   = 0x08;  ///< Status register
	static constexpr uint64_t REG_CONTROL  = 0x0C;  ///< Control register

	// Status register bits
	static constexpr uint32_t STATUS_BUSY       = 0x01;  ///< Bit 0: Device busy
	static constexpr uint32_t STATUS_DATA_READY = 0x02;  ///< Bit 1: Data ready

	// Control register bits
	static constexpr uint32_t CONTROL_RESET = 0x01;  ///< Bit 0: Reset device

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
	 * @brief Complete pending echo operation
	 */
	void completeEcho();

	/**
	 * @brief Reset device to initial state
	 */
	void resetDevice();

	// SST infrastructure
	SST::Output         out_;            ///< Output handler
	SST::Link*          cpu_link_;       ///< Link to CPU (QEMU)
	SST::TimeConverter* tc_;             ///< Time converter
	SST::Cycle_t        current_cycle_;  ///< Current cycle

	// Device configuration
	uint64_t base_addr_;     ///< Base address in memory map
	uint64_t size_;          ///< Device memory size
	uint64_t echo_latency_;  ///< Echo operation latency

	// Device state
	uint32_t data_in_;   ///< DATA_IN register value
	uint32_t data_out_;  ///< DATA_OUT register value
	uint32_t status_;    ///< STATUS register value
	uint32_t control_;   ///< CONTROL register value

	// Echo operation state
	bool     echo_pending_;         ///< Echo operation in progress
	uint64_t echo_complete_cycle_;  ///< Cycle when echo completes
	uint64_t pending_req_id_;       ///< Request ID of pending operation

	// Statistics
	uint64_t total_loads_;   ///< Total load operations
	uint64_t total_stores_;  ///< Total store operations
	uint64_t total_echos_;   ///< Total echo operations
};

}  // namespace QEMUIntegration
}  // namespace ACALSim

#endif  // __ACALSIM_DEVICE_COMPONENT_HH__
