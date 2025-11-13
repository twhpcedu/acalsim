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

#ifndef __QEMU_COMPONENT_HH__
#define __QEMU_COMPONENT_HH__

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <map>
#include <queue>

namespace ACALSim {
namespace QEMUIntegration {

// Forward declarations - these match the device component events
enum class TransactionType : uint8_t { LOAD = 0, STORE = 1 };

/**
 * @brief Memory transaction event sent from QEMU to Device
 */
class MemoryTransactionEvent : public SST::Event {
public:
	MemoryTransactionEvent() : type_(TransactionType::LOAD), address_(0), data_(0), size_(0), req_id_(0) {}

	MemoryTransactionEvent(TransactionType type, uint64_t addr, uint32_t data, uint32_t size, uint64_t req_id)
	    : type_(type), address_(addr), data_(data), size_(size), req_id_(req_id) {}

	TransactionType getType() const { return type_; }
	uint64_t        getAddress() const { return address_; }
	uint32_t        getData() const { return data_; }
	uint32_t        getSize() const { return size_; }
	uint64_t        getReqId() const { return req_id_; }

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
	TransactionType type_;
	uint64_t        address_;
	uint32_t        data_;
	uint32_t        size_;
	uint64_t        req_id_;
};

/**
 * @brief Memory response event sent from Device back to QEMU
 */
class MemoryResponseEvent : public SST::Event {
public:
	MemoryResponseEvent() : req_id_(0), data_(0), success_(false) {}

	MemoryResponseEvent(uint64_t req_id, uint32_t data, bool success)
	    : req_id_(req_id), data_(data), success_(success) {}

	uint64_t getReqId() const { return req_id_; }
	uint32_t getData() const { return data_; }
	bool     getSuccess() const { return success_; }

	void serialize_order(SST::Core::Serialization::serializer& ser) override {
		Event::serialize_order(ser);
		ser & req_id_;
		ser & data_;
		ser & success_;
	}

	ImplementSerializable(ACALSim::QEMUIntegration::MemoryResponseEvent);

private:
	uint64_t req_id_;
	uint32_t data_;
	bool     success_;
};

/**
 * @brief Pending memory transaction waiting for response
 */
struct PendingTransaction {
	uint64_t        req_id;       ///< Request ID
	TransactionType type;         ///< LOAD or STORE
	uint64_t        address;      ///< Memory address
	uint32_t        data;         ///< Data (for stores)
	SST::Cycle_t    issue_cycle;  ///< Cycle when issued
};

/**
 * @brief QEMU SST Component
 *
 * This component simulates QEMU RISC-V emulator behavior for testing
 * distributed SST simulation. In a full implementation, this would
 * integrate with actual QEMU, but for this simple example, it simulates
 * a test program that:
 * 1. Writes test patterns to device
 * 2. Reads back echoed data
 * 3. Verifies correctness
 *
 * The component demonstrates:
 * - Sending MemoryTransactionEvent to device via SST Link
 * - Receiving MemoryResponseEvent from device
 * - Handling request/response matching
 * - Test program simulation
 */
class QEMUComponent : public SST::Component {
public:
	/**
	 * @brief SST ELI registration
	 */
	SST_ELI_REGISTER_COMPONENT(QEMUComponent, "qemu", "RISCV", SST_ELI_ELEMENT_VERSION(1, 0, 0),
	                           "QEMU RISC-V emulator wrapper for distributed simulation",
	                           COMPONENT_CATEGORY_UNCATEGORIZED)

	/**
	 * @brief Parameter documentation
	 */
	SST_ELI_DOCUMENT_PARAMS({{"clock", "Clock frequency", "1GHz"},
	                         {"device_base", "Device base address", "0x10000000"},
	                         {"device_size", "Device memory size", "4096"},
	                         {"verbose", "Verbosity level", "1"},
	                         {"test_pattern", "Test data pattern to write", "0xDEADBEEF"},
	                         {"num_iterations", "Number of test iterations", "5"}})

	/**
	 * @brief Port documentation
	 */
	SST_ELI_DOCUMENT_PORTS({{"device_port", "Port for device communication", {"ACALSim.MemoryTransaction"}}})

	/**
	 * @brief Constructor
	 * @param id Component ID
	 * @param params Component parameters
	 */
	QEMUComponent(SST::ComponentId_t id, SST::Params& params);

	/**
	 * @brief Destructor
	 */
	~QEMUComponent() override;

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
	 * @brief Handle incoming memory response from device
	 * @param ev Memory response event
	 */
	void handleMemoryResponse(SST::Event* ev);

private:
	/**
	 * @brief Test program states
	 */
	enum class TestState {
		IDLE,         ///< Initial state
		WRITE_DATA,   ///< Writing to device DATA_IN
		WAIT_BUSY,    ///< Waiting for device to process
		READ_STATUS,  ///< Reading device STATUS
		READ_DATA,    ///< Reading from device DATA_OUT
		VERIFY,       ///< Verifying result
		DONE          ///< Test complete
	};

	/**
	 * @brief Send load transaction to device
	 * @param addr Address to read
	 * @param size Transaction size (1, 2, or 4 bytes)
	 * @return Request ID
	 */
	uint64_t sendLoad(uint64_t addr, uint32_t size);

	/**
	 * @brief Send store transaction to device
	 * @param addr Address to write
	 * @param data Data to write
	 * @param size Transaction size (1, 2, or 4 bytes)
	 * @return Request ID
	 */
	uint64_t sendStore(uint64_t addr, uint32_t data, uint32_t size);

	/**
	 * @brief Process pending responses
	 */
	void processResponses();

	/**
	 * @brief Run test program state machine
	 */
	void runTestProgram();

	/**
	 * @brief Check if device region contains address
	 * @param addr Address to check
	 * @return true if in device region
	 */
	bool isDeviceAddress(uint64_t addr) const;

	// SST infrastructure
	SST::Output         out_;            ///< Output handler
	SST::Link*          device_link_;    ///< Link to device
	SST::TimeConverter* tc_;             ///< Time converter
	SST::Cycle_t        current_cycle_;  ///< Current cycle

	// Device configuration
	uint64_t device_base_;     ///< Device base address
	uint64_t device_size_;     ///< Device size
	uint32_t test_pattern_;    ///< Test data pattern
	uint32_t num_iterations_;  ///< Number of test iterations

	// Transaction management
	uint64_t                               next_req_id_;           ///< Next request ID
	std::map<uint64_t, PendingTransaction> pending_transactions_;  ///< Pending requests
	std::queue<MemoryResponseEvent*>       response_queue_;        ///< Received responses

	// Test program state
	TestState test_state_;        ///< Current test state
	uint32_t  iteration_;         ///< Current iteration
	uint64_t  write_req_id_;      ///< Request ID of write operation
	uint64_t  read_req_id_;       ///< Request ID of read operation
	uint32_t  read_data_;         ///< Data read from device
	uint32_t  status_data_;       ///< Status register value
	bool      waiting_response_;  ///< Waiting for response

	// Statistics
	uint64_t total_loads_;      ///< Total load operations
	uint64_t total_stores_;     ///< Total store operations
	uint64_t total_successes_;  ///< Total successful tests
	uint64_t total_failures_;   ///< Total failed tests

	// Device register offsets
	static constexpr uint64_t REG_DATA_IN  = 0x00;
	static constexpr uint64_t REG_DATA_OUT = 0x04;
	static constexpr uint64_t REG_STATUS   = 0x08;
	static constexpr uint64_t REG_CONTROL  = 0x0C;

	// Status register bits
	static constexpr uint32_t STATUS_BUSY       = 0x01;
	static constexpr uint32_t STATUS_DATA_READY = 0x02;
};

}  // namespace QEMUIntegration
}  // namespace ACALSim

#endif  // __QEMU_COMPONENT_HH__
