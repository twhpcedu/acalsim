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

#include "ACALSimComputeDeviceComponent.hh"
#include <cstdio>

using namespace ACALSim::QEMUIntegration;

//
// Constructor
//
ACALSimComputeDeviceComponent::ACALSimComputeDeviceComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id) {

	// Get parameters
	base_addr_ = params.find<uint64_t>("base_addr", 0x10300000);
	size_ = params.find<uint64_t>("size", 4096);
	compute_latency_ = params.find<uint64_t>("compute_latency", 100);
	int verbose = params.find<int>("verbose", 1);

	// Initialize output
	out_.init("ComputeDevice[@f:@l:@p] ", verbose, 0, SST::Output::STDOUT);

	// Configure CPU link
	cpu_link_ = configureLink("cpu_port", new SST::Event::Handler<ACALSimComputeDeviceComponent>(
	                                          this, &ACALSimComputeDeviceComponent::handleMemoryTransaction));
	if (!cpu_link_) {
		out_.fatal(CALL_INFO, -1, "Failed to configure cpu_port\n");
	}

	// Configure peer link (optional - may not be connected)
	peer_link_ = configureLink("peer_port", new SST::Event::Handler<ACALSimComputeDeviceComponent>(
	                                            this, &ACALSimComputeDeviceComponent::handlePeerMessage));

	// Register clock
	std::string clock_freq = params.find<std::string>("clock", "1GHz");
	tc_ = registerClock(clock_freq, new SST::Clock::Handler<ACALSimComputeDeviceComponent>(
	                                     this, &ACALSimComputeDeviceComponent::clockTick));

	// Initialize device state
	resetDevice();

	// Initialize statistics
	total_loads_ = 0;
	total_stores_ = 0;
	total_computations_ = 0;
	total_peer_msgs_ = 0;

	out_.verbose(CALL_INFO, 1, 0, "ComputeDevice initialized at base=0x%lx size=%lu\n", base_addr_, size_);
}

//
// Destructor
//
ACALSimComputeDeviceComponent::~ACALSimComputeDeviceComponent() {
	// Nothing to do
}

//
// Setup
//
void ACALSimComputeDeviceComponent::setup() {
	out_.verbose(CALL_INFO, 1, 0, "Setting up ComputeDevice\n");
}

//
// Finish
//
void ACALSimComputeDeviceComponent::finish() {
	out_.verbose(CALL_INFO, 1, 0, "Finishing ComputeDevice\n");
	out_.verbose(CALL_INFO, 1, 0, "Statistics:\n");
	out_.verbose(CALL_INFO, 1, 0, "  Total Loads:        %lu\n", total_loads_);
	out_.verbose(CALL_INFO, 1, 0, "  Total Stores:       %lu\n", total_stores_);
	out_.verbose(CALL_INFO, 1, 0, "  Total Computations: %lu\n", total_computations_);
	out_.verbose(CALL_INFO, 1, 0, "  Total Peer Messages:%lu\n", total_peer_msgs_);
}

//
// Clock tick
//
bool ACALSimComputeDeviceComponent::clockTick(SST::Cycle_t cycle) {
	current_cycle_ = cycle;

	// Check if computation is complete
	if (compute_pending_ && cycle >= compute_complete_cycle_) {
		completeComputation();
	}

	return false;  // Continue simulation
}

//
// Handle memory transaction from QEMU
//
void ACALSimComputeDeviceComponent::handleMemoryTransaction(SST::Event* ev) {
	auto* trans = dynamic_cast<MemoryTransactionEvent*>(ev);
	if (!trans) {
		out_.fatal(CALL_INFO, -1, "Received invalid event type\n");
	}

	// Get transaction details
	TransactionType type = trans->getType();
	uint64_t        addr = trans->getAddress();
	uint32_t        data = trans->getData();
	uint32_t        size = trans->getSize();
	uint64_t        req_id = trans->getReqId();

	// Calculate offset from base address
	if (addr < base_addr_ || addr >= base_addr_ + size_) {
		out_.fatal(CALL_INFO, -1, "Address 0x%lx out of range [0x%lx, 0x%lx)\n", addr, base_addr_,
		           base_addr_ + size_);
	}
	uint64_t offset = addr - base_addr_;

	if (type == TransactionType::LOAD) {
		processLoad(offset, size, req_id);
	} else {
		processStore(offset, data, size, req_id);
	}

	delete ev;
}

//
// Handle peer message
//
void ACALSimComputeDeviceComponent::handlePeerMessage(SST::Event* ev) {
	auto* msg = dynamic_cast<DeviceMessageEvent*>(ev);
	if (!msg) {
		out_.verbose(CALL_INFO, 1, 0, "Received invalid peer message type\n");
		delete ev;
		return;
	}

	DeviceMessageEvent::MessageType type = msg->getType();
	uint32_t                        data = msg->getData();

	out_.verbose(CALL_INFO, 2, 0, "Received peer message: type=%u data=0x%08x\n", type, data);

	switch (type) {
	case DeviceMessageEvent::DATA_REQUEST:
		// Peer is requesting data - send our current result
		if (peer_link_) {
			peer_link_->send(new DeviceMessageEvent(DeviceMessageEvent::DATA_RESPONSE, result_, 0));
			out_.verbose(CALL_INFO, 2, 0, "Sent result 0x%08x to peer\n", result_);
		}
		break;

	case DeviceMessageEvent::DATA_RESPONSE:
		// Peer sent us data - store in peer_data_in
		peer_data_in_ = data;
		status_ |= STATUS_PEER_READY;
		out_.verbose(CALL_INFO, 2, 0, "Received data 0x%08x from peer\n", data);
		break;

	case DeviceMessageEvent::COMPUTE_REQUEST:
		// Peer is requesting computation - perform operation and send result
		// For simplicity, just double the value
		if (peer_link_) {
			uint32_t result = data * 2;
			peer_link_->send(new DeviceMessageEvent(DeviceMessageEvent::COMPUTE_RESPONSE, result, 0));
			out_.verbose(CALL_INFO, 2, 0, "Computed result 0x%08x for peer\n", result);
		}
		break;

	case DeviceMessageEvent::COMPUTE_RESPONSE:
		// Peer sent computation result
		peer_data_in_ = data;
		status_ |= STATUS_PEER_READY;
		out_.verbose(CALL_INFO, 2, 0, "Received computation result 0x%08x from peer\n", data);
		break;

	default:
		out_.verbose(CALL_INFO, 1, 0, "Unknown peer message type: %u\n", type);
		break;
	}

	delete ev;
}

//
// Process load operation
//
void ACALSimComputeDeviceComponent::processLoad(uint64_t addr, uint32_t size, uint64_t req_id) {
	total_loads_++;

	uint32_t value = readRegister(addr);

	out_.verbose(CALL_INFO, 2, 0, "LOAD: addr=0x%lx+0x%lx size=%u value=0x%08x\n", base_addr_, addr, size, value);

	// Send response immediately
	auto* resp = new MemoryResponseEvent(req_id, value, true);
	cpu_link_->send(resp);
}

//
// Process store operation
//
void ACALSimComputeDeviceComponent::processStore(uint64_t addr, uint32_t data, uint32_t size, uint64_t req_id) {
	total_stores_++;

	out_.verbose(CALL_INFO, 2, 0, "STORE: addr=0x%lx+0x%lx size=%u data=0x%08x\n", base_addr_, addr, size, data);

	writeRegister(addr, data);

	// Send acknowledgment
	auto* resp = new MemoryResponseEvent(req_id, 0, true);
	cpu_link_->send(resp);
}

//
// Read device register
//
uint32_t ACALSimComputeDeviceComponent::readRegister(uint64_t offset) {
	switch (offset) {
	case REG_OPERAND_A:
		return operand_a_;
	case REG_OPERAND_B:
		return operand_b_;
	case REG_OPERATION:
		return operation_;
	case REG_RESULT:
		return result_;
	case REG_STATUS:
		return status_;
	case REG_CONTROL:
		return control_;
	case REG_PEER_DATA_OUT:
		return peer_data_out_;
	case REG_PEER_DATA_IN:
		// Clear PEER_READY bit on read
		status_ &= ~STATUS_PEER_READY;
		return peer_data_in_;
	default:
		out_.verbose(CALL_INFO, 1, 0, "Read from unknown register offset 0x%lx\n", offset);
		return 0;
	}
}

//
// Write device register
//
void ACALSimComputeDeviceComponent::writeRegister(uint64_t offset, uint32_t value) {
	switch (offset) {
	case REG_OPERAND_A:
		operand_a_ = value;
		break;

	case REG_OPERAND_B:
		operand_b_ = value;
		break;

	case REG_OPERATION:
		operation_ = value;
		break;

	case REG_CONTROL:
		control_ = value;
		if (value & CONTROL_RESET) {
			resetDevice();
		}
		if (value & CONTROL_TRIGGER) {
			triggerComputation();
		}
		break;

	case REG_PEER_DATA_OUT:
		peer_data_out_ = value;
		sendToPeer(value);
		break;

	default:
		out_.verbose(CALL_INFO, 1, 0, "Write to unknown/read-only register offset 0x%lx\n", offset);
		break;
	}
}

//
// Trigger computation
//
void ACALSimComputeDeviceComponent::triggerComputation() {
	if (compute_pending_) {
		out_.verbose(CALL_INFO, 1, 0, "Computation already in progress\n");
		return;
	}

	out_.verbose(CALL_INFO, 2, 0, "Triggering computation: op=%u A=0x%08x B=0x%08x\n", operation_, operand_a_,
	             operand_b_);

	// Set busy flag, clear ready and error flags
	status_ = STATUS_BUSY;

	// Schedule computation completion
	compute_pending_ = true;
	compute_complete_cycle_ = current_cycle_ + compute_latency_;

	total_computations_++;
}

//
// Complete computation
//
void ACALSimComputeDeviceComponent::completeComputation() {
	compute_pending_ = false;

	// Perform operation
	bool error = false;
	switch (operation_) {
	case OP_ADD:
		result_ = operand_a_ + operand_b_;
		break;
	case OP_SUB:
		result_ = operand_a_ - operand_b_;
		break;
	case OP_MUL:
		result_ = operand_a_ * operand_b_;
		break;
	case OP_DIV:
		if (operand_b_ == 0) {
			result_ = 0;
			error = true;
		} else {
			result_ = operand_a_ / operand_b_;
		}
		break;
	default:
		result_ = 0;
		error = true;
		break;
	}

	// Update status
	if (error) {
		status_ = STATUS_ERROR;
	} else {
		status_ = STATUS_READY;
	}

	out_.verbose(CALL_INFO, 2, 0, "Computation complete: result=0x%08x status=0x%02x\n", result_, status_);
}

//
// Reset device
//
void ACALSimComputeDeviceComponent::resetDevice() {
	operand_a_ = 0;
	operand_b_ = 0;
	operation_ = OP_ADD;
	result_ = 0;
	status_ = 0;
	control_ = 0;
	peer_data_out_ = 0;
	peer_data_in_ = 0;
	compute_pending_ = false;
	compute_complete_cycle_ = 0;
	pending_req_id_ = 0;

	out_.verbose(CALL_INFO, 2, 0, "Device reset\n");
}

//
// Send data to peer device
//
void ACALSimComputeDeviceComponent::sendToPeer(uint32_t data) {
	if (!peer_link_) {
		out_.verbose(CALL_INFO, 1, 0, "No peer link configured\n");
		return;
	}

	// Send data request to peer
	peer_link_->send(new DeviceMessageEvent(DeviceMessageEvent::DATA_REQUEST, data, 0));
	total_peer_msgs_++;

	out_.verbose(CALL_INFO, 2, 0, "Sent data 0x%08x to peer\n", data);
}
