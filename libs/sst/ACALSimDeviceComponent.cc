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

#include "sst/ACALSimDeviceComponent.hh"

#include <sst/core/sst_config.h>

using namespace ACALSim::QEMUIntegration;

ACALSimDeviceComponent::ACALSimDeviceComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id),
      current_cycle_(0),
      echo_pending_(false),
      echo_complete_cycle_(0),
      pending_req_id_(0),
      total_loads_(0),
      total_stores_(0),
      total_echos_(0) {
	// Initialize output
	int verbose = params.find<int>("verbose", 1);
	out_.init("ACALSimDevice[@p:@l]: ", verbose, 0, SST::Output::STDOUT);

	out_.verbose(CALL_INFO, 1, 0, "Initializing ACALSim Device Component\n");

	// Get parameters
	std::string clock_freq = params.find<std::string>("clock", "1GHz");
	base_addr_             = params.find<uint64_t>("base_addr", 0x10000000ULL);
	size_                  = params.find<uint64_t>("size", 4096);
	echo_latency_          = params.find<uint64_t>("echo_latency", 10);

	out_.verbose(CALL_INFO, 1, 0, "Configuration:\n");
	out_.verbose(CALL_INFO, 1, 0, "  Clock: %s\n", clock_freq.c_str());
	out_.verbose(CALL_INFO, 1, 0, "  Base Address: 0x%lx\n", base_addr_);
	out_.verbose(CALL_INFO, 1, 0, "  Size: %lu bytes\n", size_);
	out_.verbose(CALL_INFO, 1, 0, "  Echo Latency: %lu cycles\n", echo_latency_);

	// Initialize device state
	resetDevice();

	// Configure clock
	tc_ = registerClock(clock_freq,
	                    new SST::Clock::Handler<ACALSimDeviceComponent>(this, &ACALSimDeviceComponent::clockTick));

	// Configure link to CPU (QEMU)
	cpu_link_ = configureLink("cpu_port", new SST::Event::Handler<ACALSimDeviceComponent>(
	                                          this, &ACALSimDeviceComponent::handleMemoryTransaction));

	if (!cpu_link_) { out_.fatal(CALL_INFO, -1, "Error: Failed to configure cpu_port link\n"); }

	// Tell SST this component is not ready to end simulation yet
	registerAsPrimaryComponent();
	primaryComponentDoNotEndSim();

	out_.verbose(CALL_INFO, 1, 0, "Device initialized successfully\n");
}

ACALSimDeviceComponent::~ACALSimDeviceComponent() {
	// Cleanup if needed
}

void ACALSimDeviceComponent::setup() { out_.verbose(CALL_INFO, 1, 0, "Setup phase\n"); }

void ACALSimDeviceComponent::finish() {
	out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");
	out_.verbose(CALL_INFO, 1, 0, "Statistics:\n");
	out_.verbose(CALL_INFO, 1, 0, "  Total Loads:  %lu\n", total_loads_);
	out_.verbose(CALL_INFO, 1, 0, "  Total Stores: %lu\n", total_stores_);
	out_.verbose(CALL_INFO, 1, 0, "  Total Echos:  %lu\n", total_echos_);
}

bool ACALSimDeviceComponent::clockTick(SST::Cycle_t cycle) {
	current_cycle_ = cycle;

	out_.verbose(CALL_INFO, 2, 0, "[DEVICE CLOCK] Cycle %lu\n", cycle);

	// Check if echo operation completes this cycle
	if (echo_pending_ && current_cycle_ >= echo_complete_cycle_) { completeEcho(); }

	return false;  // Clock handler: false = continue ticking, true = done
}

void ACALSimDeviceComponent::handleMemoryTransaction(SST::Event* ev) {
	MemoryTransactionEvent* trans = dynamic_cast<MemoryTransactionEvent*>(ev);
	if (!trans) {
		out_.fatal(CALL_INFO, -1, "Error: Received invalid event type\n");
		return;
	}

	uint64_t addr   = trans->getAddress();
	uint32_t data   = trans->getData();
	uint32_t size   = trans->getSize();
	uint64_t req_id = trans->getReqId();

	// Check if address is in device range
	if (addr < base_addr_ || addr >= (base_addr_ + size_)) {
		out_.verbose(CALL_INFO, 2, 0, "Error: Address 0x%lx out of range [0x%lx - 0x%lx]\n", addr, base_addr_,
		             base_addr_ + size_ - 1);
		auto* resp = new MemoryResponseEvent(req_id, 0, false);
		cpu_link_->send(resp);
		delete trans;
		return;
	}

	// Convert to offset from base
	uint64_t offset = addr - base_addr_;

	// Process transaction
	if (trans->getType() == TransactionType::LOAD) {
		out_.verbose(CALL_INFO, 2, 0, "LOAD: addr=0x%lx offset=0x%lx size=%u req_id=%lu\n", addr, offset, size, req_id);
		processLoad(offset, size, req_id);
	} else {  // STORE
		out_.verbose(CALL_INFO, 2, 0, "STORE: addr=0x%lx offset=0x%lx data=0x%x size=%u req_id=%lu\n", addr, offset,
		             data, size, req_id);
		processStore(offset, data, size, req_id);
	}

	delete trans;
}

void ACALSimDeviceComponent::processLoad(uint64_t addr, uint32_t size, uint64_t req_id) {
	total_loads_++;

	// Read from register
	uint32_t value = readRegister(addr);

	// Apply size mask
	uint32_t mask = 0xFFFFFFFF;
	if (size == 1)
		mask = 0xFF;
	else if (size == 2)
		mask = 0xFFFF;

	value &= mask;

	out_.verbose(CALL_INFO, 3, 0, "Read register offset=0x%lx value=0x%x\n", addr, value);

	// Send response immediately (no latency for reads in this simple model)
	auto* resp = new MemoryResponseEvent(req_id, value, true);
	cpu_link_->send(resp);
}

void ACALSimDeviceComponent::processStore(uint64_t addr, uint32_t data, uint32_t size, uint64_t req_id) {
	total_stores_++;

	// Apply size mask
	uint32_t mask = 0xFFFFFFFF;
	if (size == 1)
		mask = 0xFF;
	else if (size == 2)
		mask = 0xFFFF;

	data &= mask;

	out_.verbose(CALL_INFO, 3, 0, "Write register offset=0x%lx data=0x%x\n", addr, data);

	// Write to register
	writeRegister(addr, data);

	// Send response immediately
	auto* resp = new MemoryResponseEvent(req_id, 0, true);
	cpu_link_->send(resp);
}

uint32_t ACALSimDeviceComponent::readRegister(uint64_t offset) {
	switch (offset) {
		case REG_DATA_IN:
			out_.verbose(CALL_INFO, 3, 0, "Read DATA_IN (write-only, returning 0)\n");
			return 0;  // Write-only register

		case REG_DATA_OUT: out_.verbose(CALL_INFO, 3, 0, "Read DATA_OUT = 0x%x\n", data_out_); return data_out_;

		case REG_STATUS: out_.verbose(CALL_INFO, 3, 0, "Read STATUS = 0x%x\n", status_); return status_;

		case REG_CONTROL: out_.verbose(CALL_INFO, 3, 0, "Read CONTROL = 0x%x\n", control_); return control_;

		default: out_.verbose(CALL_INFO, 2, 0, "Read from undefined register offset 0x%lx\n", offset); return 0;
	}
}

void ACALSimDeviceComponent::writeRegister(uint64_t offset, uint32_t value) {
	switch (offset) {
		case REG_DATA_IN:
			out_.verbose(CALL_INFO, 3, 0, "Write DATA_IN = 0x%x\n", value);
			data_in_ = value;

			// If not already busy, start echo operation
			if (!echo_pending_) {
				status_ |= STATUS_BUSY;
				echo_pending_        = true;
				echo_complete_cycle_ = current_cycle_ + echo_latency_;
				out_.verbose(CALL_INFO, 2, 0, "Starting echo operation (will complete at cycle %lu)\n",
				             echo_complete_cycle_);
			}
			break;

		case REG_DATA_OUT: out_.verbose(CALL_INFO, 2, 0, "Attempted write to DATA_OUT (read-only, ignored)\n"); break;

		case REG_STATUS: out_.verbose(CALL_INFO, 2, 0, "Attempted write to STATUS (read-only, ignored)\n"); break;

		case REG_CONTROL:
			out_.verbose(CALL_INFO, 3, 0, "Write CONTROL = 0x%x\n", value);
			control_ = value;

			// Check for reset command
			if (control_ & CONTROL_RESET) {
				out_.verbose(CALL_INFO, 1, 0, "Device reset triggered\n");
				resetDevice();
			}
			break;

		default:
			out_.verbose(CALL_INFO, 2, 0, "Write to undefined register offset 0x%lx value 0x%x\n", offset, value);
			break;
	}
}

void ACALSimDeviceComponent::completeEcho() {
	out_.verbose(CALL_INFO, 2, 0, "Echo operation complete: DATA_IN (0x%x) -> DATA_OUT\n", data_in_);

	// Copy data from input to output
	data_out_ = data_in_;

	// Update status
	status_ &= ~STATUS_BUSY;
	status_ |= STATUS_DATA_READY;

	// Clear pending flag
	echo_pending_ = false;

	// Update statistics
	total_echos_++;
}

void ACALSimDeviceComponent::resetDevice() {
	out_.verbose(CALL_INFO, 2, 0, "Resetting device to initial state\n");

	// Clear all registers
	data_in_  = 0;
	data_out_ = 0;
	status_   = 0;  // Not busy, no data ready
	control_  = 0;

	// Clear echo state
	echo_pending_        = false;
	echo_complete_cycle_ = 0;
}
