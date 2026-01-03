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

#include "QEMUComponent.hh"

#include <sst/core/sst_config.h>

using namespace ACALSim::QEMUIntegration;

QEMUComponent::QEMUComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id),
      current_cycle_(0),
      next_req_id_(1),
      test_state_(TestState::IDLE),
      iteration_(0),
      write_req_id_(0),
      read_req_id_(0),
      read_data_(0),
      status_data_(0),
      waiting_response_(false),
      total_loads_(0),
      total_stores_(0),
      total_successes_(0),
      total_failures_(0) {
	// Initialize output
	int verbose = params.find<int>("verbose", 1);
	out_.init("QEMU[@p:@l]: ", verbose, 0, SST::Output::STDOUT);

	out_.verbose(CALL_INFO, 1, 0, "Initializing QEMU Component\\n");

	// Get parameters
	std::string clock_freq = params.find<std::string>("clock", "1GHz");
	device_base_           = params.find<uint64_t>("device_base", 0x10000000ULL);
	device_size_           = params.find<uint64_t>("device_size", 4096);
	test_pattern_          = params.find<uint32_t>("test_pattern", 0xDEADBEEF);
	num_iterations_        = params.find<uint32_t>("num_iterations", 5);

	out_.verbose(CALL_INFO, 1, 0, "Configuration:\\n");
	out_.verbose(CALL_INFO, 1, 0, "  Clock: %s\\n", clock_freq.c_str());
	out_.verbose(CALL_INFO, 1, 0, "  Device Base: 0x%lx\\n", device_base_);
	out_.verbose(CALL_INFO, 1, 0, "  Device Size: %lu bytes\\n", device_size_);
	out_.verbose(CALL_INFO, 1, 0, "  Test Pattern: 0x%x\\n", test_pattern_);
	out_.verbose(CALL_INFO, 1, 0, "  Iterations: %u\\n", num_iterations_);

	// Configure clock
	tc_ = registerClock(clock_freq, new SST::Clock::Handler<QEMUComponent>(this, &QEMUComponent::clockTick));

	// Configure link to device
	device_link_ = configureLink("device_port",
	                             new SST::Event::Handler<QEMUComponent>(this, &QEMUComponent::handleMemoryResponse));

	if (!device_link_) { out_.fatal(CALL_INFO, -1, "Error: Failed to configure device_port link\\n"); }

	// Tell SST this component is not ready to end simulation yet
	registerAsPrimaryComponent();
	primaryComponentDoNotEndSim();

	out_.verbose(CALL_INFO, 1, 0, "QEMU component initialized successfully\\n");
}

QEMUComponent::~QEMUComponent() {
	// Clean up any pending responses
	while (!response_queue_.empty()) {
		delete response_queue_.front();
		response_queue_.pop();
	}
}

void QEMUComponent::setup() {
	out_.verbose(CALL_INFO, 1, 0, "Setup phase\\n");
	out_.verbose(CALL_INFO, 1, 0, "Starting test program simulation\\n");
}

void QEMUComponent::finish() {
	out_.verbose(CALL_INFO, 1, 0, "Finish phase\\n");
	out_.verbose(CALL_INFO, 1, 0, "Test Results:\\n");
	out_.verbose(CALL_INFO, 1, 0, "  Iterations:    %u\\n", iteration_);
	out_.verbose(CALL_INFO, 1, 0, "  Successes:     %lu\\n", total_successes_);
	out_.verbose(CALL_INFO, 1, 0, "  Failures:      %lu\\n", total_failures_);
	out_.verbose(CALL_INFO, 1, 0, "  Total Loads:   %lu\\n", total_loads_);
	out_.verbose(CALL_INFO, 1, 0, "  Total Stores:  %lu\\n", total_stores_);

	if (total_failures_ == 0 && iteration_ == num_iterations_) {
		out_.verbose(CALL_INFO, 1, 0, "\\n*** TEST PASSED ***\\n");
	} else {
		out_.verbose(CALL_INFO, 1, 0, "\\n*** TEST FAILED ***\\n");
	}
}

bool QEMUComponent::clockTick(SST::Cycle_t cycle) {
	current_cycle_ = cycle;

	out_.verbose(CALL_INFO, 1, 0, "[CLOCK] Cycle %lu, State: %d\\n", cycle, static_cast<int>(test_state_));

	// Process any pending responses
	processResponses();

	// Run test program state machine
	runTestProgram();

	// Clock handler return value: false = continue, true = done
	bool is_done = (test_state_ == TestState::DONE);
	out_.verbose(CALL_INFO, 1, 0, "[CLOCK] Returning %s (state=%d)\\n", is_done ? "true (DONE)" : "false (CONTINUE)",
	             static_cast<int>(test_state_));

	// If test is done, tell SST this component is ready to end
	if (is_done) { primaryComponentOKToEndSim(); }

	return is_done;  // true = unregister clock, false = keep ticking
}

void QEMUComponent::handleMemoryResponse(SST::Event* ev) {
	MemoryResponseEvent* resp = dynamic_cast<MemoryResponseEvent*>(ev);
	if (!resp) {
		out_.fatal(CALL_INFO, -1, "Error: Received invalid event type\\n");
		return;
	}

	out_.verbose(CALL_INFO, 3, 0, "Received response: req_id=%lu data=0x%x success=%d\\n", resp->getReqId(),
	             resp->getData(), resp->getSuccess());

	// Queue response for processing
	response_queue_.push(resp);
}

void QEMUComponent::processResponses() {
	while (!response_queue_.empty()) {
		MemoryResponseEvent* resp = response_queue_.front();
		response_queue_.pop();

		uint64_t req_id = resp->getReqId();

		// Find matching pending transaction
		auto it = pending_transactions_.find(req_id);
		if (it == pending_transactions_.end()) {
			out_.verbose(CALL_INFO, 2, 0, "Warning: Response for unknown request %lu\\n", req_id);
			delete resp;
			continue;
		}

		PendingTransaction& trans   = it->second;
		uint64_t            latency = current_cycle_ - trans.issue_cycle;

		out_.verbose(CALL_INFO, 2, 0, "Completed %s: addr=0x%lx latency=%lu cycles\\n",
		             (trans.type == TransactionType::LOAD) ? "LOAD" : "STORE", trans.address, latency);

		if (trans.type == TransactionType::LOAD) {
			// Store read data for test program
			if (trans.address == device_base_ + REG_DATA_OUT) {
				read_data_ = resp->getData();
			} else if (trans.address == device_base_ + REG_STATUS) {
				status_data_ = resp->getData();
			}
		}

		// Remove from pending
		pending_transactions_.erase(it);
		waiting_response_ = false;

		delete resp;
	}
}

void QEMUComponent::runTestProgram() {
	// Don't issue new requests if waiting for response
	if (waiting_response_) { return; }

	switch (test_state_) {
		case TestState::IDLE:
			// Start first iteration
			iteration_  = 0;
			test_state_ = TestState::WRITE_DATA;
			out_.verbose(CALL_INFO, 1, 0, "\\n=== Starting Test Iteration %u ===\\n", iteration_ + 1);
			break;

		case TestState::WRITE_DATA: {
			// Write test pattern to device DATA_IN register
			uint32_t pattern = test_pattern_ + iteration_;  // Vary pattern per iteration
			out_.verbose(CALL_INFO, 1, 0, "Writing pattern 0x%x to DATA_IN\\n", pattern);
			write_req_id_     = sendStore(device_base_ + REG_DATA_IN, pattern, 4);
			waiting_response_ = true;
			test_state_       = TestState::WAIT_BUSY;
			break;
		}

		case TestState::WAIT_BUSY:
			// Wait a few cycles for device to start processing
			out_.verbose(CALL_INFO, 2, 0, "Waiting for device to process...\\n");
			test_state_ = TestState::READ_STATUS;
			break;

		case TestState::READ_STATUS: {
			// Read device STATUS register
			out_.verbose(CALL_INFO, 2, 0, "Reading STATUS register\\n");
			read_req_id_      = sendLoad(device_base_ + REG_STATUS, 4);
			waiting_response_ = true;
			test_state_       = TestState::READ_DATA;
			break;
		}

		case TestState::READ_DATA: {
			// Check if device is ready
			if (status_data_ & STATUS_BUSY) {
				// Still busy, read status again
				out_.verbose(CALL_INFO, 2, 0, "Device busy (status=0x%x), checking again\\n", status_data_);
				read_req_id_      = sendLoad(device_base_ + REG_STATUS, 4);
				waiting_response_ = true;
				// Stay in READ_DATA state
			} else if (status_data_ & STATUS_DATA_READY) {
				// Device ready, read DATA_OUT
				out_.verbose(CALL_INFO, 2, 0, "Device ready, reading DATA_OUT\\n");
				read_req_id_      = sendLoad(device_base_ + REG_DATA_OUT, 4);
				waiting_response_ = true;
				test_state_       = TestState::VERIFY;
			} else {
				// Read status again
				read_req_id_      = sendLoad(device_base_ + REG_STATUS, 4);
				waiting_response_ = true;
			}
			break;
		}

		case TestState::VERIFY: {
			// Verify read data matches written pattern
			uint32_t expected = test_pattern_ + iteration_;
			if (read_data_ == expected) {
				out_.verbose(CALL_INFO, 1, 0, "✓ Test iteration %u PASSED (read=0x%x)\\n", iteration_ + 1, read_data_);
				total_successes_++;
			} else {
				out_.verbose(CALL_INFO, 1, 0, "✗ Test iteration %u FAILED (expected=0x%x, read=0x%x)\\n",
				             iteration_ + 1, expected, read_data_);
				total_failures_++;
			}

			// Move to next iteration or finish
			iteration_++;
			if (iteration_ < num_iterations_) {
				test_state_ = TestState::WRITE_DATA;
				out_.verbose(CALL_INFO, 1, 0, "\\n=== Starting Test Iteration %u ===\\n", iteration_ + 1);
			} else {
				out_.verbose(CALL_INFO, 1, 0, "\\n=== All Test Iterations Complete ===\\n");
				test_state_ = TestState::DONE;
			}
			break;
		}

		case TestState::DONE:
			// Test complete, will stop simulation
			break;
	}
}

uint64_t QEMUComponent::sendLoad(uint64_t addr, uint32_t size) {
	uint64_t req_id = next_req_id_++;

	out_.verbose(CALL_INFO, 2, 0, "Issuing LOAD: addr=0x%lx size=%u req_id=%lu\\n", addr, size, req_id);

	// Create and send transaction event
	auto* trans = new MemoryTransactionEvent(TransactionType::LOAD, addr, 0, size, req_id);
	device_link_->send(trans);

	// Track pending transaction
	PendingTransaction pending;
	pending.req_id                = req_id;
	pending.type                  = TransactionType::LOAD;
	pending.address               = addr;
	pending.data                  = 0;
	pending.issue_cycle           = current_cycle_;
	pending_transactions_[req_id] = pending;

	total_loads_++;
	return req_id;
}

uint64_t QEMUComponent::sendStore(uint64_t addr, uint32_t data, uint32_t size) {
	uint64_t req_id = next_req_id_++;

	out_.verbose(CALL_INFO, 2, 0, "Issuing STORE: addr=0x%lx data=0x%x size=%u req_id=%lu\\n", addr, data, size,
	             req_id);

	// Create and send transaction event
	auto* trans = new MemoryTransactionEvent(TransactionType::STORE, addr, data, size, req_id);
	device_link_->send(trans);

	// Track pending transaction
	PendingTransaction pending;
	pending.req_id                = req_id;
	pending.type                  = TransactionType::STORE;
	pending.address               = addr;
	pending.data                  = data;
	pending.issue_cycle           = current_cycle_;
	pending_transactions_[req_id] = pending;

	total_stores_++;
	return req_id;
}

bool QEMUComponent::isDeviceAddress(uint64_t addr) const {
	return (addr >= device_base_ && addr < (device_base_ + device_size_));
}
