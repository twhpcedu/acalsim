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

#include "sst/ACALSimMMIODevice.hh"

using namespace ACALSim::QEMUIntegration;

ACALSimMMIODevice::ACALSimMMIODevice(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id),
      cpu_link_(nullptr),
      irq_link_(nullptr),
      reg_ctrl_(0),
      reg_status_(0),
      reg_int_status_(0),
      reg_int_enable_(0),
      reg_src_addr_(0),
      reg_dst_addr_(0),
      reg_length_(0),
      reg_data_in_(0),
      reg_data_out_(0),
      cycle_count_(0),
      irq_asserted_(false),
      total_ops_(0),
      total_latency_(0) {
	// Get parameters
	base_addr_   = params.find<uint64_t>("base_addr", 0x10001000);
	size_        = params.find<uint64_t>("size", 4096);
	verbose_     = params.find<uint32_t>("verbose", 1);
	reg_latency_ = params.find<uint32_t>("default_latency", 100);
	irq_num_     = params.find<uint32_t>("irq_num", 1);

	// Initialize device state
	current_op_.active      = false;
	current_op_.start_cycle = 0;
	current_op_.end_cycle   = 0;
	current_op_.src_addr    = 0;
	current_op_.dst_addr    = 0;
	current_op_.length      = 0;

	// Configure output
	output_.init("ACALSimMMIODevice[@f:@l:@p] ", verbose_, 0, SST::Output::STDOUT);

	// Configure CPU link (for MMIO transactions)
	cpu_link_ = configureLink(
	    "cpu_port", new SST::Event::Handler<ACALSimMMIODevice>(this, &ACALSimMMIODevice::handleMemoryTransaction));
	if (!cpu_link_) { output_.fatal(CALL_INFO, -1, "Error: Failed to configure cpu_port\n"); }

	// Configure IRQ link (for interrupt signaling)
	irq_link_ = configureLink("irq_port");
	if (!irq_link_) { output_.fatal(CALL_INFO, -1, "Error: Failed to configure irq_port\n"); }

	// Register statistics
	stat_mmio_reads_    = registerStatistic<uint64_t>("mmio_reads");
	stat_mmio_writes_   = registerStatistic<uint64_t>("mmio_writes");
	stat_ops_completed_ = registerStatistic<uint64_t>("operations_completed");
	stat_interrupts_    = registerStatistic<uint64_t>("interrupts_generated");
	stat_avg_latency_   = registerStatistic<uint64_t>("avg_operation_latency");

	// Register clock
	std::string clock_freq = params.find<std::string>("clock", "1GHz");
	registerClock(clock_freq, new SST::Clock::Handler<ACALSimMMIODevice>(this, &ACALSimMMIODevice::clockTick));

	output_.verbose(CALL_INFO, 1, 0, "ACALSim MMIO Device initialized: base=0x%lx, size=%lu, irq=%u, latency=%u\n",
	                base_addr_, size_, irq_num_, reg_latency_);
}

ACALSimMMIODevice::~ACALSimMMIODevice() {}

void ACALSimMMIODevice::setup() { output_.verbose(CALL_INFO, 1, 0, "ACALSim MMIO Device setup complete\n"); }

void ACALSimMMIODevice::finish() {
	output_.verbose(CALL_INFO, 1, 0, "ACALSim MMIO Device finish: MMIO ops: R=%lu W=%lu, Operations=%lu, IRQs=%lu\n",
	                stat_mmio_reads_->getCollectionCount(), stat_mmio_writes_->getCollectionCount(), total_ops_,
	                stat_interrupts_->getCollectionCount());
}

bool ACALSimMMIODevice::clockTick(SST::Cycle_t cycle) {
	cycle_count_ = cycle;

	// Check if current operation should complete
	if (current_op_.active && cycle >= current_op_.end_cycle) { completeOperation(); }

	return false;  // Keep clock running
}

void ACALSimMMIODevice::handleMemoryTransaction(SST::Event* ev) {
	MemoryTransactionEvent* trans = dynamic_cast<MemoryTransactionEvent*>(ev);
	if (!trans) { output_.fatal(CALL_INFO, -1, "Error: Received non-transaction event on cpu_port\n"); }

	uint64_t addr   = trans->getAddress();
	uint64_t offset = addr - base_addr_;
	uint32_t data   = trans->getData();
	uint32_t size   = trans->getSize();
	uint64_t req_id = trans->getReqId();

	// Validate address range
	if (offset >= size_) {
		output_.output(CALL_INFO, "Warning: Access out of range: offset=0x%lx\n", offset);
		MemoryResponseEvent* resp = new MemoryResponseEvent(req_id, 0, false);
		cpu_link_->send(resp);
		delete ev;
		return;
	}

	bool     is_load   = (trans->getType() == TransactionType::LOAD);
	uint32_t resp_data = 0;
	bool     success   = true;

	if (is_load) {
		// MMIO Read
		resp_data = readRegister(offset);
		stat_mmio_reads_->addData(1);

		output_.verbose(CALL_INFO, 2, 0, "MMIO READ: offset=0x%lx, data=0x%x, size=%u, req_id=%lu\n", offset, resp_data,
		                size, req_id);
	} else {
		// MMIO Write
		writeRegister(offset, data);
		stat_mmio_writes_->addData(1);

		output_.verbose(CALL_INFO, 2, 0, "MMIO WRITE: offset=0x%lx, data=0x%x, size=%u, req_id=%lu\n", offset, data,
		                size, req_id);
	}

	// Send response back to QEMU
	MemoryResponseEvent* resp = new MemoryResponseEvent(req_id, resp_data, success);
	cpu_link_->send(resp);

	delete ev;
}

uint32_t ACALSimMMIODevice::readRegister(uint64_t offset) {
	switch (offset) {
		case REG_CTRL: return reg_ctrl_;

		case REG_STATUS: return reg_status_;

		case REG_INT_STATUS: return reg_int_status_;

		case REG_INT_ENABLE: return reg_int_enable_;

		case REG_SRC_ADDR: return reg_src_addr_;

		case REG_DST_ADDR: return reg_dst_addr_;

		case REG_LENGTH: return reg_length_;

		case REG_LATENCY: return reg_latency_;

		case REG_DATA_IN: return reg_data_in_;

		case REG_DATA_OUT: return reg_data_out_;

		case REG_CYCLE_COUNT: return static_cast<uint32_t>(cycle_count_);

		default: output_.verbose(CALL_INFO, 3, 0, "Read from unknown register: offset=0x%lx\n", offset); return 0;
	}
}

void ACALSimMMIODevice::writeRegister(uint64_t offset, uint32_t value) {
	switch (offset) {
		case REG_CTRL:
			reg_ctrl_ = value;

			// Check for start operation
			if (value & CTRL_START) { startOperation(); }

			// Check for reset
			if (value & CTRL_RESET) { resetDevice(); }

			// Clear start bit (auto-clear)
			reg_ctrl_ &= ~CTRL_START;
			break;

		case REG_STATUS:
			// Read-only register
			output_.verbose(CALL_INFO, 3, 0, "Attempted write to read-only STATUS register\n");
			break;

		case REG_INT_STATUS:
			// Write-1-to-clear
			reg_int_status_ &= ~value;
			updateInterruptLine();
			break;

		case REG_INT_ENABLE:
			reg_int_enable_ = value;
			updateInterruptLine();
			break;

		case REG_SRC_ADDR: reg_src_addr_ = value; break;

		case REG_DST_ADDR: reg_dst_addr_ = value; break;

		case REG_LENGTH: reg_length_ = value; break;

		case REG_LATENCY: reg_latency_ = value; break;

		case REG_DATA_IN:
			reg_data_in_ = value;
			// Simple echo operation
			reg_data_out_ = value;
			break;

		case REG_DATA_OUT:
		case REG_CYCLE_COUNT:
			// Read-only registers
			output_.verbose(CALL_INFO, 3, 0, "Attempted write to read-only register: offset=0x%lx\n", offset);
			break;

		default:
			output_.verbose(CALL_INFO, 3, 0, "Write to unknown register: offset=0x%lx, value=0x%x\n", offset, value);
			break;
	}
}

void ACALSimMMIODevice::startOperation() {
	if (current_op_.active) {
		output_.output(CALL_INFO, "Warning: Starting operation while previous operation is active\n");
		return;
	}

	// Setup operation
	current_op_.active      = true;
	current_op_.start_cycle = cycle_count_;
	current_op_.end_cycle   = cycle_count_ + reg_latency_;
	current_op_.src_addr    = reg_src_addr_;
	current_op_.dst_addr    = reg_dst_addr_;
	current_op_.length      = reg_length_;

	// Update status
	reg_status_ |= STATUS_BUSY;
	reg_status_ &= ~STATUS_DONE;

	output_.verbose(CALL_INFO, 1, 0, "Operation started: src=0x%x, dst=0x%x, len=%u, latency=%lu cycles\n",
	                current_op_.src_addr, current_op_.dst_addr, current_op_.length,
	                current_op_.end_cycle - current_op_.start_cycle);
}

void ACALSimMMIODevice::completeOperation() {
	if (!current_op_.active) { return; }

	uint64_t latency = cycle_count_ - current_op_.start_cycle;

	// Update status
	reg_status_ &= ~STATUS_BUSY;
	reg_status_ |= STATUS_DONE;

	// Mark operation as complete
	current_op_.active = false;

	// Update statistics
	total_ops_++;
	total_latency_ += latency;
	stat_ops_completed_->addData(1);
	stat_avg_latency_->addData(latency);

	output_.verbose(CALL_INFO, 1, 0, "Operation completed: latency=%lu cycles, total_ops=%lu\n", latency, total_ops_);

	// Generate interrupt if enabled
	if (reg_ctrl_ & CTRL_INT_EN) { generateInterrupt(INT_COMPLETE); }
}

void ACALSimMMIODevice::resetDevice() {
	output_.verbose(CALL_INFO, 1, 0, "Device reset\n");

	// Reset registers
	reg_ctrl_       = 0;
	reg_status_     = 0;
	reg_int_status_ = 0;
	reg_int_enable_ = 0;
	reg_src_addr_   = 0;
	reg_dst_addr_   = 0;
	reg_length_     = 0;
	reg_data_in_    = 0;
	reg_data_out_   = 0;

	// Cancel current operation
	current_op_.active = false;

	// Clear interrupt line
	if (irq_asserted_) {
		InterruptEvent* irq = new InterruptEvent(irq_num_, InterruptEvent::Type::DEASSERT);
		irq_link_->send(irq);
		irq_asserted_ = false;
	}
}

void ACALSimMMIODevice::generateInterrupt(uint32_t irq_bits) {
	// Set interrupt status bits
	reg_int_status_ |= irq_bits;

	output_.verbose(CALL_INFO, 2, 0, "Interrupt generated: bits=0x%x, int_status=0x%x\n", irq_bits, reg_int_status_);

	updateInterruptLine();
}

void ACALSimMMIODevice::clearInterrupt(uint32_t irq_bits) {
	// Clear interrupt status bits
	reg_int_status_ &= ~irq_bits;

	output_.verbose(CALL_INFO, 2, 0, "Interrupt cleared: bits=0x%x, int_status=0x%x\n", irq_bits, reg_int_status_);

	updateInterruptLine();
}

void ACALSimMMIODevice::updateInterruptLine() {
	// Determine if interrupt should be asserted
	// IRQ is asserted if any enabled interrupt is pending
	bool should_assert = (reg_int_status_ & reg_int_enable_) != 0;

	if (should_assert && !irq_asserted_) {
		// Assert interrupt
		InterruptEvent* irq = new InterruptEvent(irq_num_, InterruptEvent::Type::ASSERT);
		irq_link_->send(irq);
		irq_asserted_ = true;
		stat_interrupts_->addData(1);

		output_.verbose(CALL_INFO, 2, 0, "IRQ %u ASSERTED: int_status=0x%x, int_enable=0x%x\n", irq_num_,
		                reg_int_status_, reg_int_enable_);

	} else if (!should_assert && irq_asserted_) {
		// Deassert interrupt
		InterruptEvent* irq = new InterruptEvent(irq_num_, InterruptEvent::Type::DEASSERT);
		irq_link_->send(irq);
		irq_asserted_ = false;

		output_.verbose(CALL_INFO, 2, 0, "IRQ %u DEASSERTED: int_status=0x%x, int_enable=0x%x\n", irq_num_,
		                reg_int_status_, reg_int_enable_);
	}
}
