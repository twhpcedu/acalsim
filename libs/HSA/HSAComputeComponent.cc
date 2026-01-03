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

#include "../../include/HSA/HSAComputeComponent.hh"

using namespace SST::ACALSim::HSA;

HSAComputeComponent::HSAComputeComponent(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), current_kernel_(nullptr), kernels_executed_(0), total_workitems_(0) {
	// Get parameters
	verbose_                = params.find<uint32_t>("verbose", 0);
	queue_depth_            = params.find<uint32_t>("queue_depth", 256);
	cycles_per_workitem_    = params.find<uint64_t>("cycles_per_workitem", 100);
	kernel_launch_overhead_ = params.find<uint64_t>("kernel_launch_overhead", 1000);
	memory_latency_         = params.find<uint64_t>("memory_latency", 100);

	// Configure output
	output_.init("HSACompute[@f:@l:@p] ", verbose_, 0, SST::Output::STDOUT);

	// Configure ports
	aql_port_ = configureLink(
	    "aql_port", new SST::Event::Handler<HSAComputeComponent>(this, &HSAComputeComponent::handleAQLPacket));
	if (!aql_port_) { output_.fatal(CALL_INFO, -1, "Error: Failed to configure aql_port\n"); }

	signal_port_ = configureLink("signal_port");
	if (!signal_port_) { output_.fatal(CALL_INFO, -1, "Error: Failed to configure signal_port\n"); }

	doorbell_port_ = configureLink(
	    "doorbell_port", new SST::Event::Handler<HSAComputeComponent>(this, &HSAComputeComponent::handleDoorbellRing));

	// Configure statistics
	stat_kernels_executed_   = registerStatistic<uint64_t>("kernels_executed");
	stat_total_workitems_    = registerStatistic<uint64_t>("total_workitems");
	stat_avg_kernel_latency_ = registerStatistic<uint64_t>("avg_kernel_latency");
	stat_queue_occupancy_    = registerStatistic<uint64_t>("queue_occupancy");

	// Register clock
	std::string clock_freq = params.find<std::string>("clock", "1GHz");
	registerClock(clock_freq, new SST::Clock::Handler<HSAComputeComponent>(this, &HSAComputeComponent::clockTick));

	output_.verbose(CALL_INFO, 1, 0, "HSA Compute Component initialized: queue_depth=%u, cycles_per_workitem=%lu\n",
	                queue_depth_, cycles_per_workitem_);
}

HSAComputeComponent::~HSAComputeComponent() {
	if (current_kernel_) {
		delete current_kernel_->packet;
		delete current_kernel_;
	}

	while (!aql_queue_.empty()) {
		QueuedKernel& qk = aql_queue_.front();
		delete qk.packet;
		aql_queue_.pop();
	}
}

void HSAComputeComponent::setup() { output_.verbose(CALL_INFO, 1, 0, "HSA Compute setup complete\n"); }

void HSAComputeComponent::finish() {
	output_.verbose(CALL_INFO, 1, 0, "HSA Compute finish: Kernels=%lu, Total Workitems=%lu\n", kernels_executed_,
	                total_workitems_);
}

bool HSAComputeComponent::clockTick(SST::Cycle_t currentCycle) {
	// Record queue occupancy
	stat_queue_occupancy_->addData(aql_queue_.size() + (current_kernel_ ? 1 : 0));

	// Check if current kernel completed
	if (current_kernel_ && currentCycle >= current_kernel_->completion_time) {
		HSAAQLPacketEvent* packet  = current_kernel_->packet;
		uint64_t           latency = current_kernel_->completion_time - current_kernel_->start_time;

		kernels_executed_++;
		total_workitems_ += packet->getTotalWorkitems();

		stat_kernels_executed_->addData(1);
		stat_total_workitems_->addData(packet->getTotalWorkitems());
		stat_avg_kernel_latency_->addData(latency);

		output_.verbose(CALL_INFO, 1, 0, "Kernel completed: dispatch=%u, latency=%lu cycles, workitems=%lu\n",
		                packet->dispatch_id, latency, packet->getTotalWorkitems());

		// Send completion signal
		sendCompletionSignal(packet->completion_signal, 0, packet->dispatch_id);

		delete current_kernel_->packet;
		delete current_kernel_;
		current_kernel_ = nullptr;
	}

	// Start next kernel from queue
	if (!current_kernel_ && !aql_queue_.empty()) {
		current_kernel_ = new QueuedKernel(aql_queue_.front());
		aql_queue_.pop();

		current_kernel_->start_time      = currentCycle;
		current_kernel_->completion_time = currentCycle + calculateKernelLatency(current_kernel_->packet);

		output_.verbose(CALL_INFO, 2, 0, "Starting kernel: dispatch=%u, estimated_latency=%lu cycles\n",
		                current_kernel_->packet->dispatch_id,
		                current_kernel_->completion_time - current_kernel_->start_time);
	}

	return false;  // Keep clock running
}

void HSAComputeComponent::handleAQLPacket(SST::Event* ev) {
	HSAAQLPacketEvent* packet = dynamic_cast<HSAAQLPacketEvent*>(ev);
	if (!packet) { output_.fatal(CALL_INFO, -1, "Error: Received non-AQL event on aql_port\n"); }

	output_.verbose(CALL_INFO, 2, 0, "Received AQL packet: type=%u, dispatch=%u, signal=0x%lx, workitems=%lu\n",
	                packet->getPacketType(), packet->dispatch_id, packet->completion_signal,
	                packet->getTotalWorkitems());

	// Check queue capacity
	if (aql_queue_.size() >= queue_depth_) {
		output_.output(CALL_INFO, "Warning: AQL queue full (depth=%u), dropping packet\n", queue_depth_);
		sendCompletionSignal(packet->completion_signal, -1, packet->dispatch_id);  // Error signal
		delete packet;
		return;
	}

	// Add to queue
	QueuedKernel qk;
	qk.packet          = packet;
	qk.start_time      = 0;
	qk.completion_time = 0;
	aql_queue_.push(qk);
}

void HSAComputeComponent::handleDoorbellRing(SST::Event* ev) {
	HSADoorbellEvent* doorbell = dynamic_cast<HSADoorbellEvent*>(ev);
	if (doorbell) {
		output_.verbose(CALL_INFO, 3, 0, "Doorbell ring received: queue=%u, value=%lu\n", doorbell->queue_id,
		                doorbell->doorbell_value);
	}
	delete ev;
}

uint64_t HSAComputeComponent::calculateKernelLatency(HSAAQLPacketEvent* packet) {
	uint64_t total_workitems = packet->getTotalWorkitems();
	uint64_t workgroups      = packet->getTotalWorkgroups();

	// Model kernel execution latency
	// Latency = launch_overhead + (workitems * cycles_per_workitem) + memory_accesses
	uint64_t compute_latency = total_workitems * cycles_per_workitem_;

	// Estimate memory accesses (assume each workgroup reads/writes kernel args)
	uint64_t memory_accesses      = workgroups * 2;  // 2 accesses per workgroup
	uint64_t memory_latency_total = memory_accesses * memory_latency_;

	uint64_t total_latency = kernel_launch_overhead_ + compute_latency + memory_latency_total;

	output_.verbose(CALL_INFO, 3, 0, "Kernel latency breakdown: launch=%lu, compute=%lu, memory=%lu, total=%lu\n",
	                kernel_launch_overhead_, compute_latency, memory_latency_total, total_latency);

	return total_latency;
}

void HSAComputeComponent::sendCompletionSignal(uint64_t signal_handle, int64_t value, uint32_t dispatch_id) {
	HSASignalEvent* signal = new HSASignalEvent(signal_handle, value, HSASignalEvent::Operation::STORE);
	signal->timestamp      = getCurrentSimTime();
	signal->dispatch_id    = dispatch_id;

	signal_port_->send(signal);

	output_.verbose(CALL_INFO, 2, 0, "Sent completion signal: handle=0x%lx, value=%ld, dispatch=%u\n", signal_handle,
	                value, dispatch_id);
}
