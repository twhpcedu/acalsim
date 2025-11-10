/*
Copyright 2023-2025 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "../../include/HSA/HSAHostComponent.hh"

using namespace SST::ACALSim::HSA;

HSAHostComponent::HSAHostComponent(SST::ComponentId_t id, SST::Params& params) :
    SST::Component(id),
    next_dispatch_id_(0),
    dispatches_submitted_(0),
    dispatches_completed_(0),
    next_dispatch_cycle_(0) {

    // Get parameters
    verbose_ = params.find<uint32_t>("verbose", 0);
    num_dispatches_ = params.find<uint32_t>("num_dispatches", 10);
    workgroup_size_x_ = params.find<uint32_t>("workgroup_size_x", 256);
    workgroup_size_y_ = params.find<uint32_t>("workgroup_size_y", 1);
    workgroup_size_z_ = params.find<uint32_t>("workgroup_size_z", 1);
    grid_size_x_ = params.find<uint32_t>("grid_size_x", 1024);
    grid_size_y_ = params.find<uint32_t>("grid_size_y", 1);
    grid_size_z_ = params.find<uint32_t>("grid_size_z", 1);
    dispatch_interval_ = params.find<uint64_t>("dispatch_interval", 1000);

    // Configure output
    output_.init("HSAHost[@f:@l:@p] ", verbose_, 0, SST::Output::STDOUT);

    // Configure ports
    aql_port_ = configureLink("aql_port");
    if (!aql_port_) {
        output_.fatal(CALL_INFO, -1, "Error: Failed to configure aql_port\n");
    }

    signal_port_ = configureLink("signal_port",
        new SST::Event::Handler<HSAHostComponent>(this, &HSAHostComponent::handleCompletionSignal));
    if (!signal_port_) {
        output_.fatal(CALL_INFO, -1, "Error: Failed to configure signal_port\n");
    }

    doorbell_port_ = configureLink("doorbell_port",
        new SST::Event::Handler<HSAHostComponent>(this, &HSAHostComponent::handleDoorbellAck));

    // Configure statistics
    stat_dispatches_submitted_ = registerStatistic<uint64_t>("dispatches_submitted");
    stat_dispatches_completed_ = registerStatistic<uint64_t>("dispatches_completed");
    stat_avg_latency_ = registerStatistic<uint64_t>("avg_latency");

    // Register clock
    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    registerClock(clock_freq,
        new SST::Clock::Handler<HSAHostComponent>(this, &HSAHostComponent::clockTick));

    output_.verbose(CALL_INFO, 1, 0,
        "HSA Host Component initialized: %u dispatches, workgroup=%ux%ux%u, grid=%ux%ux%u\n",
        num_dispatches_, workgroup_size_x_, workgroup_size_y_, workgroup_size_z_,
        grid_size_x_, grid_size_y_, grid_size_z_);
}

HSAHostComponent::~HSAHostComponent() {
}

void HSAHostComponent::setup() {
    output_.verbose(CALL_INFO, 1, 0, "HSA Host setup complete\n");
}

void HSAHostComponent::finish() {
    output_.verbose(CALL_INFO, 1, 0,
        "HSA Host finish: Submitted=%u, Completed=%u\n",
        dispatches_submitted_, dispatches_completed_);

    if (dispatches_completed_ != dispatches_submitted_) {
        output_.output(CALL_INFO,
            "Warning: Not all dispatches completed (%u/%u)\n",
            dispatches_completed_, dispatches_submitted_);
    }
}

bool HSAHostComponent::clockTick(SST::Cycle_t currentCycle) {
    // Check if we should submit a new dispatch
    if (dispatches_submitted_ < num_dispatches_ && currentCycle >= next_dispatch_cycle_) {
        submitKernelDispatch();
        next_dispatch_cycle_ = currentCycle + dispatch_interval_;
    }

    // Continue running until all dispatches complete
    return false;
}

void HSAHostComponent::submitKernelDispatch() {
    uint32_t dispatch_id = next_dispatch_id_++;
    uint64_t submit_time = getCurrentSimTime();

    // Create AQL packet
    HSAAQLPacketEvent* packet = createAQLPacket(dispatch_id);
    packet->submit_time = submit_time;

    // Track submission
    uint64_t signal_handle = packet->completion_signal;
    signal_to_dispatch_[signal_handle] = dispatch_id;
    dispatch_submit_time_[dispatch_id] = submit_time;

    // Send packet through AQL queue link
    aql_port_->send(packet);

    dispatches_submitted_++;
    stat_dispatches_submitted_->addData(1);

    output_.verbose(CALL_INFO, 2, 0,
        "Submitted dispatch %u: signal=0x%lx, workitems=%lu, workgroups=%lu\n",
        dispatch_id, signal_handle, packet->getTotalWorkitems(), packet->getTotalWorkgroups());

    // Optionally ring doorbell
    if (doorbell_port_) {
        HSADoorbellEvent* doorbell = new HSADoorbellEvent(0, dispatches_submitted_);
        doorbell->timestamp = submit_time;
        doorbell_port_->send(doorbell);
    }
}

HSAAQLPacketEvent* HSAHostComponent::createAQLPacket(uint32_t dispatch_id) {
    HSAAQLPacketEvent* packet = new HSAAQLPacketEvent();

    // Set packet type and header
    packet->setPacketType(HSA_PACKET_TYPE_KERNEL_DISPATCH);
    packet->header |= (HSA_FENCE_SCOPE_SYSTEM & 0x3);  // Fence scope bits

    // Set work dimensions
    packet->workgroup_size_x = workgroup_size_x_;
    packet->workgroup_size_y = workgroup_size_y_;
    packet->workgroup_size_z = workgroup_size_z_;
    packet->grid_size_x = grid_size_x_;
    packet->grid_size_y = grid_size_y_;
    packet->grid_size_z = grid_size_z_;

    // Set kernel information (simulated addresses)
    packet->kernel_object = 0x100000 + (dispatch_id * 0x1000);
    packet->kernarg_address = 0x200000 + (dispatch_id * 0x1000);

    // Set memory requirements
    packet->private_segment_size = 1024;   // 1KB per work-item
    packet->group_segment_size = 4096;     // 4KB per work-group

    // Set completion signal (unique handle based on dispatch ID)
    packet->completion_signal = 0x1000 + dispatch_id;
    packet->dispatch_id = dispatch_id;

    return packet;
}

void HSAHostComponent::handleCompletionSignal(SST::Event* ev) {
    HSASignalEvent* signal = dynamic_cast<HSASignalEvent*>(ev);
    if (!signal) {
        output_.fatal(CALL_INFO, -1, "Error: Received non-signal event on signal port\n");
    }

    uint64_t signal_handle = signal->signal_handle;
    int64_t signal_value = signal->signal_value;

    output_.verbose(CALL_INFO, 2, 0,
        "Received completion signal: handle=0x%lx, value=%ld\n",
        signal_handle, signal_value);

    // Find associated dispatch
    auto it = signal_to_dispatch_.find(signal_handle);
    if (it == signal_to_dispatch_.end()) {
        output_.output(CALL_INFO, "Warning: Unknown signal handle 0x%lx\n", signal_handle);
        delete ev;
        return;
    }

    uint32_t dispatch_id = it->second;

    // Check if kernel completed successfully (value == 0)
    if (signal_value == 0) {
        dispatches_completed_++;
        stat_dispatches_completed_->addData(1);

        // Calculate latency
        auto time_it = dispatch_submit_time_.find(dispatch_id);
        if (time_it != dispatch_submit_time_.end()) {
            uint64_t latency = getCurrentSimTime() - time_it->second;
            stat_avg_latency_->addData(latency);

            output_.verbose(CALL_INFO, 1, 0,
                "Dispatch %u completed: latency=%lu ns\n",
                dispatch_id, latency);

            dispatch_submit_time_.erase(time_it);
        }

        signal_to_dispatch_.erase(it);

        // Check if all dispatches complete
        if (dispatches_completed_ == num_dispatches_) {
            output_.verbose(CALL_INFO, 1, 0,
                "All %u dispatches completed\n", num_dispatches_);
        }
    } else {
        output_.output(CALL_INFO,
            "Warning: Dispatch %u signaled with error value %ld\n",
            dispatch_id, signal_value);
    }

    delete ev;
}

void HSAHostComponent::handleDoorbellAck(SST::Event* ev) {
    HSADoorbellEvent* ack = dynamic_cast<HSADoorbellEvent*>(ev);
    if (ack) {
        output_.verbose(CALL_INFO, 3, 0,
            "Doorbell acknowledged: queue=%u, value=%lu\n",
            ack->queue_id, ack->doorbell_value);
    }
    delete ev;
}
