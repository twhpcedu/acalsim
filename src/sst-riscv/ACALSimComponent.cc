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

#include "ACALSimComponent.hh"

#include <sst/core/sst_config.h>

// Include actual ACALSim headers here (not in .hh to avoid pulling in PyTorch)
// Note: We don't include SimConfig.hh or SimBase.hh because they require C++20
// and SST forces C++17 compilation. Derived classes can include what they need.

using namespace ACALSim::SSTIntegration;

// ==================== ACALSimComponent Implementation ====================

ACALSimComponent::ACALSimComponent(::SST::ComponentId_t id, ::SST::Params& params)
    : ::SST::Component(id),
      current_tick_(0),
      max_ticks_(0),
      num_ports_(0),
      events_received_(0),
      events_sent_(0),
      ticks_executed_(0) {
	// Initialize output for logging
	int verbose = params.find<int>("verbose", 1);
	out_.init("ACALSimComponent[@p:@l]: ", verbose, 0, ::SST::Output::STDOUT);

	out_.verbose(CALL_INFO, 1, 0, "Initializing ACALSim SST Component\n");

	// Get clock frequency
	clock_freq_ = params.find<std::string>("clock", "1GHz");
	out_.verbose(CALL_INFO, 2, 0, "Clock frequency: %s\n", clock_freq_.c_str());

	// Get max ticks
	max_ticks_ = params.find<uint64_t>("max_ticks", 0);
	if (max_ticks_ > 0) { out_.verbose(CALL_INFO, 2, 0, "Max ticks: %lu\n", max_ticks_); }

	// Initialize ACALSim simulator
	initACALSimulator(params);

	// Configure links (ports)
	configureLinks(params);

	// Register clock handler
	tc_ = registerClock(clock_freq_, new ::SST::Clock::Handler2<ACALSimComponent, &ACALSimComponent::clockTick>(this));

	out_.verbose(CALL_INFO, 1, 0, "Component initialization complete\n");

	// Tell SST we're a primary component that controls simulation end
	registerAsPrimaryComponent();
	primaryComponentDoNotEndSim();
}

ACALSimComponent::~ACALSimComponent() { out_.verbose(CALL_INFO, 1, 0, "Destroying ACALSimComponent\n"); }

void ACALSimComponent::initACALSimulator(::SST::Params& params) {
	out_.verbose(CALL_INFO, 2, 0, "Initializing ACALSim simulator\n");

	// Get simulator configuration
	std::string config_file = params.find<std::string>("config_file", "");
	std::string sim_name    = params.find<std::string>("name", "acalsim0");

	// Initialize ACALSim configuration
	if (!config_file.empty()) {
		out_.verbose(CALL_INFO, 2, 0, "Loading config from: %s\n", config_file.c_str());
		// In a real implementation, load JSON config here
	}

	// Store simulator name
	config_name_ = sim_name;

	// Note: In a complete implementation, you would:
	// 1. Create the appropriate SimBase-derived instance based on "simulator_type" param
	// 2. Initialize it with the config
	// 3. Register any modules/components
	//
	// For now, this is a framework that users would extend with their specific simulator types

	out_.verbose(CALL_INFO, 2, 0, "ACALSim simulator '%s' initialized\n", sim_name.c_str());
}

void ACALSimComponent::configureLinks(::SST::Params& params) {
	out_.verbose(CALL_INFO, 2, 0, "Configuring SST links\n");

	// Get number of ports to configure
	num_ports_ = params.find<int>("num_ports", 0);

	// Configure each port
	for (int i = 0; i < num_ports_; i++) {
		std::string port_name = "port" + std::to_string(i);

		// Configure link with event handler
		::SST::Link* link = configureLink(
		    port_name, new ::SST::Event::Handler2<ACALSimComponent, &ACALSimComponent::handleEvent>(this));

		if (link) {
			links_[i]      = link;
			port_names_[i] = port_name;
			out_.verbose(CALL_INFO, 3, 0, "Configured link: %s (port %d)\n", port_name.c_str(), i);
		} else {
			out_.verbose(CALL_INFO, 2, 0, "Warning: Link %s not connected\n", port_name.c_str());
		}
	}

	out_.verbose(CALL_INFO, 2, 0, "Configured %d links\n", (int)links_.size());
}

void ACALSimComponent::setup() {
	out_.verbose(CALL_INFO, 1, 0, "Setup phase\n");

	// Initialize the ACALSim simulator
	// Derived classes should override this and initialize their simulator
}

void ACALSimComponent::init(unsigned int phase) {
	out_.verbose(CALL_INFO, 2, 0, "Init phase %u\n", phase);

	// Multi-phase initialization allows components to exchange
	// configuration information. ACALSim components could use this
	// to discover neighbors, exchange parameters, etc.
}

void ACALSimComponent::finish() {
	out_.verbose(CALL_INFO, 1, 0, "Finish phase\n");

	// Cleanup ACALSim simulator
	// Derived classes should override this and cleanup their simulator

	// Print statistics
	out_.output(CALL_INFO, "\n=== ACALSim Component Statistics ===\n");
	out_.output(CALL_INFO, "Ticks executed:    %lu\n", ticks_executed_);
	out_.output(CALL_INFO, "Events received:   %lu\n", events_received_);
	out_.output(CALL_INFO, "Events sent:       %lu\n", events_sent_);
	out_.output(CALL_INFO, "====================================\n");

	// Tell SST we're done
	primaryComponentOKToEndSim();
}

bool ACALSimComponent::clockTick(::SST::Cycle_t cycle) {
	// Execute one ACALSim simulation step
	// Derived classes should override this and step their simulator
	(void)cycle;  // Unused in base class

	// Increment tick counter
	current_tick_++;
	ticks_executed_++;

	// Process any outbound events from ACALSim
	processOutboundEvents();

	// Check if we should continue
	bool should_continue = true;

	// Stop if max_ticks reached
	if (max_ticks_ > 0 && current_tick_ >= max_ticks_) {
		out_.verbose(CALL_INFO, 1, 0, "Reached max ticks (%lu), ending simulation\n", max_ticks_);
		should_continue = false;
	}

	// Stop if simulator is done
	// Derived classes should override this to check their simulator state

	if (!should_continue) { primaryComponentOKToEndSim(); }

	return should_continue;
}

void ACALSimComponent::handleEvent(::SST::Event* ev) {
	events_received_++;

	auto* acal_ev = dynamic_cast<ACALSimSSTEvent*>(ev);
	if (!acal_ev) {
		out_.verbose(CALL_INFO, 1, 0, "Warning: Received non-ACALSim event\n");
		delete ev;
		return;
	}

	out_.verbose(CALL_INFO, 3, 0, "Received event\n");

	// Get the ACALSim packet
	auto packet = acal_ev->getPacket();

	if (packet) {
		// Deliver packet to ACALSim simulator
		// Derived classes should override handleEvent to process packets
		// In a real implementation, you would route the packet to the
		// appropriate SimModule or inject it into the appropriate channel/port

		// For example:
		// simulator_->injectPacket(packet);
	}

	delete ev;
}

void ACALSimComponent::sendPacket(std::shared_ptr<SimPacket> packet, int port_id, Tick delay) {
	auto it = links_.find(port_id);
	if (it == links_.end()) {
		out_.verbose(CALL_INFO, 1, 0, "Warning: Cannot send packet - port %d not connected\n", port_id);
		return;
	}

	// Create SST event wrapper
	auto* sst_ev = new ACALSimSSTEvent(packet);

	// Send with delay (convert ACALSim ticks to SST time)
	::SST::SimTime_t sst_delay = tc_->getFactor() * delay;
	it->second->send(sst_delay, sst_ev);

	events_sent_++;
	out_.verbose(CALL_INFO, 3, 0, "Sent packet on port %d with delay %lu\n", port_id, delay);
}

void ACALSimComponent::processOutboundEvents() {
	// In a real implementation, this would:
	// 1. Check ACALSim simulator's outbound queues/channels
	// 2. Extract packets destined for external SST components
	// 3. Call sendPacket() to transmit them via SST links
	//
	// For example:
	// for (auto& [port_id, channel] : simulator_->getOutboundChannels()) {
	//     while (!channel->empty()) {
	//         auto packet = channel->pop();
	//         sendPacket(packet, port_id);
	//     }
	// }
}

// Register component with SST
// This must be after the class definition
using namespace ACALSim::SSTIntegration;
