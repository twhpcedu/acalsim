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

#ifndef __ACALSIM_SST_COMPONENT_HH__
#define __ACALSIM_SST_COMPONENT_HH__

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/timeConverter.h>

// Minimal ACALSim includes (avoids PyTorch and heavy dependencies)
#include "ACALSim_Minimal.hh"

namespace ACALSim {
namespace SSTIntegration {

/**
 * @brief SST Event wrapper for ACALSim SimPackets
 *
 * This class wraps ACALSim SimPacket objects as SST Events
 * to enable communication between ACALSim components and
 * native SST components via SST Links.
 */
class ACALSimSSTEvent : public ::SST::Event {
public:
	/**
	 * @brief Default constructor for serialization
	 */
	ACALSimSSTEvent() : packet_(nullptr) {}

	/**
	 * @brief Constructor
	 * @param packet Shared pointer to the ACALSim packet to wrap
	 */
	explicit ACALSimSSTEvent(std::shared_ptr<SimPacket> packet) : packet_(packet) {}

	/**
	 * @brief Get the wrapped ACALSim packet
	 * @return Shared pointer to the SimPacket
	 */
	std::shared_ptr<SimPacket> getPacket() const { return packet_; }

	/**
	 * @brief Set the wrapped packet
	 * @param packet New packet to wrap
	 */
	void setPacket(std::shared_ptr<SimPacket> packet) { packet_ = packet; }

	/**
	 * @brief Clone this event (required by SST)
	 * @return Cloned event
	 */
	::SST::Event* clone() override { return new ACALSimSSTEvent(packet_); }

	/**
	 * @brief Serialize event for parallel/distributed simulation
	 */
	void serialize_order(::SST::Core::Serialization::serializer& ser) override {
		Event::serialize_order(ser);
		// Note: SimPacket serialization would need to be implemented
		// for distributed SST simulations
	}

	ImplementSerializable(ACALSim::SSTIntegration::ACALSimSSTEvent);

private:
	std::shared_ptr<SimPacket> packet_;
};

/**
 * @brief SST Component wrapper for ACALSim SimBase instances
 *
 * This class wraps an ACALSim simulator (SimBase-derived) as an SST Component,
 * enabling ACALSim components to participate in SST simulations. The wrapper:
 * - Bridges ACALSim's event-driven model to SST's discrete event simulation
 * - Maps ACALSim SimPorts to SST Links
 * - Synchronizes ACALSim's internal tick with SST's simulation time
 * - Translates ACALSim configuration from SST Params
 */
class ACALSimComponent : public ::SST::Component {
public:
	/**
	 * @brief SST ELI (Element Library Information) - Component registration
	 *
	 * This macro registers the component with SST and defines its metadata
	 */
	SST_ELI_REGISTER_COMPONENT(ACALSimComponent,                       // Class name
	                           "acalsim",                              // Library name
	                           "ACALSimComponent",                     // Component name
	                           SST_ELI_ELEMENT_VERSION(1, 0, 0),       // Version
	                           "ACALSim Simulator Component Wrapper",  // Description
	                           COMPONENT_CATEGORY_PROCESSOR            // Category
	)

	/**
	 * @brief SST ELI - Parameter documentation
	 *
	 * Defines the configuration parameters accepted from Python config
	 */
	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency (e.g., '1GHz')", "1GHz"},
	                        {"simulator_type", "Type of ACALSim simulator to instantiate", "CPPSimBase"},
	                        {"config_file", "Path to ACALSim JSON configuration file", ""},
	                        {"name", "Name of this simulator instance", "acalsim0"},
	                        {"verbose", "Verbosity level (0-5)", "1"},
	                        {"max_ticks", "Maximum simulation ticks (0 = unlimited)", "0"})

	/**
	 * @brief SST ELI - Port documentation
	 *
	 * Defines the ports (links) this component can use
	 */
	SST_ELI_DOCUMENT_PORTS({"port%(num_ports)d", "ACALSim communication ports", {}})

	/**
	 * @brief Constructor
	 * @param id SST-assigned component ID
	 * @param params Configuration parameters from Python
	 */
	ACALSimComponent(::SST::ComponentId_t id, ::SST::Params& params);

	/**
	 * @brief Destructor
	 */
	~ACALSimComponent() override;

	/**
	 * @brief SST setup phase - called after all components are constructed
	 */
	void setup() override;

	/**
	 * @brief SST init phase - multi-round initialization
	 * @param phase Initialization phase number
	 */
	void init(unsigned int phase) override;

	/**
	 * @brief SST finish phase - cleanup and statistics
	 */
	void finish() override;

	/**
	 * @brief Clock handler - called every clock cycle
	 * @param cycle Current cycle number
	 * @return true if component should continue, false to end simulation
	 */
	bool clockTick(::SST::Cycle_t cycle);

private:
	/**
	 * @brief Event handler for incoming SST events on links
	 * @param ev Incoming SST event
	 */
	void handleEvent(::SST::Event* ev);

	/**
	 * @brief Initialize ACALSim simulator from SST params
	 * @param params SST configuration parameters
	 */
	void initACALSimulator(::SST::Params& params);

	/**
	 * @brief Configure SST links to map to ACALSim ports
	 * @param params SST configuration parameters
	 */
	void configureLinks(::SST::Params& params);

	/**
	 * @brief Send ACALSim packet out via SST link
	 * @param packet Packet to send
	 * @param port_id Port number to send on
	 * @param delay Delay in ACALSim ticks
	 */
	void sendPacket(std::shared_ptr<SimPacket> packet, int port_id, Tick delay = 0);

	/**
	 * @brief Process events from ACALSim simulator's outbound queue
	 */
	void processOutboundEvents();

	// SST infrastructure
	::SST::Output         out_;         ///< SST logging output
	::SST::TimeConverter* tc_;          ///< Time conversion (ACALSim ticks <-> SST time)
	std::string           clock_freq_;  ///< Clock frequency string

	// ACALSim simulator state
	// Note: Actual simulator instance should be created in derived classes
	// std::unique_ptr<SimBase> simulator_;  // Users create their own in derived classes
	std::string config_name_;   ///< Simulator configuration name
	Tick        current_tick_;  ///< Current simulation tick
	Tick        max_ticks_;     ///< Maximum ticks (0 = unlimited)

	// Port/Link mapping
	std::map<int, ::SST::Link*> links_;       ///< Map of port ID -> SST Link
	std::map<int, std::string>  port_names_;  ///< Map of port ID -> port name
	int                         num_ports_;   ///< Number of configured ports

	// Statistics
	uint64_t events_received_;  ///< Count of events received from SST
	uint64_t events_sent_;      ///< Count of events sent to SST
	uint64_t ticks_executed_;   ///< Count of ticks executed
};

/**
 * @brief Factory function to create ACALSim-based SST components
 *
 * This allows users to create custom ACALSim simulators and
 * register them as SST components.
 */
template <typename T>
class ACALSimComponentFactory {
public:
	static_assert(std::is_base_of<SimBase, T>::value, "T must derive from SimBase");

	static ::SST::Component* create(::SST::ComponentId_t id, ::SST::Params& params) {
		// Create base component
		auto* comp = new ACALSimComponent(id, params);
		// Custom simulator creation would happen here
		return comp;
	}
};

}  // namespace SSTIntegration
}  // namespace ACALSim

#endif  // __ACALSIM_SST_COMPONENT_HH__
