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

#ifndef __SIMPLE_PROCESSOR_HH__
#define __SIMPLE_PROCESSOR_HH__

#include "../ACALSimComponent.hh"
#include "ACALSim.hh"

/**
 * @brief Simple processor simulator for SST integration example
 *
 * This is a minimal ACALSim simulator that demonstrates:
 * - Event-driven execution
 * - Packet generation and transmission
 * - Integration with SST framework
 */
class SimpleProcessor : public SimBase {
public:
	SimpleProcessor(SimConfig* config) : SimBase(config) {
		instruction_count_ = 0;
		memory_requests_   = 0;
		max_instructions_  = config->get<uint64_t>("max_instructions", 1000);
	}

	~SimpleProcessor() override = default;

	/**
	 * @brief Initialize processor
	 */
	void init() override {
		SimBase::init();
		LOG(INFO, "SimpleProcessor initialized (max instructions: %lu)", max_instructions_);
	}

	/**
	 * @brief Execute one simulation step
	 */
	void step() override {
		// Process events in event queue
		SimBase::step();

		// Simulate instruction execution
		if (instruction_count_ < max_instructions_) {
			// Generate memory request every 10 instructions
			if (instruction_count_ % 10 == 0) { generateMemoryRequest(); }
			instruction_count_++;
		}
	}

	/**
	 * @brief Check if simulation is complete
	 */
	bool isDone() const override { return instruction_count_ >= max_instructions_; }

	/**
	 * @brief Cleanup and print statistics
	 */
	void cleanup() override {
		LOG(INFO, "SimpleProcessor statistics:");
		LOG(INFO, "  Instructions executed: %lu", instruction_count_);
		LOG(INFO, "  Memory requests sent: %lu", memory_requests_);
		SimBase::cleanup();
	}

private:
	/**
	 * @brief Generate a memory request packet
	 */
	void generateMemoryRequest() {
		// Create a memory request packet
		auto packet = std::make_shared<SimPacket>();
		packet->setType(SimPacket::MEMREQ);
		packet->setAddress(0x1000 + (memory_requests_ * 64));  // Sequential addresses
		packet->setSize(64);                                   // 64-byte cache line

		memory_requests_++;

		LOG(DEBUG, "Generated memory request #%lu (addr: 0x%lx)", memory_requests_, packet->getAddress());

		// In a real implementation, this would be sent via a channel or port
		// For SST integration, the wrapper would extract this and send via SST Link
	}

	uint64_t instruction_count_;  ///< Number of instructions executed
	uint64_t memory_requests_;    ///< Number of memory requests generated
	uint64_t max_instructions_;   ///< Maximum instructions to execute
};

/**
 * @brief SST Component wrapper for SimpleProcessor
 */
class SimpleProcessorSSTComponent : public ACALSim::SST::ACALSimComponent {
public:
	SST_ELI_REGISTER_COMPONENT(SimpleProcessorSSTComponent, "acalsim", "SimpleProcessor",
	                           SST_ELI_ELEMENT_VERSION(1, 0, 0), "Simple Processor Example for ACALSim-SST Integration",
	                           COMPONENT_CATEGORY_PROCESSOR)

	SST_ELI_DOCUMENT_PARAMS({"clock", "Clock frequency", "1GHz"},
	                        {"max_instructions", "Maximum instructions to execute", "1000"},
	                        {"verbose", "Verbosity level", "1"})

	SST_ELI_DOCUMENT_PORTS({"mem_port", "Memory interface port", {}})

	SimpleProcessorSSTComponent(::SST::ComponentId_t id, ::SST::Params& params) : ACALSimComponent(id, params) {
		// Create SimpleProcessor instance
		// This would be integrated with the base class simulator management
	}
};

#endif  // __SIMPLE_PROCESSOR_HH__
