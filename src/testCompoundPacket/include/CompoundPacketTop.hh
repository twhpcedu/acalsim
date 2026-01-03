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

#pragma once

#include "ACALSim.hh"
#include "Consumer.hh"
#include "Producer.hh"

namespace test_compound {

/**
 * @brief Top-level configuration for CompoundPacket bandwidth modeling example
 *
 * This example demonstrates how to use CompoundPacket to model different
 * bandwidth requirements in ACALSim simulations.
 */
class CompoundPacketTop : public acalsim::SimTop {
public:
	explicit CompoundPacketTop() : acalsim::SimTop() {}
	~CompoundPacketTop() override = default;

protected:
	void registerSimulators() final;
	void preSimInitSetup() final;
	void postSimInitSetup() final;
	void registerCLIArguments() final;

private:
	Producer* producer_;
	Consumer* consumer_;

	// Configurable parameters
	size_t totalPackets_    = 100;  // Total logical packets to transfer
	size_t packetsPerCycle_ = 4;    // Bandwidth: packets per CompoundPacket
	size_t queueSize_       = 4;    // Consumer queue size
};

}  // namespace test_compound
