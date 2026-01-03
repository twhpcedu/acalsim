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

#pragma once

#include <string>

#include "ACALSim.hh"
#include "DataPacket.hh"

namespace test_compound {

/**
 * @brief Consumer that receives data packets via CompoundPacket
 *
 * The consumer receives CompoundPackets from producers and unpacks them
 * to process individual data packets. It tracks statistics to demonstrate
 * the bandwidth modeling.
 */
class Consumer : public acalsim::CPPSimBase {
public:
	/**
	 * @brief Construct a Consumer
	 * @param name Consumer name
	 * @param expectedPackets Expected total number of logical packets
	 * @param queueSize SlavePort queue size
	 */
	explicit Consumer(const std::string& name, size_t expectedPackets, size_t queueSize);

	~Consumer() override = default;

	void init() final;
	void step() final;
	void cleanup() final;
	void masterPortRetry(acalsim::MasterPort* port) final {}

	bool   isDone() const { return receivedPackets_ >= expectedPackets_; }
	size_t getReceivedPackets() const { return receivedPackets_; }
	size_t getReceivedCompounds() const { return receivedCompounds_; }

private:
	void receivePackets();

	acalsim::SlavePort* inPort_;

	size_t expectedPackets_;
	size_t queueSize_;
	size_t receivedPackets_   = 0;
	size_t receivedCompounds_ = 0;
};

}  // namespace test_compound
