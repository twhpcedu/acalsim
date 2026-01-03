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
 * @brief Producer that generates data packets using CompoundPacket for bandwidth modeling
 *
 * The producer demonstrates how to use CompoundPacket to model different bandwidth
 * requirements. By setting packetsPerCycle, you control how many logical packets
 * are packed into each compound packet sent per cycle.
 *
 * Example bandwidth modeling:
 * - packetsPerCycle = 1: 64 bytes/cycle (single packet)
 * - packetsPerCycle = 4: 256 bytes/cycle (4x bandwidth)
 * - packetsPerCycle = 8: 512 bytes/cycle (8x bandwidth)
 */
class Producer : public acalsim::CPPSimBase {
public:
	/**
	 * @brief Construct a Producer
	 * @param name Producer name
	 * @param id Producer ID for packet tagging
	 * @param totalPackets Total number of logical packets to send
	 * @param packetsPerCycle Number of packets to pack per CompoundPacket (bandwidth)
	 */
	explicit Producer(const std::string& name, uint32_t id, size_t totalPackets, size_t packetsPerCycle);

	~Producer() override = default;

	void init() final;
	void step() final;
	void cleanup() final;
	void masterPortRetry(acalsim::MasterPort* port) final;

	bool   isDone() const { return sentPackets_ >= totalPackets_; }
	size_t getSentPackets() const { return sentPackets_; }

private:
	void trySendCompound();

	acalsim::MasterPort* outPort_;

	uint32_t id_;
	size_t   totalPackets_;
	size_t   packetsPerCycle_;  // Bandwidth: packets per compound
	size_t   sentPackets_  = 0;
	uint32_t nextSeqNum_   = 0;
	bool     retryPending_ = false;

	// Pending compound packet for retry
	acalsim::CompoundPacket<DataPacket>* pendingCompound_ = nullptr;
};

}  // namespace test_compound
