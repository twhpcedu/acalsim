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

#include "ACALSim.hh"

namespace test_compound {

/**
 * @brief Simple data packet representing a unit of data transfer
 *
 * Each packet represents a fixed-size data unit (e.g., 64 bytes).
 * By packing multiple packets into a CompoundPacket, users can model
 * different bandwidth requirements.
 */
class DataPacket : public acalsim::SimPacket {
public:
	static constexpr size_t PACKET_SIZE_BYTES = 64;  // Each packet = 64 bytes

	explicit DataPacket(uint32_t seqNum = 0, uint32_t producerId = 0)
	    : acalsim::SimPacket(PTYPE::DATA), seqNum_(seqNum), producerId_(producerId) {}

	~DataPacket() override = default;

	void visit(acalsim::Tick /* when */, acalsim::SimModule& /* module */) override {}
	void visit(acalsim::Tick /* when */, acalsim::SimBase& /* simulator */) override {}

	uint32_t getSeqNum() const { return seqNum_; }
	uint32_t getProducerId() const { return producerId_; }

	std::string getName() const override {
		return "DataPacket[seq=" + std::to_string(seqNum_) + ", producer=" + std::to_string(producerId_) + "]";
	}

private:
	uint32_t seqNum_;      ///< Sequence number for tracking
	uint32_t producerId_;  ///< Which producer sent this packet
};

}  // namespace test_compound
