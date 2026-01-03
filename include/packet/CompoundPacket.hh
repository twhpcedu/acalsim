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
#include <vector>

#include "packet/SimPacket.hh"

namespace acalsim {

/**
 * @brief CompoundPacket - wraps multiple packets for efficient transport through SimPort
 *
 * The MasterPort has a single-entry buffer, limiting throughput to 1 packet per cycle.
 * This class allows batching multiple logical packets into a single compound packet,
 * enabling users to model different bandwidth requirements.
 *
 * By packing N packets into a single CompoundPacket, you can effectively model
 * N units of bandwidth per cycle through a SimPort connection.
 *
 * Template parameter allows this to work with any packet type derived from SimPacket.
 *
 * Usage:
 *   // Create a compound packet
 *   auto* compound = new CompoundPacket<MyPacketType>(sourceId);
 *
 *   // Add packets to model higher bandwidth (e.g., 4 packets = 4x bandwidth)
 *   compound->addPacket(pkt1);
 *   compound->addPacket(pkt2);
 *   compound->addPacket(pkt3);
 *   compound->addPacket(pkt4);
 *
 *   // Send through MasterPort (counts as 1 cycle)
 *   masterPort->push(compound);
 *
 *   // Receiver unpacks
 *   auto packets = compound->extractPackets();
 *   for (auto* pkt : packets) { process(pkt); }
 *   delete compound;
 */
template <typename PacketType>
class CompoundPacket : public SimPacket {
public:
	/**
	 * @brief Construct an empty compound packet
	 * @param sourceId Identifier of the source (for routing/debugging)
	 * @param pktType The packet type (default: MEMRESP)
	 */
	explicit CompoundPacket(uint32_t sourceId = 0, PTYPE pktType = PTYPE::DATA)
	    : SimPacket(pktType), sourceId_(sourceId), packets_() {
		packets_.reserve(16);  // Pre-allocate for typical batch size
	}

	~CompoundPacket() override {
		// Note: We do NOT delete the contained packets here.
		// The receiver takes ownership when unpacking via extractPackets().
		// If the CompoundPacket is destroyed without unpacking,
		// the caller must handle cleanup of contained packets.
	}

	// SimPacket virtual methods (compound packets don't trigger events directly)
	void visit(Tick /* when */, SimModule& /* module */) override {}
	void visit(Tick /* when */, SimBase& /* simulator */) override {}

	/**
	 * @brief Add a packet to the compound
	 * @param packet Pointer to the packet (ownership transferred to receiver on unpack)
	 */
	void addPacket(PacketType* packet) { packets_.push_back(packet); }

	/**
	 * @brief Get the number of packets in this compound
	 * @return Number of contained packets
	 */
	size_t size() const { return packets_.size(); }

	/**
	 * @brief Check if the compound is empty
	 * @return true if no packets are contained
	 */
	bool empty() const { return packets_.empty(); }

	/**
	 * @brief Get read-only access to all packets
	 * @return Const reference to the vector of packet pointers
	 */
	const std::vector<PacketType*>& getPackets() const { return packets_; }

	/**
	 * @brief Get mutable access to all packets
	 * @return Reference to the vector of packet pointers
	 */
	std::vector<PacketType*>& getPackets() { return packets_; }

	/**
	 * @brief Extract all packets and clear the compound
	 *
	 * Transfers ownership of all packets to the caller.
	 * After this call, the compound is empty.
	 *
	 * @return Vector of packet pointers
	 */
	std::vector<PacketType*> extractPackets() {
		std::vector<PacketType*> result = std::move(packets_);
		packets_.clear();
		return result;
	}

	/**
	 * @brief Get the source identifier
	 * @return The source ID set during construction
	 */
	uint32_t getSourceId() const { return sourceId_; }

	/**
	 * @brief Get a descriptive name for this packet
	 * @return String describing the compound packet and its size
	 */
	std::string getName() const override { return "CompoundPacket[" + std::to_string(packets_.size()) + " packets]"; }

	/**
	 * @brief Check if this is a compound packet (for type identification)
	 * @return Always returns true for CompoundPacket
	 */
	bool isCompoundPacket() const { return true; }

protected:
	uint32_t                 sourceId_;  ///< Source identifier for routing/debugging
	std::vector<PacketType*> packets_;   ///< Contained packets
};

}  // namespace acalsim
