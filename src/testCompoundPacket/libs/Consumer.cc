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

#include "Consumer.hh"

namespace test_compound {

Consumer::Consumer(const std::string& name, size_t expectedPackets, size_t queueSize)
    : acalsim::CPPSimBase(name), expectedPackets_(expectedPackets), queueSize_(queueSize) {
	// Create input SlavePort in constructor
	inPort_ = this->addSlavePort("in", queueSize_);
}

void Consumer::init() { CLASS_INFO << "[Consumer] Initialized: expecting " << expectedPackets_ << " packets"; }

void Consumer::step() { receivePackets(); }

void Consumer::cleanup() {
	CLASS_INFO << "[Consumer] Cleanup: received " << receivedPackets_ << "/" << expectedPackets_ << " packets in "
	           << receivedCompounds_ << " compound packets";

	if (receivedPackets_ > 0 && receivedCompounds_ > 0) {
		double avgPacketsPerCompound = static_cast<double>(receivedPackets_) / receivedCompounds_;
		CLASS_INFO << "[Consumer] Average packets per CompoundPacket: " << avgPacketsPerCompound
		           << " (effective bandwidth: " << (avgPacketsPerCompound * DataPacket::PACKET_SIZE_BYTES)
		           << " bytes/cycle)";
	}
}

void Consumer::receivePackets() {
	// Check if there are packets to receive
	if (!inPort_->isPopValid()) { return; }

	// Pop the packet from the port
	acalsim::SimPacket* pkt = inPort_->pop();
	if (pkt == nullptr) { return; }

	// Try to cast to CompoundPacket
	auto* compound = dynamic_cast<acalsim::CompoundPacket<DataPacket>*>(pkt);
	if (compound != nullptr) {
		// Handle compound packet
		receivedCompounds_++;

		CLASS_INFO << "[Consumer] Received CompoundPacket from source " << compound->getSourceId() << " with "
		           << compound->size() << " packets at tick " << acalsim::top->getGlobalTick();

		// Extract and process individual packets
		auto packets = compound->extractPackets();
		for (DataPacket* dataPkt : packets) {
			CLASS_INFO << "[Consumer]   - Unpacked " << dataPkt->getName();
			receivedPackets_++;
			delete dataPkt;  // Clean up individual packet
		}

		delete compound;  // Clean up compound container

		if (isDone()) { CLASS_INFO << "[Consumer] All " << expectedPackets_ << " packets received!"; }
	} else {
		// Handle single DataPacket (for backward compatibility)
		auto* dataPkt = dynamic_cast<DataPacket*>(pkt);
		if (dataPkt != nullptr) {
			CLASS_INFO << "[Consumer] Received single " << dataPkt->getName() << " at tick "
			           << acalsim::top->getGlobalTick();
			receivedPackets_++;
			delete dataPkt;
		} else {
			CLASS_WARNING << "[Consumer] Received unknown packet type";
			delete pkt;
		}
	}
}

}  // namespace test_compound
