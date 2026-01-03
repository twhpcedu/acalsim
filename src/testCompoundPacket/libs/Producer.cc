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

#include "Producer.hh"

namespace test_compound {

Producer::Producer(const std::string& name, uint32_t id, size_t totalPackets, size_t packetsPerCycle)
    : acalsim::CPPSimBase(name), id_(id), totalPackets_(totalPackets), packetsPerCycle_(packetsPerCycle) {
	// Create output MasterPort in constructor
	outPort_ = this->addMasterPort("out");
}

void Producer::init() {
	CLASS_INFO << "[Producer] Initialized: id=" << id_ << ", totalPackets=" << totalPackets_
	           << ", packetsPerCycle=" << packetsPerCycle_
	           << " (bandwidth: " << (packetsPerCycle_ * DataPacket::PACKET_SIZE_BYTES) << " bytes/cycle)";
}

void Producer::step() {
	if (isDone() && pendingCompound_ == nullptr) { return; }

	trySendCompound();
}

void Producer::cleanup() {
	CLASS_INFO << "[Producer] Cleanup: sent " << sentPackets_ << "/" << totalPackets_ << " packets";

	// Clean up pending compound if any
	if (pendingCompound_ != nullptr) {
		for (auto* pkt : pendingCompound_->getPackets()) { delete pkt; }
		delete pendingCompound_;
		pendingCompound_ = nullptr;
	}
}

void Producer::masterPortRetry(acalsim::MasterPort* /* port */) {
	retryPending_ = true;
	trySendCompound();
}

void Producer::trySendCompound() {
	// If we have a pending compound from a failed push, try to send it first
	if (pendingCompound_ != nullptr) {
		if (outPort_->isPushReady()) {
			bool success = outPort_->push(pendingCompound_);
			if (success) {
				CLASS_INFO << "[Producer] Retry: sent pending CompoundPacket with " << pendingCompound_->size()
				           << " packets at tick " << acalsim::top->getGlobalTick();
				sentPackets_ += pendingCompound_->size();
				pendingCompound_ = nullptr;
				retryPending_    = false;
			}
		}
		return;
	}

	// Check if we have more packets to send
	if (sentPackets_ >= totalPackets_) { return; }

	// Check if port is ready
	if (!outPort_->isPushReady()) { return; }

	// Create a new CompoundPacket
	auto* compound = new acalsim::CompoundPacket<DataPacket>(id_);

	// Pack multiple packets into the compound (up to remaining packets)
	size_t packetsToSend = std::min(packetsPerCycle_, totalPackets_ - sentPackets_);
	for (size_t i = 0; i < packetsToSend; ++i) {
		auto* pkt = new DataPacket(nextSeqNum_++, id_);
		compound->addPacket(pkt);
	}

	// Try to send
	bool success = outPort_->push(compound);
	if (success) {
		CLASS_INFO << "[Producer] Sent CompoundPacket with " << compound->size() << " packets (seq "
		           << (nextSeqNum_ - packetsToSend) << "-" << (nextSeqNum_ - 1) << ") at tick "
		           << acalsim::top->getGlobalTick()
		           << ", bandwidth: " << (compound->size() * DataPacket::PACKET_SIZE_BYTES) << " bytes/cycle";
		sentPackets_ += compound->size();
	} else {
		// Store for retry
		pendingCompound_ = compound;
		CLASS_INFO << "[Producer] Backpressure: storing CompoundPacket with " << compound->size()
		           << " packets for retry";
	}
}

}  // namespace test_compound
