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

#include "hw/CrossBar.hh"

namespace acalsim {
namespace crossbar {

// Function to retry operations on a specific master port
// @param port_name The name of the port to retry operations on
void CrossBar::masterPortRetry(MasterPort* port) {
	// Can improve simulation efficiency by only redoing the demux for the specific port
	// TODO: Consider implementing port-specific retry logic
	this->demux("Resp", true);
	this->demux("Req", true);
	isRetry = true;
}

// Advance the CrossBar simulation by one step
// This performs demultiplexing on both response and request channels
void CrossBar::step() {
	if (!isRetry) {
		this->demux("Resp", false);  // Process response channel
		this->demux("Req", false);   // Process request channel
	} else {
		isRetry = false;
	}
}

// Handle demultiplexing and forwarding of packets through the crossbar
// @param cname The channel name to demultiplex ("Resp" or "Req")
void CrossBar::demux(const std::string& cname, bool release_pressure) {
	// Process each master port in the crossbar
	for (int i = 0; i < this->channels[cname]->getNMasters(); i++) {
		// For each master, pop from the pipeline register and forward to the outbound master port
		SimPipeRegister* src_reg = this->getPipeRegister(cname, i);

		if (release_pressure) { src_reg->clearStallFlag(); }

		// Skip if the register doesn't contain valid data
		if (!src_reg || !src_reg->isValid()) { continue; }

		// Extract the packet from the register and determine its destination
		if (auto* packet = dynamic_cast<CrossBarPacket*>(src_reg->value())) {
			// Get the destination slave index from the packet
			auto sIdx = packet->getDstIdx();

			// Get the master port that connects to the destination slave
			auto mpToSlave = this->getMasterPortsBySlave(cname, sIdx)[i];

			// If the slave is ready to receive data, forward the packet
			if (mpToSlave->isPushReady()) {
				// Forward to the outbound master port
				// The packet will be arbitrated in phase 2
				mpToSlave->push(src_reg->pop());
			} else {
				// Backpressure from the slave side
				// Source register content remains intact
				src_reg->setStallFlag();
			}
		}
	}
}

}  // namespace crossbar
}  // namespace acalsim
