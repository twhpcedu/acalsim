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

/**
 * @file NocEvent.cc
 * @brief NOC routing events for request/response forwarding
 *
 * Implements NocReqEvent and NocRespEvent, which handle packet routing through the NOC layer.
 * These events use CallbackEvent base class for asynchronous callback management.
 *
 * **Event Types:**
 * ```
 * NocReqEvent (TG → NOC):
 *   - Wraps NocReqPacket
 *   - Stores caller's callback (TrafficEvent::NocRespHandler)
 *   - Invokes NocSim::accept() for visitor pattern dispatch
 *
 * NocRespEvent (NOC → TG):
 *   - Wraps NocRespPacket
 *   - Executes caller's callback directly
 *   - Delivers response back to TrafficGenerator
 * ```
 *
 * **Callback Chaining:**
 * ```
 * TrafficEvent creates callback → NocReqEvent stores it
 *   → NocSim preserves it in reqQueue
 *   → NocRespEvent forwards it back to TrafficGenerator
 * ```
 *
 * @see NocReqEvent For request forwarding
 * @see NocRespEvent For response delivery
 * @see CallbackEvent For base callback management
 */

#include "NocEvent.hh"

#include "CacheEvent.hh"
#include "CachePacket.hh"
#include "NocSim.hh"

/**
 * @brief Process NOC request event - forward packet to NocSim
 *
 * Invoked when NOC receives request from TrafficGenerator. Embeds caller's callback
 * into packet and dispatches via visitor pattern.
 *
 * **Processing Steps:**
 * 1. Verify callback exists
 * 2. Store callback in NocReqPacket
 * 3. Call NocSim::accept() to trigger visitor pattern
 * 4. Visitor invokes NocReqPacket::visit(NocSim&)
 * 5. Which calls NocSim::handleTGRequest()
 *
 * **Callback Preservation:**
 * ```cpp
 * nocReqPkt->setCallback(this->callerCallback);
 * // Stores TrafficEvent::NocRespHandler for later use
 * ```
 *
 * @note Uses visitor pattern for type-safe dispatch
 * @see NocReqPacket::visit() Visitor target
 * @see NocSim::handleTGRequest() Processing logic
 */
void NocReqEvent::process() {
	if (this->callerCallback) {
		CLASS_INFO << "Process NocReqEvent with transaction id: " << this->tid << " at Tick=" << top->getGlobalTick();
		// std::function<void(Tick, int, NocRespPacket*, SimBase*)> callback =
		//     [this](Tick when, int id, NocRespPacket* pkt, SimBase* sim) { this->nocReqCallback(when, this->id, pkt,
		//     sim); };
		nocReqPkt->setCallback(this->callerCallback);  // Insert this callback into downstream event with this packet
		((NocSim*)callee)->accept(top->getGlobalTick(), (SimPacket&)*nocReqPkt);
	}
}
