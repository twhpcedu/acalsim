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

/**
 * @file CacheEvent.cc
 * @brief Cache request event for memory operation processing
 *
 * Implements CacheReqEvent, which forwards unwrapped memory requests to CacheSim.
 * This is the final event in the request path before cache processing.
 *
 * **Event Flow:**
 * ```
 * Tick 12: CacheReqEvent::process()
 *   → Embeds callback (NocSim::nocReqCallback)
 *   → Calls CacheSim::accept()
 *   → Triggers visitor pattern: CacheReqPacket::visit(CacheSim&)
 *   → Executes CacheSim::handleNOCRequest()
 *   → Cache invokes callback immediately (synchronous response)
 * ```
 *
 * **Callback Chain:**
 * ```
 * TrafficEvent::NocRespHandler (original)
 *   → Stored in NocReqPacket
 *   → Forwarded to CacheReqPacket as NocSim::nocReqCallback
 *   → Invoked by CacheSim after processing
 *   → NocSim forwards back to TrafficGenerator
 * ```
 *
 * @see CacheReqEvent For cache request event
 * @see CacheSim::handleNOCRequest() Processing logic
 * @see CallbackEvent For base callback management
 */

#include "CacheEvent.hh"

#include "CacheSim.hh"

/**
 * @brief Process cache request event - dispatch to CacheSim
 *
 * Invoked when cache receives request from NOC. Embeds callback into packet
 * and dispatches via visitor pattern for processing.
 *
 * **Processing Steps:**
 * 1. Verify callback exists (NocSim::nocReqCallback)
 * 2. Store callback in CacheReqPacket
 * 3. Call CacheSim::accept() to trigger visitor pattern
 * 4. Visitor invokes CacheReqPacket::visit(CacheSim&)
 * 5. Which calls CacheSim::handleNOCRequest()
 * 6. Cache immediately invokes callback with response
 *
 * **Callback Embedding:**
 * ```cpp
 * cacheReqPkt->setCallback(this->callerCallback);
 * // Stores NocSim::nocReqCallback for response delivery
 * ```
 *
 * **Synchronous Response:**
 * Unlike NOC events, cache response is delivered synchronously via callback
 * invocation (no separate CacheRespEvent needed).
 *
 * @note Uses visitor pattern for type-safe dispatch
 * @note Response delivered via callback, not channel
 * @see CacheReqPacket::visit() Visitor target
 * @see CacheSim::handleNOCRequest() Processing and response generation
 */
void CacheReqEvent::process() {
	if (this->callerCallback) {
		CLASS_INFO << "Process CacheReqEvent with transaction id: " << this->tid << " at Tick=" << top->getGlobalTick();
		cacheReqPkt->setCallback(this->callerCallback);
		((CacheSim*)callee)->accept(top->getGlobalTick(), (SimPacket&)*cacheReqPkt);
	}
}
