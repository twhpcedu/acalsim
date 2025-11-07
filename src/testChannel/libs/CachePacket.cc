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
 * @file CachePacket.cc
 * @brief Cache-level packet types implementing visitor pattern for memory operations
 *
 * Implements CacheReqPacket and CacheRespPacket with visitor pattern dispatch.
 * These packets represent unwrapped memory requests/responses at the cache layer,
 * stripped of NOC routing information.
 *
 * **Packet Hierarchy:**
 * ```
 * SimPacket (base)
 *   ├─ CacheReqPacket (PTYPE::MEMREQ)
 *   │    ├─ addr: Memory address
 *   │    ├─ size: Transfer size
 *   │    └─ callback: NocSim::nocReqCallback
 *   └─ CacheRespPacket (PTYPE::MEMRESP)
 *        └─ data: Pointer to response data
 * ```
 *
 * **Layer Unwrapping:**
 * ```
 * TrafficGenerator creates:
 *   NocReqPacket (addr, size, callback=TrafficEvent::NocRespHandler)
 *
 * NocSim unwraps to:
 *   CacheReqPacket (addr, size, callback=NocSim::nocReqCallback)
 *
 * CacheSim processes and creates:
 *   NocRespPacket (data=100)
 *
 * Response flows back through callbacks (no CacheRespPacket used)
 * ```
 *
 * @see CacheReqPacket Request packet for cache operations
 * @see CacheSim::handleNOCRequest() Visitor target
 */

#include "CachePacket.hh"

#include "CacheSim.hh"
#include "NocPacket.hh"
#include "NocSim.hh"

/**
 * @brief Visit SimModule (not implemented for CacheRespPacket)
 * @param when Tick when packet should be processed
 * @param module Target SimModule
 */
void CacheRespPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void CacheRespPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visit SimBase (not implemented for CacheRespPacket)
 *
 * CacheRespPacket is not used in the current implementation.
 * Cache responses are delivered via NocRespPacket and callbacks.
 *
 * @param when Tick when packet should be processed
 * @param simulator Target simulator
 */
void CacheRespPacket::visit(Tick when, SimBase& simulator) {
	CLASS_ERROR << "void CacheRespPacket::visit (SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Visit SimModule (not implemented for CacheReqPacket)
 * @param when Tick when packet should be processed
 * @param module Target SimModule
 */
void CacheReqPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void CacheReqPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visit SimBase - dispatch request to CacheSim for processing
 *
 * Visitor pattern entry point for CacheReqPacket. When CacheReqEvent.process()
 * calls accept() on CacheSim, this method is invoked to process the memory request.
 *
 * **Processing Flow:**
 * ```
 * 1. Dynamic cast to verify simulator is CacheSim
 * 2. If CacheSim: Call handleNOCRequest() to generate response
 * 3. If not CacheSim: Log error (packet sent to wrong simulator)
 * ```
 *
 * **Response Generation:**
 * ```cpp
 * cacheSim->handleNOCRequest(this, simulator->getID(), when)
 *   → Generates pseudo data (data=100)
 *   → Creates NocRespPacket
 *   → Invokes callback with response
 * ```
 *
 * @param when Tick when packet arrived at cache
 * @param simulator Target simulator (should be CacheSim)
 *
 * @note Uses simulator ID (not packet ID) for transaction tracking
 * @see CacheSim::handleNOCRequest() Processing logic
 * @see CacheReqEvent::process() Caller of this visitor
 */
// When CacheSim visit this packet, it will create cacheRespPkt packed in nocRespPkt
// Then executing callback to return nocRespPkt
void CacheReqPacket::visit(Tick when, SimBase& simulator) {
	if (dynamic_cast<CacheSim*>((SimBase*)(&simulator))) {
		// Return a pseudo data for test
		dynamic_cast<CacheSim*>((SimBase*)(&simulator))->handleNOCRequest(this, (&simulator)->getID(), when);
	} else {
		CLASS_INFO << "Invalid packet type!";
	}
}
