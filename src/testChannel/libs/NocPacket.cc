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
 * @file NocPacket.cc
 * @brief NOC packet types implementing visitor pattern for routing
 *
 * Implements NocReqPacket and NocRespPacket with visitor pattern dispatch.
 * These packets carry memory requests/responses through the NOC layer with
 * callback preservation for asynchronous response handling.
 *
 * **Packet Hierarchy:**
 * ```
 * SimPacket (base)
 *   ├─ NocReqPacket (PTYPE::MEMREQ)
 *   │    ├─ addr: Memory address
 *   │    ├─ size: Transfer size
 *   │    └─ callback: Response handler
 *   └─ NocRespPacket (PTYPE::MEMRESP)
 *        └─ data: Response data
 * ```
 *
 * **Visitor Pattern Dispatch:**
 * ```
 * NocReqEvent.process()
 *   → nocReqPkt.visit(when, NocSim&)
 *   → dynamic_cast<NocSim*> checks type
 *   → nocSim->handleTGRequest(this, id, when)
 * ```
 *
 * @see NocReqPacket Request packet with callback
 * @see NocRespPacket Response packet with data
 * @see NocSim::handleTGRequest() Visitor target
 */

#include "NocPacket.hh"

#include "CacheEvent.hh"
#include "CachePacket.hh"
#include "NocSim.hh"

using namespace acalsim;

/**
 * @brief Visit SimModule (not implemented for NocRespPacket)
 * @param when Tick when packet should be processed
 * @param module Target SimModule
 */
void NocRespPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void NocRespPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visit SimBase (not implemented for NocRespPacket)
 *
 * NocRespPacket uses callback-based delivery instead of visitor pattern.
 * Responses are delivered via NocRespEvent which directly invokes callbacks.
 *
 * @param when Tick when packet should be processed
 * @param simulator Target simulator
 */
void NocRespPacket::visit(Tick when, SimBase& simulator) {
	CLASS_ERROR << "void NocRespPacket::visit (SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Visit SimModule (not implemented for NocReqPacket)
 * @param when Tick when packet should be processed
 * @param module Target SimModule
 */
void NocReqPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void NocReqPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visit SimBase - dispatch request to NocSim for forwarding
 *
 * Visitor pattern entry point for NocReqPacket. When NocReqEvent.process()
 * calls accept() on NocSim, this method is invoked to dispatch the request.
 *
 * **Processing Flow:**
 * ```
 * 1. Dynamic cast to verify simulator is NocSim
 * 2. If NocSim: Call handleTGRequest() to forward to Cache
 * 3. If not NocSim: Log error (packet sent to wrong simulator)
 * ```
 *
 * **Type-Safe Dispatch:**
 * ```cpp
 * if (dynamic_cast<NocSim*>(&simulator)) {
 *     nocSim->handleTGRequest(this, this->getID(), when);
 * }
 * ```
 *
 * @param when Tick when packet arrived at NOC
 * @param simulator Target simulator (should be NocSim)
 *
 * @note Uses packet's own ID (getID()) for transaction tracking
 * @see NocSim::handleTGRequest() Forwarding logic
 * @see NocReqEvent::process() Caller of this visitor
 */
// When NocSim visit this packet, it will depacket this packet and forward payload to downstream CacheSim
void NocReqPacket::visit(Tick when, SimBase& simulator) {
	if (dynamic_cast<NocSim*>((SimBase*)(&simulator))) {
		dynamic_cast<NocSim*>((SimBase*)(&simulator))->handleTGRequest(this, this->getID(), when);
	} else {
		CLASS_ERROR << "Invalid packet type!";
	}
}
