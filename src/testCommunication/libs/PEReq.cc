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
 * @file PEReq.cc
 * @brief PE request and response packet implementations
 *
 * This file implements PEReqPacket and PERespPacket classes, which form the
 * core communication protocol for PE computation requests in testCommunication.
 * These packets demonstrate advanced features including callback handling,
 * request-response pairing, and the visitor pattern.
 *
 * **Request-Response Protocol:**
 *
 * The PE protocol uses paired request and response packets:
 * ```
 * TrafficGenerator                         PE
 *       │                                   │
 *       │  Create PEReqPacket               │
 *       │  + PERespPacket (pre-allocated)   │
 *       │  + Callback function              │
 *       │                                   │
 *       ├──── PEReqPacket (via channel) ───►│
 *       │                                   │
 *       │                           Process request
 *       │                           Compute: d = a*b + c
 *       │                           Update PERespPacket
 *       │                                   │
 *       │◄──── Invoke callback ─────────────┤
 *       │      (immediate, not via channel) │
 *       │                                   │
 * Process response                          │
 * Extract result                            │
 * Free PERespPacket                         │
 * ```
 *
 * **Packet Roles:**
 *
 * 1. **PEReqPacket** (Request)
 *    - Carries computation parameters: a, b, c for d = a*b + c
 *    - Contains callback function for response handling
 *    - Holds pointer to pre-allocated PERespPacket
 *    - Sent via channel from TrafficGenerator to PE
 *
 * 2. **PERespPacket** (Response)
 *    - Pre-allocated by requester before sending request
 *    - Updated by PE with computation result
 *    - Passed to callback function (not sent via channel)
 *    - Freed by callback handler
 *
 * **Callback Mechanism:**
 *
 * Unlike channel-based responses, this protocol uses callbacks:
 * ```cpp
 * // TrafficEvent creates callback:
 * std::function<void(int, PERespPacket*)> callback =
 *     [this](int id, PERespPacket* pkt) {
 *         TrafficGenerator::PERespHandler(id, pkt);
 *     };
 *
 * // Store in PEReqPacket:
 * peReqPkt->setCallback(callback);
 *
 * // PE invokes callback after computation:
 * auto callback = peReqPkt->getCallback();
 * callback(transactionID, peRespPkt);
 * ```
 *
 * **Why Pre-allocate Response Packet?**
 *
 * Benefits of pre-allocation:
 * 1. **Performance**: No allocation during critical path
 * 2. **Determinism**: Memory allocation happens at request time
 * 3. **Simplicity**: Response ownership clear (requester owns)
 * 4. **Flexibility**: Requester can customize response packet
 *
 * Alternative (allocate in PE):
 * ```cpp
 * // PE would need to allocate:
 * PERespPacket* resp = new PERespPacket(reqType, resultPtr);
 * callback(tid, resp);
 * // Caller still responsible for freeing
 * ```
 *
 * **Visitor Pattern Implementation:**
 *
 * PEReqPacket implements visitor pattern for routing:
 * ```
 * PEReqPacket::visit(SimBase&)
 *   │
 *   ├─ dynamic_cast<PE*>
 *   │   └─► PE::peReqPacketHandler()
 *   │
 *   └─ else: error
 * ```
 *
 * **Actual Processing Flow:**
 *
 * In testCommunication, packets wrapped in events:
 * ```
 * 1. TrafficEvent creates PEReqPacket
 * 2. Wraps in PEReqEvent (CallbackEvent)
 * 3. Wraps in EventPacket with target tick
 * 4. Sends via channel
 * 5. PE receives EventPacket
 * 6. Framework schedules PEReqEvent
 * 7. PEReqEvent::process() handles computation
 * 8. Process invokes callback directly
 * ```
 *
 * Note: The visitor methods (visit()) exist for alternative usage patterns
 * where packets are sent directly without event wrappers.
 *
 * **Packet Type Enum:**
 *
 * PEReqTypeEnum::TEST is the only defined type in this example:
 * - Could be extended with more operation types
 * - Examples: ADD, MUL, MAC, LOAD, STORE, etc.
 * - PE could switch on request type for different operations
 *
 * **Memory Management:**
 *
 * Allocation and ownership:
 * ```cpp
 * // In TrafficEvent::process():
 * int* resultPtr = new int;  // Storage for result
 * PERespPacket* resp = new PERespPacket(TEST, resultPtr);
 * PEReqPacket* req = new PEReqPacket(TEST, a, b, c, resp);
 * // resp and resultPtr owned by resp packet
 * // req owned by event system
 *
 * // In TrafficGenerator::PERespHandler():
 * int result = pkt->getResult();  // Read result
 * free(pkt);  // Free response packet (and internally resultPtr)
 * ```
 *
 * **Design Decisions:**
 *
 * 1. **Why Callbacks Instead of Channel Responses?**
 *    - Lower latency (immediate vs next phase)
 *    - Simpler routing (no reverse channel setup)
 *    - Transaction context preserved via closure
 *    - Appropriate for single-hop communication
 *
 * 2. **When to Use Channel Responses?**
 *    - Multi-hop response path (e.g., through NOC)
 *    - Response delay modeling needed
 *    - Buffering/queuing of responses
 *    - Broadcast or multicast responses
 *
 * 3. **Why Separate Request and Response Packets?**
 *    - Clear separation of concerns
 *    - Different data in each direction
 *    - Type safety (can't confuse req/resp)
 *    - Allows async response (timing decoupled)
 *
 * **Extension Examples:**
 *
 * Adding new operation types:
 * ```cpp
 * enum class PEReqTypeEnum {
 *     TEST,      // d = a*b + c
 *     ADD,       // d = a + b
 *     MUL,       // d = a * b
 *     MAC,       // d = a*b + c (multiply-accumulate)
 *     CUSTOM     // User-defined operation
 * };
 *
 * // In PEReqEvent::process():
 * switch (peReqPkt->getReqType()) {
 *     case PEReqTypeEnum::ADD:
 *         result = a + b;
 *         break;
 *     case PEReqTypeEnum::MUL:
 *         result = a * b;
 *         break;
 *     // ...
 * }
 * ```
 *
 * @see PEReq.hh for class definitions
 * @see PEEvent.cc for actual request processing
 * @see TrafficEvent.cc for request generation
 * @see TrafficGenerator.cc for response handling
 */

#include "PEReq.hh"

#include "PE.hh"

/**
 * @brief Visit a SimModule with PEReqPacket (not implemented)
 *
 * This visitor method would handle module-level routing of PE requests.
 * Currently not implemented as the example uses flat simulator structure.
 *
 * **Potential Use Cases:**
 *
 * If PE were decomposed into modules:
 * ```
 * PE (SimBase)
 *   ├─ ALU (SimModule)
 *   ├─ FPU (SimModule)
 *   └─ LoadStoreUnit (SimModule)
 * ```
 *
 * Then requests could be routed to specific modules:
 * ```cpp
 * void PEReqPacket::visit(Tick when, SimModule& module) {
 *     if (reqType == PEReqTypeEnum::TEST) {
 *         if (auto alu = dynamic_cast<ALU*>(&module)) {
 *             alu->handlePERequest(when, this);
 *         }
 *     } else if (reqType == PEReqTypeEnum::FP_OP) {
 *         if (auto fpu = dynamic_cast<FPU*>(&module)) {
 *             fpu->handlePERequest(when, this);
 *         }
 *     }
 * }
 * ```
 *
 * @param when Simulation tick when packet arrived
 * @param module Reference to target SimModule
 *
 * @note Currently unimplemented - would enable fine-grained module routing
 */
void PEReqPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void PEReqPacket::visit(SimModule& module)is not implemented yet!";
}

/**
 * @brief Visit a SimBase (simulator) with PEReqPacket
 *
 * Implements the visitor pattern for routing PEReqPackets to PE simulators.
 * This method provides an alternative to event-based processing, allowing
 * direct packet delivery via visit().
 *
 * **Implementation:**
 * ```cpp
 * if (auto pe = dynamic_cast<PE*>(&simulator)) {
 *     pe->peReqPacketHandler(when, this);
 *     // Handler logs reception but doesn't process
 *     // Actual processing in PEReqEvent::process()
 * }
 * ```
 *
 * **Usage Pattern:**
 *
 * Direct visit (alternative to event-based):
 * ```cpp
 * // Create request packet
 * PEReqPacket* req = new PEReqPacket(TEST, a, b, c, resp);
 *
 * // Option 1: Event-based (current example)
 * PEReqEvent* event = new PEReqEvent(tid, pe, callback, req);
 * EventPacket* eventPkt = new EventPacket(event, targetTick);
 * sim->pushToMasterChannelPort("DSPE", eventPkt);
 *
 * // Option 2: Direct visit (alternative)
 * req->visit(currentTick, *peSimulator);
 * // Immediately calls PE::peReqPacketHandler()
 * ```
 *
 * **Why Event-Based Preferred:**
 *
 * testCommunication uses PEReqEvent instead of direct visit:
 * 1. **Deferred Execution**: Schedule for future tick
 * 2. **Callback Support**: Events carry callback functions
 * 3. **Framework Integration**: Automatic event queue management
 * 4. **Timing Flexibility**: Can model communication latency
 *
 * **Handler Behavior:**
 *
 * PE::peReqPacketHandler() currently only logs reception:
 * ```cpp
 * void PE::peReqPacketHandler(Tick when, SimPacket* pkt) {
 *     auto peReqPkt = dynamic_cast<PEReqPacket*>(pkt);
 *     if (peReqPkt) {
 *         CLASS_INFO << "Received PE Request Packet";
 *         // Actual computation would go here if using direct visit
 *     }
 * }
 * ```
 *
 * For direct visit usage, handler would need:
 * ```cpp
 * void PE::peReqPacketHandler(Tick when, SimPacket* pkt) {
 *     auto peReqPkt = dynamic_cast<PEReqPacket*>(pkt);
 *     if (peReqPkt) {
 *         // Extract parameters
 *         int result = peReqPkt->getA() * peReqPkt->getB() +
 *                      peReqPkt->getC();
 *         // Update response
 *         peReqPkt->getPERespPkt()->setResult(result);
 *         // Invoke callback
 *         peReqPkt->getCallback()(transactionID,
 *                                  peReqPkt->getPERespPkt());
 *     }
 * }
 * ```
 *
 * **Extensibility:**
 *
 * To support multiple simulator types:
 * ```cpp
 * void PEReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto pe = dynamic_cast<PE*>(&simulator)) {
 *         pe->peReqPacketHandler(when, this);
 *     } else if (auto dsp = dynamic_cast<DSP*>(&simulator)) {
 *         dsp->handlePERequest(when, this);
 *     } else if (auto gpu = dynamic_cast<GPU*>(&simulator)) {
 *         gpu->processComputation(when, this);
 *     } else {
 *         CLASS_ERROR << "Unsupported simulator type for PEReqPacket";
 *     }
 * }
 * ```
 *
 * @param when Simulation tick when packet should be processed
 * @param simulator Reference to target simulator
 *
 * @note In testCommunication, actual processing happens in PEReqEvent::process()
 *       This visitor method provides alternative direct-call interface
 *
 * @see PE::peReqPacketHandler() for handler implementation
 * @see PEReqEvent::process() for event-based processing
 */
void PEReqPacket::visit(Tick when, SimBase& simulator) {
	auto pe = dynamic_cast<PE*>(&simulator);
	if (pe) {
		pe->peReqPacketHandler(when, this);
	} else {
		CLASS_ERROR << "Invalid simulator type";
	}
}

/**
 * @brief Visit a SimModule with PERespPacket (not implemented)
 *
 * This visitor method would handle module-level routing of PE responses.
 * Currently not needed as responses use callbacks instead of packet routing.
 *
 * **Why Not Implemented:**
 *
 * The request-response protocol uses callbacks for responses:
 * ```
 * Request: TrafficGenerator ──packet──► PE (via channel)
 * Response: PE ──callback──► TrafficGenerator (direct call)
 * ```
 *
 * Responses don't travel via visitor pattern because:
 * - Callback mechanism is more direct
 * - No routing needed (callback knows destination)
 * - Lower overhead than packet routing
 * - Transaction context preserved in closure
 *
 * **If Response Routing Were Needed:**
 *
 * Potential implementation for channel-based responses:
 * ```cpp
 * void PERespPacket::visit(Tick when, SimModule& module) {
 *     if (auto tgModule = dynamic_cast<TrafficGenModule*>(&module)) {
 *         tgModule->handlePEResponse(when, this);
 *     } else if (auto cacheModule = dynamic_cast<CacheModule*>(&module)) {
 *         cacheModule->processPEResponse(when, this);
 *     }
 * }
 * ```
 *
 * Use case for channel-based responses:
 * ```
 * TrafficGenerator ──req──► NOC ──req──► PE
 *                   ◄─resp─ NOC ◄─resp─ PE
 * // Response needs routing through NOC back to TG
 * ```
 *
 * @param when Simulation tick when packet arrived
 * @param module Reference to target SimModule
 *
 * @note Not implemented - responses use callback mechanism instead
 *
 * @see TrafficGenerator::PERespHandler() for callback-based response handling
 */
void PERespPacket::visit(Tick when, SimModule& module) {
	CLASS_ERROR << "void PERespPacket::visit(SimModule& module) is not implemented yet!";
}

/**
 * @brief Visit a SimBase (simulator) with PERespPacket (not implemented)
 *
 * This visitor method would route response packets to simulators. Currently
 * not implemented because the protocol uses callbacks for response delivery.
 *
 * **Current Response Flow:**
 *
 * Callback-based (this example):
 * ```
 * PE::compute()
 *   └─► Update PERespPacket with result
 *       └─► Invoke callback(tid, respPkt)
 *           └─► TrafficGenerator::PERespHandler(tid, respPkt)
 * ```
 *
 * No packet routing needed:
 * - Callback function pointer directly references handler
 * - No need to traverse simulators or modules
 * - Immediate execution in same tick
 *
 * **If Packet-Based Responses Were Used:**
 *
 * Alternative implementation:
 * ```cpp
 * void PERespPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto tg = dynamic_cast<TrafficGenerator*>(&simulator)) {
 *         tg->handlePEResponse(when, this);
 *     } else if (auto cache = dynamic_cast<CacheSim*>(&simulator)) {
 *         cache->processPEResponse(when, this);
 *     } else {
 *         CLASS_ERROR << "Invalid simulator type for PERespPacket";
 *     }
 * }
 * ```
 *
 * Usage scenario:
 * ```cpp
 * // In PE, after computing result:
 * PERespPacket* resp = peReqPkt->getPERespPkt();
 * resp->setResult(computedValue);
 *
 * // Option 1: Callback (current)
 * peReqPkt->getCallback()(tid, resp);
 *
 * // Option 2: Channel-based (if visitor implemented)
 * EventPacket* eventPkt = new EventPacket(resp, tick + delay);
 * this->pushToMasterChannelPort("USTrafficGenerator", eventPkt);
 * // Would require visitor implementation for routing
 * ```
 *
 * **When to Use Packet-Based Responses:**
 *
 * Consider implementing visitor for responses when:
 * 1. Response must traverse multiple simulators
 * 2. Response delay needs explicit modeling
 * 3. Response buffering/queuing required
 * 4. Response destination not known at request time
 * 5. Response broadcast needed (multiple recipients)
 *
 * **Callback vs Packet Response Comparison:**
 *
 * Callbacks (current):
 * - Immediate execution
 * - Zero routing overhead
 * - Transaction context preserved
 * - Best for single-hop
 *
 * Packet-based:
 * - Explicit timing control
 * - Can model network delays
 * - Supports multi-hop routing
 * - Better for complex topologies
 *
 * @param when Simulation tick when packet should be processed
 * @param simulator Reference to target simulator
 *
 * @note Not implemented - use TrafficGenerator::PERespHandler callback instead
 *
 * @see PEReqEvent::process() for callback invocation
 * @see TrafficGenerator::PERespHandler() for response handling
 */
void PERespPacket::visit(Tick when, SimBase& simulator) {
	CLASS_ERROR << "void PERespPacket::visit(SimBase& simulator) is not implemented yet!";
}
