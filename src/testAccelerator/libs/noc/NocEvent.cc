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
 * @file NocEvent.cc
 * @brief NoC routing event processor for inter-component packet delivery
 *
 * This file implements RNocEvent, the **routing event wrapper** for NoC-bound packets in the
 * accelerator system. It delivers RNocPackets to NocSim for routing decisions, enabling
 * communication between MCPU, PEs, and Cache through the central NoC fabric.
 *
 * **RNocEvent Role in System:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────┐
 * │                      NoC Event Processing Flow                     │
 * │                                                                    │
 * │  Event Sources (Multiple):                                         │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. MCPU Task Generation                                      │ │
 * │  │    - MCPUSim creates RNocEvent                               │ │
 * │  │    - Contains task routing info                              │ │
 * │  │    - Destination: PE or Cache                                │ │
 * │  │                                                              │ │
 * │  │ 2. PE Response Return                                        │ │
 * │  │    - PESim creates RNocEvent                                 │ │
 * │  │    - Contains result routing info                            │ │
 * │  │    - Destination: MCPU (TRAFFIC_GENERATOR)                   │ │
 * │  │                                                              │ │
 * │  │ 3. Cache Response Return                                     │ │
 * │  │    - CacheSim creates RNocEvent                              │ │
 * │  │    - Contains memory data routing                            │ │
 * │  │    - Destination: MCPU                                       │ │
 * │  │                                                              │ │
 * │  │ 4. PE Stress Traffic (Test 1)                                │ │
 * │  │    - PEEvent creates RNocEvent (100x per PE)                 │ │
 * │  │    - Blackhole traffic to Cache                              │ │
 * │  │    - Tests NoC throughput                                    │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Event Processing:                                                 │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ RNocEvent::process()                                         │ │
 * │  │    ├─ Cast callee to NocSim*                                 │ │
 * │  │    ├─ Get current tick from framework                        │ │
 * │  │    ├─ Extract RNocPacket from event                          │ │
 * │  │    └─ Deliver: nocSim->accept(tick, packet)                  │ │
 * │  │        └─ Triggers: packet->visit(*nocSim)                   │ │
 * │  │            └─ Invokes: RNocPacket::visit(NocSim&)            │ │
 * │  │                └─ Calls: nocSim->RNocPacketHandler()         │ │
 * │  │                    └─ Routes to destination                  │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Event Processing Flow:**
 * ```
 * Tick 5: Framework dequeues RNocEvent
 *   ├─ Scheduled time reached (when = 5)
 *   ├─ Framework calls: RNocEvent::process()
 *   │
 *   └─ process() Implementation:
 *       ├─ 1. Cast callee to NocSim*:
 *       │    NocSim* nocSim = (NocSim*)this->callee
 *       │
 *       ├─ 2. Get global simulation tick:
 *       │    Tick currentTick = top->getGlobalTick()  // = 5
 *       │
 *       ├─ 3. Get packet from event:
 *       │    SimPacket* pkt = this->getPacket()
 *       │    // Returns RNocPacket*
 *       │
 *       ├─ 4. Cast packet reference:
 *       │    SimPacket& pktRef = (SimPacket&)*pkt
 *       │
 *       └─ 5. Deliver to NoC simulator:
 *            nocSim->accept(currentTick, pktRef)
 *            └─ NocSim::accept() invokes visitor pattern
 *                └─ pktRef.visit(*nocSim)
 *                    └─ RNocPacket::visit(NocSim&)
 *                        └─ nocSim.RNocPacketHandler(5, packet)
 *                            └─ Route packet based on destination
 * ```
 *
 * **Event Creation Examples:**
 *
 * **Example 1: MCPU Task Offload (MCPUSim.cc)**
 * ```cpp
 * // MCPU sends task to PE
 * void MCPUSim::genTraffic(uint32_t destRId, DestDMTypeEnum destDMType) {
 *     SimBase* nocSim = getDownStream("DSNOC");
 *     Tick when = top->getGlobalTick() + 5;
 *
 *     // Create routing info
 *     DLLRoutingInfo* rInfo = new DLLRoutingInfo(
 *         FlitTypeEnum::HEAD,
 *         TrafficTypeEnum::UNICAST,
 *         destRId,       // PE #0-15 or Cache #0-3
 *         destDMType     // PE or CACHE
 *     );
 *
 *     // Package in shared container
 *     auto ptr = std::make_shared<SharedContainer<DLLRNocFrame>>();
 *     ptr->add(transactionID, rInfo);
 *
 *     // Create NoC packet
 *     RNocPacket* rNocPkt = new RNocPacket(when, ptr);
 *
 *     // Create NoC routing event
 *     RNocEvent* rNocEvent = new RNocEvent(
 *         transactionID,          // ID
 *         "NOC Request Packet",   // Name
 *         rNocPkt,                // Packet
 *         nocSim                  // Target
 *     );
 *
 *     // Schedule event
 *     EventPacket* eventPkt = new EventPacket(rNocEvent, when);
 *     pushToMasterChannelPort("DSNOC", eventPkt);
 * }
 * ```
 *
 * **Example 2: PE Response Return (PESim.cc)**
 * ```cpp
 * // PE returns result to MCPU
 * void PESim::RNocPacketHandler(Tick when, SimPacket* pkt) {
 *     std::string destName = "USNOC_PE" + std::to_string(peID);
 *     SimBase* nocSim = getUpStream(destName);
 *
 *     // Create response routing (back to MCPU)
 *     DLLRoutingInfo* rInfo = new DLLRoutingInfo(
 *         FlitTypeEnum::HEAD,
 *         TrafficTypeEnum::UNICAST,
 *         0,                              // destRId = 0 (MCPU)
 *         DestDMTypeEnum::TRAFFIC_GENERATOR
 *     );
 *
 *     auto ptr = std::make_shared<SharedContainer<DLLRNocFrame>>();
 *     ptr->add(pktID, rInfo);
 *
 *     RNocPacket* rNocPkt = new RNocPacket(when + 10, ptr);
 *
 *     // Create response event
 *     RNocEvent* rNocEvent = new RNocEvent(
 *         pktID,
 *         "NOC Request Packet",
 *         rNocPkt,
 *         nocSim
 *     );
 *
 *     EventPacket* eventPkt = new EventPacket(rNocEvent, when + 10);
 *     pushToMasterChannelPort(destName, eventPkt);
 * }
 * ```
 *
 * **Event Execution Timeline:**
 * ```
 * Tick 5: MCPU creates RNocEvent
 *   ├─ Event scheduled for Tick 5
 *   ├─ EventPacket pushed to channel: DSNOC
 *   └─ Framework inserts into event queue
 *
 * Tick 5: Framework processes event
 *   ├─ RNocEvent::process() invoked
 *   ├─ ((NocSim*)callee)->accept(5, RNocPacket&)
 *   └─ NocSim::RNocPacketHandler(5, packet)
 *       ├─ Extract routing info: destRId=0, destDMType=PE
 *       ├─ Route decision: execUnicast()
 *       ├─ Create PERNocPacket
 *       └─ Schedule delivery to PE #0 at Tick 15
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Universal Router Event:**
 *    - Used by all components (MCPU, PE, Cache)
 *    - Generic routing to any destination
 *    - NocSim determines final target
 *
 * 2. **Visitor Pattern Dispatch:**
 *    - accept() triggers packet->visit()
 *    - RNocPacket::visit(NocSim&) called
 *    - Type-safe routing handler invocation
 *
 * 3. **Minimal Process Implementation:**
 *    - Single-line method
 *    - Pure event delivery mechanism
 *    - All routing logic in NocSim
 *
 * 4. **Bidirectional Communication:**
 *    - Handles both requests and responses
 *    - Same event type for all directions
 *    - Routing info determines flow
 *
 * 5. **High-Volume Support:**
 *    - Used for stress testing (Test 1)
 *    - 1600+ events in stress mode
 *    - Efficient event queue management
 *
 * **Class Structure:**
 * ```cpp
 * class RNocEvent : public SimEvent {
 * private:
 *     int tid;              // Transaction ID
 *     std::string name;     // "NOC Request Packet"
 *     SimPacket* packet;    // RNocPacket* payload
 *     void* callee;         // NocSim* target
 *
 * public:
 *     // Constructor
 *     RNocEvent(int tid, std::string name,
 *               SimPacket* pkt, void* callee);
 *
 *     // Event processing (implemented here)
 *     void process() override;
 * };
 * ```
 *
 * **Routing Flow Summary:**
 * ```
 * Source      RNocEvent      NoC Router    Destination
 * ───────────────────────────────────────────────────────
 * MCPU    →   RNocEvent  →   NocSim   →   PE #x
 * MCPU    →   RNocEvent  →   NocSim   →   Cache #x
 * PE #x   →   RNocEvent  →   NocSim   →   MCPU
 * Cache #x→   RNocEvent  →   NocSim   →   MCPU
 * PE #x   →   RNocEvent  →   NocSim   →   Cache #x (Test 1)
 * ```
 *
 * **Related Files:**
 * - NocEvent.hh: RNocEvent class definition
 * - NocSim.cc: Target router receiving events
 * - NocPacket.cc: RNocPacket visitor implementation
 * - MCPUSim.cc: Creates task routing events
 * - PESim.cc: Creates response routing events
 * - CacheSim.cc: Creates response routing events
 * - PEEvent.cc: Creates stress test routing events
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see NocSim.cc for routing logic and packet handling
 * @see NocPacket.cc for visitor pattern implementation
 * @see testAccelerator.cc for system architecture overview
 */

#include "noc/NocEvent.hh"

#include "noc/NocSim.hh"

using namespace acalsim;

/**
 * @brief Process NoC routing event by delivering packet to NocSim
 *
 * This method is invoked by the simulation framework when the event reaches its
 * scheduled time. It delivers the RNocPacket to NocSim for routing decisions,
 * which determines the final destination based on routing information.
 *
 * **Processing Steps:**
 * 1. Cast generic callee (void*) to NocSim*
 * 2. Get current global tick from framework (top->getGlobalTick())
 * 3. Extract RNocPacket from event wrapper
 * 4. Invoke NocSim::accept(tick, packet)
 * 5. Accept triggers visitor: packet->visit(*nocSim)
 * 6. Visitor dispatches to: NocSim::RNocPacketHandler()
 * 7. NoC routes packet to final destination (PE/Cache/MCPU)
 *
 * **Example Execution:**
 * ```
 * Tick 5: process() invoked by framework
 *   └─ ((NocSim*)callee)->accept(5, *packet)
 *       └─ NocSim::accept(5, RNocPacket&)
 *           └─ packet.visit(*this)
 *               └─ RNocPacket::visit(NocSim&)
 *                   └─ nocSim->RNocPacketHandler(5, packet)
 *                       └─ execUnicast() routes to destination
 * ```
 */
void RNocEvent::process() { ((NocSim*)this->callee)->accept(top->getGlobalTick(), (SimPacket&)*this->getPacket()); }
