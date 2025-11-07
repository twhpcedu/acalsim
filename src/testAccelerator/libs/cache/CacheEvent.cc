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
 * @brief Cache request event processor implementing event-packet delivery
 *
 * This file implements CacheRNocEvent, the **event wrapper** for cache-bound packets in the
 * accelerator system. It processes cache memory access requests routed through the NoC,
 * delivering CacheRNocPackets to CacheSim using the event-driven simulation framework.
 *
 * **CacheRNocEvent Role in Communication:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────┐
 * │                     Cache Event Processing Flow                    │
 * │                                                                    │
 * │  Event Lifecycle:                                                  │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Event Creation (in NocSim::execUnicast)                   │ │
 * │  │    ├─ CacheRNocPacket created with routing info              │ │
 * │  │    ├─ CacheRNocEvent wraps the packet                        │ │
 * │  │    ├─ EventPacket wraps event with timestamp                 │ │
 * │  │    └─ Pushed to channel port                                 │ │
 * │  │                                                              │ │
 * │  │ 2. Event Scheduling (Framework)                              │ │
 * │  │    ├─ Framework inserts event into priority queue            │ │
 * │  │    ├─ Scheduled for delivery at specified tick               │ │
 * │  │    └─ Events ordered by timestamp                            │ │
 * │  │                                                              │ │
 * │  │ 3. Event Processing (CacheRNocEvent::process)                │ │
 * │  │    ├─ Framework invokes process() at scheduled tick          │ │
 * │  │    ├─ Cast callee to CacheSim*                               │ │
 * │  │    ├─ Extract packet from event                              │ │
 * │  │    └─ Invoke CacheSim::accept(tick, packet)                  │ │
 * │  │                                                              │ │
 * │  │ 4. Packet Delivery (Visitor Pattern)                         │ │
 * │  │    ├─ CacheSim::accept() receives packet                     │ │
 * │  │    ├─ Invokes packet->visit(*this)                           │ │
 * │  │    └─ CacheRNocPacket::visit() dispatches to handler         │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Event Structure:                                                  │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ CacheRNocEvent                                               │ │
 * │  │ ├─ tid: Transaction ID                                       │ │
 * │  │ ├─ name: "DSCACHE{id} ReqPkt"                                │ │
 * │  │ ├─ packet: CacheRNocPacket*                                  │ │
 * │  │ ├─ callee: CacheSim* (target simulator)                      │ │
 * │  │ └─ process(): Event handler (implemented here)               │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Event Processing Flow:**
 * ```
 * Tick 95: Framework dequeues CacheRNocEvent
 *   ├─ Scheduled time reached (when = 95)
 *   ├─ Framework calls: CacheRNocEvent::process()
 *   │
 *   └─ process() Implementation:
 *       ├─ 1. Cast callee to CacheSim*:
 *       │    CacheSim* cacheSim = (CacheSim*)this->callee
 *       │
 *       ├─ 2. Get global simulation tick:
 *       │    Tick currentTick = top->getGlobalTick()  // = 95
 *       │
 *       ├─ 3. Get packet from event:
 *       │    SimPacket* pkt = this->getPacket()
 *       │    // Returns CacheRNocPacket*
 *       │
 *       ├─ 4. Cast packet reference:
 *       │    SimPacket& pktRef = (SimPacket&)*pkt
 *       │
 *       └─ 5. Deliver to target simulator:
 *            cacheSim->accept(currentTick, pktRef)
 *            └─ CacheSim::accept() invokes visitor pattern
 *                └─ pktRef.visit(*cacheSim)
 *                    └─ CacheRNocPacket::visit(CacheSim&)
 *                        └─ cacheSim.RNocPacketHandler(...)
 * ```
 *
 * **Event Creation Example (from NocSim.cc):**
 * ```cpp
 * // In NocSim::execUnicast() - Routing to Cache
 * void NocSim::execUnicast(...) {
 *     // Create cache request packet
 *     CacheRNocPacket* cRNocPkt = new CacheRNocPacket(
 *         when + 10,              // Delivery time
 *         rNocPkt->getSharedContainer(),  // Routing info
 *         blackhole               // Stress test flag
 *     );
 *
 *     // Create cache request event
 *     CacheRNocEvent* cRNocEvent = new CacheRNocEvent(
 *         pktID,                  // Transaction ID
 *         destName + " ReqPkt",   // "DSCACHE2 ReqPkt"
 *         cRNocPkt,               // Packet payload
 *         (void*)cacheSim         // Target simulator
 *     );
 *
 *     // Wrap in EventPacket for channel delivery
 *     EventPacket* eventPkt = new EventPacket(
 *         cRNocEvent,             // Event to schedule
 *         top->getGlobalTick() + 10  // When to deliver
 *     );
 *
 *     // Send via channel port
 *     this->pushToMasterChannelPort(destName, eventPkt);
 * }
 * ```
 *
 * **Event Execution Timeline:**
 * ```
 * Tick 85: NoC creates CacheRNocEvent
 *   ├─ Event created with when = 95
 *   ├─ EventPacket scheduled for Tick 95
 *   └─ Pushed to channel: DSCACHE2
 *
 * Tick 85-94: Event queued in framework
 *   └─ Waiting in priority queue
 *
 * Tick 95: Event scheduled time reached
 *   ├─ Framework dequeues CacheRNocEvent
 *   ├─ Calls: CacheRNocEvent::process()
 *   │    └─ ((CacheSim*)callee)->accept(95, packet)
 *   │
 *   └─ CacheSim receives packet
 *       └─ RNocPacketHandler(95, CacheRNocPacket)
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Event-Packet Separation:**
 *    - CacheRNocEvent: Scheduling and delivery mechanism
 *    - CacheRNocPacket: Payload and data
 *    - Clean separation of concerns
 *
 * 2. **Type-Safe Casting:**
 *    - callee stored as void* (generic)
 *    - Cast to CacheSim* in process()
 *    - Assumes caller provides correct type
 *
 * 3. **Global Tick Access:**
 *    - top->getGlobalTick() provides current simulation time
 *    - Passed to accept() for timing context
 *    - Enables time-aware processing
 *
 * 4. **Visitor Pattern Integration:**
 *    - accept() triggers visitor pattern
 *    - Enables type-safe packet dispatch
 *    - Decouples event from packet handling
 *
 * 5. **Minimal Implementation:**
 *    - Single-line process() method
 *    - Delegates all logic to CacheSim
 *    - Pure delivery mechanism
 *
 * **Class Hierarchy:**
 * ```cpp
 * // Base event class (framework-provided)
 * class SimEvent {
 *     virtual void process() = 0;  // Must override
 * };
 *
 * // Cache-specific event (defined in CacheEvent.hh)
 * class CacheRNocEvent : public SimEvent {
 * private:
 *     int tid;              // Transaction ID
 *     std::string name;     // Debug name
 *     SimPacket* packet;    // Payload (CacheRNocPacket*)
 *     void* callee;         // Target simulator (CacheSim*)
 *
 * public:
 *     // Constructor
 *     CacheRNocEvent(int tid, std::string name,
 *                    SimPacket* pkt, void* callee);
 *
 *     // Event processing (implemented in this file)
 *     void process() override;
 *
 *     // Accessors
 *     SimPacket* getPacket();
 * };
 * ```
 *
 * **Usage Pattern:**
 * ```cpp
 * // 1. Create packet with data
 * CacheRNocPacket* pkt = new CacheRNocPacket(...);
 *
 * // 2. Wrap in event for delivery
 * CacheRNocEvent* event = new CacheRNocEvent(id, name, pkt, target);
 *
 * // 3. Schedule event for future time
 * EventPacket* eventPkt = new EventPacket(event, when);
 *
 * // 4. Send via channel
 * pushToMasterChannelPort(portName, eventPkt);
 *
 * // 5. Framework schedules and processes
 * // ... (automatic) ...
 *
 * // 6. At scheduled time: process() called
 * //    └─ Delivers to target simulator
 * ```
 *
 * **Comparison with Other Events:**
 * ```
 * Event Type       Target Simulator    Packet Type        Use Case
 * ────────────────────────────────────────────────────────────────────
 * CacheRNocEvent → CacheSim           CacheRNocPacket    Cache requests
 * PERNocEvent    → PESim              PERNocPacket       PE task requests
 * RNocEvent      → NocSim             RNocPacket         Generic NoC routing
 * MCPUEvent      → MCPUSim            MCPUPacket         Response to host
 * ```
 *
 * **Related Files:**
 * - CacheEvent.hh: CacheRNocEvent class definition
 * - CacheSim.cc: Target simulator receiving events
 * - CachePacket.hh: CacheRNocPacket definition
 * - NocSim.cc: Creates and sends cache events
 * - testAccelerator.cc: System architecture overview
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see CacheSim.cc for event reception and packet handling
 * @see NocSim.cc for event creation logic
 * @see CacheEvent.hh for class declaration
 */

#include "cache/CacheEvent.hh"

#include "cache/CacheSim.hh"

/**
 * @brief Process cache request event by delivering packet to CacheSim
 *
 * This method is invoked by the simulation framework when the event's scheduled
 * time is reached. It casts the callee to CacheSim and delivers the packet via
 * the accept() method, which triggers the visitor pattern for type-safe dispatch.
 *
 * **Processing Steps:**
 * 1. Cast generic callee (void*) to CacheSim*
 * 2. Get current global simulation tick from top
 * 3. Extract packet from event wrapper
 * 4. Invoke CacheSim::accept(tick, packet)
 * 5. Accept triggers visitor: packet->visit(*cacheSim)
 * 6. Visitor dispatches to: CacheSim::RNocPacketHandler()
 *
 * **Example Execution:**
 * ```
 * // Framework calls at scheduled tick
 * Tick 95: process() invoked
 *   └─ ((CacheSim*)callee)->accept(95, *packet)
 *       └─ CacheSim::accept(95, CacheRNocPacket&)
 *           └─ packet.visit(*this)
 *               └─ CacheSim::RNocPacketHandler(95, packet)
 * ```
 */
void CacheRNocEvent::process() {
	((CacheSim*)this->callee)->accept(top->getGlobalTick(), (SimPacket&)*this->getPacket());
}
