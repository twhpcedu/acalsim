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
 * @file PEEvent.cc
 * @brief PE task event processor with stress test traffic generation capability
 *
 * This file implements PERNocEvent, the **event wrapper** for PE-bound task packets in the
 * accelerator system. Beyond standard event delivery, it implements **Test 1 stress testing**
 * where each PE generates 100 additional BLACKHOLE packets to saturate the NoC and validate
 * high-throughput routing scenarios.
 *
 * **PERNocEvent Dual Role:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────┐
 * │                     PE Event Processing Flow                       │
 * │                                                                    │
 * │  Standard Operation (Test 0, Test 2):                              │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Event Delivery                                            │ │
 * │  │    ├─ PERNocEvent::process() invoked                         │ │
 * │  │    ├─ Log: "Process PERNocEvent with transaction id: X"      │ │
 * │  │    ├─ Cast callee to PESim*                                  │ │
 * │  │    └─ Deliver: peSim->accept(tick, packet)                   │ │
 * │  │                                                              │ │
 * │  │ 2. Packet Processing                                         │ │
 * │  │    └─ PESim::RNocPacketHandler() processes task             │ │
 * │  │        └─ Returns response to MCPU                           │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Stress Test Mode (Test 1 Only):                                   │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Standard Event Delivery (same as above)                   │ │
 * │  │                                                              │ │
 * │  │ 2. Traffic Generation (AFTER packet delivery)                │ │
 * │  │    ├─ Check: if (getTestNum() == 1)                          │ │
 * │  │    ├─ for (destRId = 0; destRId < 100; destRId++)            │ │
 * │  │    │    ├─ Create BLACKHOLE routing info                     │ │
 * │  │    │    ├─ Target: Cache #(destRId % 4)                      │ │
 * │  │    │    ├─ ID: (tid + 1) * 1000 + destRId                    │ │
 * │  │    │    ├─ Schedule: currentTick + destRId + 1               │ │
 * │  │    │    ├─ Create RNocPacket + RNocEvent                     │ │
 * │  │    │    └─ Push to "DSNOC" channel                           │ │
 * │  │    │                                                          │ │
 * │  │    └─ Result: 100 additional packets sent to NoC             │ │
 * │  │        └─ Distributed across 4 cache tiles                   │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Event Processing Flow (Test 0/2):**
 * ```
 * Tick 5: Framework dequeues PERNocEvent
 *   ├─ Scheduled time reached (when = 5)
 *   ├─ Framework calls: PERNocEvent::process()
 *   │
 *   └─ process() Implementation:
 *       ├─ 1. Log transaction ID:
 *       │    CLASS_INFO << "Process PERNocEvent with transaction id: 0"
 *       │
 *       ├─ 2. Cast callee to PESim*:
 *       │    PESim* peSim = (PESim*)this->callee
 *       │
 *       ├─ 3. Get global simulation tick:
 *       │    Tick currentTick = top->getGlobalTick()  // = 5
 *       │
 *       ├─ 4. Get packet from event:
 *       │    SimPacket* pkt = this->getPacket()
 *       │    // Returns PERNocPacket*
 *       │
 *       ├─ 5. Deliver to target PE:
 *       │    peSim->accept(currentTick, *pkt)
 *       │    └─ PESim::accept() invokes visitor pattern
 *       │        └─ pkt->visit(*peSim)
 *       │            └─ PERNocPacket::visit(PESim&)
 *       │                └─ peSim->RNocPacketHandler(5, pkt)
 *       │
 *       └─ 6. Check test mode:
 *            if (peSim->getTestNum() == 1) {
 *                // Stress test mode - generate additional traffic
 *            }
 *            // For Test 0/2: No additional traffic generated
 * ```
 *
 * **Stress Test Traffic Generation (Test 1):**
 * ```
 * Tick 5: PE #0 receives initial task from MCPU
 *   ├─ PERNocEvent::process() invoked
 *   ├─ Deliver packet to PE #0
 *   └─ Check: getTestNum() == 1 (TRUE)
 *       │
 *       └─ for (destRId = 0; destRId < 100; destRId++):
 *           │
 *           ├─ Iteration 0 (Cache #0):
 *           │    ├─ destDMType = BLACKHOLE
 *           │    ├─ nocSim = getDownStream("DSNOC")
 *           │    ├─ id = (0 + 1) * 1000 + 0 = 1000
 *           │    ├─ when = 5 + 0 + 1 = 6
 *           │    ├─ Create DLLRoutingInfo(UNICAST, 0 % 4 = 0, BLACKHOLE)
 *           │    ├─ Create RNocPacket scheduled for Tick 6
 *           │    ├─ Create RNocEvent(id=1000, ...)
 *           │    ├─ Push to channel: DSNOC
 *           │    └─ Log: "Issue traffic with transaction id: 1000 at Tick=5"
 *           │
 *           ├─ Iteration 1 (Cache #1):
 *           │    ├─ id = 1001, when = 7, destCache = 1 % 4 = 1
 *           │    └─ Create and send packet...
 *           │
 *           ├─ Iteration 2 (Cache #2):
 *           │    ├─ id = 1002, when = 8, destCache = 2 % 4 = 2
 *           │    └─ Create and send packet...
 *           │
 *           ... (continues for 100 iterations)
 *           │
 *           └─ Iteration 99 (Cache #3):
 *                ├─ id = 1099, when = 105, destCache = 99 % 4 = 3
 *                └─ Create and send packet...
 *
 * Result: 100 BLACKHOLE packets generated by PE #0
 *   - Spread across ticks 6-105
 *   - Distributed across 4 caches (round-robin)
 *   - Each cache receives 25 packets
 * ```
 *
 * **System-Wide Test 1 Traffic:**
 * ```
 * Total Traffic Generated:
 *   - 16 PEs each generate 100 packets
 *   - Total: 16 × 100 = 1600 BLACKHOLE packets
 *   - Plus: 16 original task packets from MCPU
 *   - Grand total: 1616 packets
 *
 * Cache Load Distribution:
 *   - 4 caches available
 *   - Each cache receives: 1600 / 4 = 400 BLACKHOLE packets
 *   - Plus: 4 normal requests from MCPU (1 per cache)
 *   - Total per cache: ~404 packets
 *
 * Timing Pattern:
 *   PE #0:  Generates packets at ticks 6-105
 *   PE #1:  Generates packets at ticks ~26-125
 *   PE #2:  Generates packets at ticks ~46-145
 *   ...
 *   PE #15: Generates packets at ticks ~306-405
 *
 * Purpose:
 *   - Stress test NoC routing capacity
 *   - Validate packet ordering under load
 *   - Test BLACKHOLE mode (no responses)
 *   - Verify event queue scalability
 * ```
 *
 * **Transaction ID Scheme (Test 1):**
 * ```
 * Original task from MCPU:
 *   tid = 0-15 (one per PE)
 *
 * Generated traffic by PE:
 *   id = (tid + 1) * 1000 + destRId
 *
 * PE #0 (tid=0):  IDs 1000-1099
 * PE #1 (tid=1):  IDs 2000-2099
 * PE #2 (tid=2):  IDs 3000-3099
 * ...
 * PE #15 (tid=15): IDs 16000-16099
 *
 * Benefits:
 *   - Non-overlapping ID ranges
 *   - Easy to identify source PE
 *   - Enables traffic tracking
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Dual-Mode Operation:**
 *    - Test 0/2: Standard event delivery only
 *    - Test 1: Event delivery + 100 packet generation
 *    - Mode checked via getTestNum()
 *
 * 2. **BLACKHOLE Routing:**
 *    - destDMType = DestDMTypeEnum::BLACKHOLE
 *    - Cache absorbs packet (no response)
 *    - Reduces response traffic overhead
 *    - Focuses stress test on routing
 *
 * 3. **Staggered Scheduling:**
 *    - when = currentTick + destRId + 1
 *    - Spreads packets across 100 ticks
 *    - Avoids single-tick congestion
 *    - Realistic traffic distribution
 *
 * 4. **Round-Robin Cache Distribution:**
 *    - destRId % 4 selects cache
 *    - Even load balancing
 *    - All caches participate
 *    - Tests multi-endpoint routing
 *
 * 5. **Logging and Traceability:**
 *    - CLASS_INFO logs each packet generation
 *    - Transaction ID tracking
 *    - Enables debugging and analysis
 *
 * **Class Structure:**
 * ```cpp
 * class PERNocEvent : public SimEvent {
 * private:
 *     int tid;              // Transaction ID
 *     std::string name;     // Debug name
 *     SimPacket* packet;    // PERNocPacket* payload
 *     void* callee;         // PESim* target
 *
 * public:
 *     // Constructor
 *     PERNocEvent(int tid, std::string name,
 *                 SimPacket* pkt, void* callee);
 *
 *     // Event processing (implemented in this file)
 *     void process() override;
 * };
 * ```
 *
 * **Usage Pattern:**
 * ```cpp
 * // Created in NocSim::execUnicast() when routing to PE
 * void NocSim::execUnicast(...) {
 *     // Create PE request packet
 *     PERNocPacket* peRNocPkt = new PERNocPacket(when + 10, ptr);
 *
 *     // Create PE request event
 *     PERNocEvent* peRNocEvent = new PERNocEvent(
 *         pktID,
 *         destName + " ReqPkt",
 *         peRNocPkt,
 *         (void*)peSim
 *     );
 *
 *     // Schedule event
 *     EventPacket* eventPkt = new EventPacket(peRNocEvent, when + 10);
 *     pushToMasterChannelPort(destName, eventPkt);
 * }
 * ```
 *
 * **Performance Impact (Test 1):**
 * ```
 * Event Queue Size:
 *   - 1616 total packets generated
 *   - Peak queue depth depends on NoC routing speed
 *   - Tests framework's event management
 *
 * Simulation Duration:
 *   - Significantly longer than Test 0
 *   - ~400+ ticks (vs ~500 for Test 0)
 *   - Dominated by packet routing time
 *
 * Memory Usage:
 *   - 1616 packet objects
 *   - 1616 event objects
 *   - Temporary increase during generation
 *   - Released after processing
 * ```
 *
 * **Related Files:**
 * - PEEvent.hh: PERNocEvent class definition
 * - PESim.cc: Target simulator receiving events
 * - PEPacket.hh: PERNocPacket definition
 * - NocSim.cc: Creates PE events, routes generated traffic
 * - CacheSim.cc: Receives BLACKHOLE traffic
 * - testAccelerator.cc: System architecture and test modes
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see PESim.cc for event reception and packet handling
 * @see NocSim.cc for event creation and routing
 * @see CacheSim.cc for BLACKHOLE traffic absorption
 * @see testAccelerator.cc for test mode configuration
 */

#include "peTile/PEEvent.hh"

#include "dataLinkLayer/DLLFrame.hh"
#include "dataLinkLayer/DLLHeader.hh"
#include "dataLinkLayer/DLLPayload.hh"
#include "dataLinkLayer/DLLRoutingInfo.hh"
#include "noc/NocEvent.hh"
#include "noc/NocPacket.hh"
#include "peTile/PESim.hh"

/**
 * @brief Process PE task event and optionally generate stress test traffic
 *
 * This method performs dual functions:
 * 1. **Standard Delivery:** Delivers PERNocPacket to PESim for task processing
 * 2. **Stress Test (Test 1):** Generates 100 additional BLACKHOLE packets to cache
 *
 * The stress test validates NoC throughput by flooding the network with fire-and-forget
 * packets, testing routing capacity and event queue management under high load.
 *
 * **Processing Steps:**
 * 1. Log transaction ID for traceability
 * 2. Cast callee to PESim* (target PE)
 * 3. Deliver packet via accept() → visitor pattern
 * 4. If Test 1 mode enabled:
 *    a. Generate 100 BLACKHOLE packets
 *    b. Round-robin distribute across 4 caches
 *    c. Stagger delivery across 100 ticks
 *    d. Assign unique transaction IDs
 *    e. Push to NoC via DSNOC channel
 *
 * **Test 1 Traffic Generation Details:**
 * - Count: 100 packets per PE
 * - Total system-wide: 16 PEs × 100 = 1600 packets
 * - Destination: Cache tiles (round-robin)
 * - Mode: BLACKHOLE (no response expected)
 * - Timing: Staggered over ticks [current+1 .. current+100]
 * - IDs: (tid+1)*1000 + [0..99]
 *
 * **Example Execution (Test 1, PE #3, tid=3):**
 * ```
 * Tick 20: PERNocEvent::process()
 *   ├─ Log: "Process PERNocEvent with transaction id: 3"
 *   ├─ Deliver packet to PESim #3
 *   ├─ getTestNum() == 1 → TRUE
 *   └─ Generate 100 packets:
 *       ├─ Packet 0:  ID=4000, Tick=21, Cache #0, BLACKHOLE
 *       ├─ Packet 1:  ID=4001, Tick=22, Cache #1, BLACKHOLE
 *       ├─ Packet 2:  ID=4002, Tick=23, Cache #2, BLACKHOLE
 *       ├─ Packet 3:  ID=4003, Tick=24, Cache #3, BLACKHOLE
 *       ...
 *       └─ Packet 99: ID=4099, Tick=120, Cache #3, BLACKHOLE
 * ```
 *
 * **Traffic Distribution (Test 1):**
 * ```
 * Cache #0: Receives packets 0, 4, 8, 12, ... 96   (25 packets per PE)
 * Cache #1: Receives packets 1, 5, 9, 13, ... 97   (25 packets per PE)
 * Cache #2: Receives packets 2, 6, 10, 14, ... 98  (25 packets per PE)
 * Cache #3: Receives packets 3, 7, 11, 15, ... 99  (25 packets per PE)
 * ```
 */
void PERNocEvent::process() {
	CLASS_INFO << "Process PERNocEvent with transaction id: " << this->tid;
	((PESim*)this->callee)->accept(top->getGlobalTick(), (SimPacket&)*this->getPacket());

	if (((PESim*)this->callee)->getTestNum() == 1) {
		for (int destRId = 0; destRId < 100; destRId++) {
			DestDMTypeEnum destDMType = DestDMTypeEnum::BLACKHOLE;
			SimBase*       nocSim     = ((PESim*)this->callee)->getDownStream("DSNOC");
			int            id         = (this->tid + 1) * 1000 + destRId;
			Tick           when       = top->getGlobalTick() + destRId + 1;

			DLLRoutingInfo* rInfo =
			    new DLLRoutingInfo(FlitTypeEnum::HEAD, TrafficTypeEnum::UNICAST, destRId % 4, destDMType);
			std::shared_ptr<SharedContainer<DLLRNocFrame>> ptr = std::make_shared<SharedContainer<DLLRNocFrame>>();
			ptr->add(id, rInfo);
			RNocPacket*  rNocPkt   = new RNocPacket(when, ptr);
			RNocEvent*   rNocEvent = new RNocEvent(id, "NOC Request Packet", rNocPkt, nocSim);
			EventPacket* eventPkt  = new EventPacket(rNocEvent, when);
			((PESim*)this->callee)->pushToMasterChannelPort("DSNOC", eventPkt);
			CLASS_INFO << "Issue traffic with transaction id: " << id << " at Tick=" << top->getGlobalTick();
		}
	}
}
