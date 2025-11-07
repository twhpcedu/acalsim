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
 * @file CacheSim.cc
 * @brief Shared cache memory simulator for accelerator system
 *
 * This file implements CacheSim, the **shared memory subsystem** in the accelerator co-design
 * pattern. The cache handles memory access requests from the host CPU (MCPU) and serves as a
 * traffic sink for stress testing (blackhole mode). This component mirrors real-world shared
 * last-level caches (LLC) in heterogeneous computing systems.
 *
 * **CacheSim Role in System:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────┐
 * │                  CacheSim (Shared Cache Memory)                    │
 * │                     (4 instances: Cache #0 - #3)                   │
 * │                                                                    │
 * │  Primary Responsibilities:                                         │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Memory Access Handling                                    │ │
 * │  │    - Receive CacheRNocPacket from NoC                        │ │
 * │  │    - Process memory requests from MCPU                       │ │
 * │  │    - Return responses (normal mode)                          │ │
 * │  │                                                              │ │
 * │  │ 2. Blackhole Traffic Sink (Test 1)                           │ │
 * │  │    - Absorb high-throughput PE→Cache traffic                 │ │
 * │  │    - No response generation (fire-and-forget)                │ │
 * │  │    - Stress test NoC routing capacity                        │ │
 * │  │                                                              │ │
 * │  │ 3. Response Generation (Normal Mode)                         │ │
 * │  │    - Create RNocPacket response                              │ │
 * │  │    - Route back to MCPU via NoC                              │ │
 * │  │    - Fixed 10-tick response latency                          │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Communication Flow:                                               │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │           ↓ CacheRNocPacket (Memory Request)                 │ │
 * │  │     SlaveChannelPort "DSCACHEx"                              │ │
 * │  │           │                                                  │ │
 * │  │           ↓                                                  │ │
 * │  │   RNocPacketHandler()                                        │ │
 * │  │           │                                                  │ │
 * │  │           ├──→ if (blackhole): return (no response)          │ │
 * │  │           │                                                  │ │
 * │  │           └──→ else: Process request                         │ │
 * │  │                │                                             │ │
 * │  │                ↓                                             │ │
 * │  │        Create RNocPacket (Response)                          │ │
 * │  │                │                                             │ │
 * │  │                ↓                                             │ │
 * │  │      MasterChannelPort "USNOC_CACHEx"                        │ │
 * │  │                ↓ RNocPacket → NoC → MCPU                     │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Channel Ports (per Cache):                                        │
 * │    ┌────────────────────────────────────────────────────────────┐ │
 * │    │ SlaveChannelPort "DSCACHEx" ← Receives requests from NoC  │ │
 * │    │ MasterChannelPort "USNOC_CACHEx" → Sends responses to NoC │ │
 * │    └────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Request Processing Flow (RNocPacketHandler):**
 * ```
 * When NoC delivers CacheRNocPacket:
 *   ├─ CacheRNocEvent.process() invoked
 *   ├─ Calls CacheSim::accept()
 *   └─ Dispatches to CacheSim::RNocPacketHandler()
 *
 * RNocPacketHandler(when, pkt) Detailed Steps:
 *   ├─ 1. Type validation:
 *   │    dynamic_cast<CacheRNocPacket*>(pkt)
 *   │
 *   ├─ 2. Check blackhole flag:
 *   │    if (cRNocPkt->blackhole) {
 *   │        CLASS_INFO << "Receive stress test #1 event!"
 *   │        return;  // Absorb packet, no response
 *   │    }
 *   │
 *   ├─ 3. Normal mode processing:
 *   │    ├─ CLASS_INFO << "Receive CacheRNocPacket"
 *   │    ├─ Extract pktID = cRNocPkt->getID()
 *   │    └─ Build destName = "USNOC_CACHE0" (always cache #0)
 *   │
 *   ├─ 4. Get upstream NoC simulator:
 *   │    nocSim = getUpStream(destName)
 *   │
 *   ├─ 5. Create response routing info:
 *   │    DLLRoutingInfo* rInfo = new DLLRoutingInfo(
 *   │        FlitTypeEnum::HEAD,
 *   │        TrafficTypeEnum::UNICAST,
 *   │        0,                              // destRId = 0 (MCPU)
 *   │        DestDMTypeEnum::TRAFFIC_GENERATOR  // Back to host
 *   │    )
 *   │
 *   ├─ 6. Package response in shared container:
 *   │    SharedContainer<DLLRNocFrame>* ptr = new SharedContainer<...>()
 *   │    ptr->add(pktID, rInfo)
 *   │
 *   ├─ 7. Create response packet:
 *   │    RNocPacket* rNocPkt = new RNocPacket(
 *   │        currentTick + 10,  // Response scheduled 10 ticks ahead
 *   │        ptr
 *   │    )
 *   │
 *   ├─ 8. Create response event:
 *   │    RNocEvent* rNocEvent = new RNocEvent(
 *   │        pktID,
 *   │        "NOC Request Packet",
 *   │        rNocPkt,
 *   │        nocSim
 *   │    )
 *   │
 *   ├─ 9. Wrap and send via channel:
 *   │    EventPacket* eventPkt = new EventPacket(
 *   │        rNocEvent,
 *   │        currentTick + 10
 *   │    )
 *   │    pushToMasterChannelPort(destName, eventPkt)
 *   │
 *   └─ 10. Error handling:
 *        else: CLASS_ERROR << "Invalid packet type"
 * ```
 *
 * **Test Mode Behaviors:**
 * ```
 * Normal Mode (Test 0, 2):
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ - Cache receives request from MCPU (tasks 16-19)         │
 *   │ - Processes memory access (simulated)                    │
 *   │ - Returns response to MCPU after 10 ticks                │
 *   │ - Standard request-response protocol                     │
 *   └──────────────────────────────────────────────────────────┘
 *
 * Blackhole Mode (Test 1):
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ - Cache receives high-volume traffic from PEs            │
 *   │ - Each PE generates 100 requests → Cache                 │
 *   │ - CacheRNocPacket has blackhole=true flag                │
 *   │ - Cache absorbs packets (no response)                    │
 *   │ - Tests NoC throughput: 16 PEs × 100 = 1600 packets      │
 *   │ - Validates high-traffic routing scenarios               │
 *   └──────────────────────────────────────────────────────────┘
 * ```
 *
 * **Execution Timeline Example (Normal Mode, Cache #2):**
 * ```
 * Tick 90:
 *   ├─ CacheRNocPacket arrives at Cache #2
 *   ├─ CacheRNocEvent.process() invoked
 *   └─ CacheSim::RNocPacketHandler(when=90, pkt=CacheRNocPacket)
 *       ├─ Check: blackhole = false
 *       ├─ Extract pktID = 18
 *       ├─ CLASS_INFO: "Receive CacheRNocPacket"
 *       ├─ Create response routing: destRId=0, TRAFFIC_GENERATOR
 *       ├─ Create RNocPacket scheduled for Tick 100
 *       ├─ Create RNocEvent
 *       ├─ Push to MasterChannelPort "USNOC_CACHE0"
 *       └─ Response delivered to NoC
 *
 * Tick 100:
 *   └─ NoC receives response from Cache #2
 *       └─ Routes to MCPU
 *
 * Tick 110:
 *   └─ MCPU receives response
 *       └─ Triggers next task generation
 * ```
 *
 * **Execution Timeline Example (Blackhole Mode, Test 1):**
 * ```
 * Tick 50:
 *   ├─ CacheRNocPacket arrives at Cache #1
 *   ├─ CacheRNocEvent.process() invoked
 *   └─ CacheSim::RNocPacketHandler(when=50, pkt=CacheRNocPacket)
 *       ├─ Check: blackhole = true
 *       ├─ CLASS_INFO: "Receive stress test #1 event!"
 *       └─ return (no response generated)
 *
 * ... (1600 more blackhole packets absorbed)
 *
 * Cache Statistics (Test 1):
 *   - Packets received: ~1600 (16 PEs × 100)
 *   - Responses sent: 0
 *   - Purpose: NoC stress testing
 * ```
 *
 * **Multi-Cache Operation:**
 * ```
 * System with 4 Caches (default configuration):
 *
 * Cache #0: Handles task 16, PE stress traffic (Test 1)
 * Cache #1: Handles task 17, PE stress traffic (Test 1)
 * Cache #2: Handles task 18, PE stress traffic (Test 1)
 * Cache #3: Handles task 19, PE stress traffic (Test 1)
 *
 * Normal Mode (Test 0):
 *   - 4 cache access tasks from MCPU
 *   - Each cache handles 1 request
 *   - All return responses
 *
 * Stress Mode (Test 1):
 *   - 1600 PE-generated requests
 *   - Load balanced across 4 caches
 *   - ~400 packets per cache
 *   - No responses (blackhole)
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Blackhole Mode Support:**
 *    - Checked via CacheRNocPacket::blackhole flag
 *    - Enabled in Test 1 for stress testing
 *    - No response packet created when active
 *    - Reduces event queue pressure
 *
 * 2. **Response Latency Model:**
 *    - Fixed 10-tick memory access time
 *    - currentTick + 10 → response delivery
 *    - Models cache lookup and data return
 *
 * 3. **Fixed Response Routing:**
 *    - Always uses "USNOC_CACHE0" port
 *    - Regardless of which cache instance
 *    - Design choice for simplified routing
 *
 * 4. **Passive Architecture:**
 *    - No init() logic (reactive only)
 *    - Only responds to incoming requests
 *    - Similar to PESim and NocSim
 *
 * 5. **Packet Type Safety:**
 *    - dynamic_cast validates CacheRNocPacket
 *    - Type-safe packet handling
 *    - Error on invalid packet type
 *
 * **Data Structures:**
 * ```cpp
 * class CacheSim : public CPPSimBase {
 * public:
 *     // Constructor: Initialize cache
 *     CacheSim(std::string name);
 *
 *     // Initialization (currently empty)
 *     void init() override;
 *
 *     // Cleanup (currently empty)
 *     void cleanup() override;
 *
 *     // Process incoming memory requests
 *     void RNocPacketHandler(Tick when, SimPacket* pkt);
 * };
 * ```
 *
 * **CacheRNocPacket Structure:**
 * ```cpp
 * class CacheRNocPacket : public SimPacket {
 *     Tick when;                              // Delivery time
 *     SharedContainer<DLLRNocFrame>* ptr;     // Routing info
 *     bool blackhole;                         // Stress test flag
 *
 *     // Constructor
 *     CacheRNocPacket(Tick when, SharedContainer<...>* ptr, bool blackhole);
 * };
 * ```
 *
 * **Usage in System:**
 * ```cpp
 * // In TestAccTop::registerSimulators()
 * cacheSim = new CacheSim("Cache Simulator");
 * this->addSimulator(cacheSim);
 *
 * // Connect to NoC (for each cache port)
 * for (int cacheID = 0; cacheID < cacheCount; ++cacheID) {
 *     nocSim->addDownStream(cacheSim, "DSCACHE" + std::to_string(cacheID));
 *     cacheSim->addUpStream(nocSim, "USNOC_CACHE" + std::to_string(cacheID));
 * }
 *
 * // Connect channel ports
 * ChannelPortManager::ConnectPort(nocSim, cacheSim,
 *     "DSCACHE" + std::to_string(cacheID),
 *     "DSCACHE" + std::to_string(cacheID));
 * ChannelPortManager::ConnectPort(cacheSim, nocSim,
 *     "USNOC_CACHE" + std::to_string(cacheID),
 *     "USNOC_CACHE" + std::to_string(cacheID));
 * ```
 *
 * **Real-World Cache Analogy:**
 * ```
 * CacheSim           →  Shared Last-Level Cache (LLC)
 * RNocPacketHandler  →  Cache controller request processor
 * 10-tick latency    →  Cache access latency
 * Blackhole mode     →  Write buffer (fire-and-forget writes)
 * Multiple caches    →  Cache slicing / Distributed cache
 * Response packet    →  Cache fill response
 * ```
 *
 * **Performance Characteristics:**
 * ```
 * Normal Mode:
 *   - Request processing: 10 ticks
 *   - Response latency: 10 ticks
 *   - Total round-trip: 20 ticks (MCPU perspective)
 *
 * Blackhole Mode (Test 1):
 *   - Packet absorption: Immediate
 *   - No response overhead
 *   - Tests NoC saturation point
 *
 * Scalability:
 *   - 4 cache instances (configurable)
 *   - Independent operation
 *   - Load distribution via routing
 * ```
 *
 * **Related Files:**
 * - CacheEvent.cc: Event processing for cache packets
 * - testAccelerator.cc: System setup and cache instantiation
 * - MCPUSim.cc: Sends memory access requests (tasks 16-19)
 * - NocSim.cc: Routes packets to/from cache
 * - PEEvent.cc: Generates blackhole traffic in Test 1
 * - CachePacket.hh: CacheRNocPacket definition
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see CacheEvent.cc for event processing implementation
 * @see MCPUSim.cc for memory access request generation
 * @see PEEvent.cc for stress test traffic generation
 * @see testAccelerator.cc for system architecture overview
 */

#include "cache/CacheSim.hh"

#include "cache/CachePacket.hh"
#include "noc/NocEvent.hh"
#include "noc/NocPacket.hh"

void CacheSim::init() {}

void CacheSim::cleanup() {}

void CacheSim::RNocPacketHandler(Tick when, SimPacket* pkt) {
	if (dynamic_cast<CacheRNocPacket*>((SimPacket*)pkt)) {
		if (dynamic_cast<CacheRNocPacket*>((SimPacket*)pkt)->blackhole) {
			CLASS_INFO << "Receive stress test #1 event!\n";
			return;
		}
		CLASS_INFO << "Receive CacheRNocPacket";
		auto        cRNocPkt = dynamic_cast<CacheRNocPacket*>((SimPacket*)pkt);
		std::string destName = "USNOC_CACHE" + std::to_string(0);
		SimBase*    nocSim   = this->getUpStream(destName);
		int         pktID    = cRNocPkt->getID();

		DLLRoutingInfo* rInfo =
		    new DLLRoutingInfo(FlitTypeEnum::HEAD, TrafficTypeEnum::UNICAST, 0, DestDMTypeEnum::TRAFFIC_GENERATOR);
		std::shared_ptr<SharedContainer<DLLRNocFrame>> ptr = std::make_shared<SharedContainer<DLLRNocFrame>>();
		ptr->add(pktID, rInfo);
		RNocPacket*  rNocPkt   = new RNocPacket(top->getGlobalTick() + 10, ptr);
		RNocEvent*   rNocEvent = new RNocEvent(pktID, "NOC Request Packet", rNocPkt, nocSim);
		EventPacket* eventPkt  = new EventPacket(rNocEvent, top->getGlobalTick() + 10);
		this->pushToMasterChannelPort(destName, eventPkt);
	} else {
		CLASS_ERROR << "Invalid packet type";
	}
}
