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
 * @file NocSim.cc
 * @brief Network-on-Chip (NoC) router implementing packet routing between system components
 *
 * This file implements NocSim, the **central routing fabric** in the accelerator co-design
 * pattern. The NoC acts as the communication backbone, routing packets between the host CPU
 * (MCPU), Processing Elements (PEs), and Cache tiles. It supports unicast, multicast, and
 * broadcast routing patterns, enabling flexible inter-component communication.
 *
 * **NocSim Role in System:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────┐
 * │                    NocSim (Network-on-Chip Router)                 │
 * │                                                                    │
 * │  Primary Responsibilities:                                         │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Packet Reception                                          │ │
 * │  │    - Receive RNocPackets from any source                     │ │
 * │  │    - Extract routing information                             │ │
 * │  │    - Determine traffic type (UNICAST/MULTICAST/BROADCAST)    │ │
 * │  │                                                              │ │
 * │  │ 2. Routing Decision                                          │ │
 * │  │    - Read destination type (PE/CACHE/TRAFFIC_GENERATOR)      │ │
 * │  │    - Read destination ID (destRId)                           │ │
 * │  │    - Select appropriate downstream channel                   │ │
 * │  │                                                              │ │
 * │  │ 3. Packet Forwarding                                         │ │
 * │  │    - Create destination-specific packet wrapper              │ │
 * │  │    - Add routing latency (10 ticks)                          │ │
 * │  │    - Push to target's SlaveChannelPort                       │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Routing Topology:                                                 │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │                        MCPU                                  │ │
 * │  │                          ↕                                   │ │
 * │  │                    [USMCPU/DSNOC]                            │ │
 * │  │                          ↕                                   │ │
 * │  │            ┌─────────────┼─────────────┐                    │ │
 * │  │            │             │             │                    │ │
 * │  │        [DSPEx]      [DSCACHEx]     [USPE_NOCx]              │ │
 * │  │            ↓             ↓             ↑                    │ │
 * │  │         PE Tiles      Cache Tiles   PE Responses            │ │
 * │  │         (0-15)          (0-3)        (0-15)                 │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Channel Ports (Multiple):                                         │
 * │    ┌────────────────────────────────────────────────────────────┐ │
 * │    │ Slave:  "USMCPU" ← MCPU requests                          │ │
 * │    │ Master: "USMCPU" → MCPU responses                         │ │
 * │    │                                                            │ │
 * │    │ Master: "DSPEx" → PE #x task requests (x=0..15)           │ │
 * │    │ Slave:  "USPE_NOCx" ← PE #x responses (x=0..15)           │ │
 * │    │ Master: "USPE_NOCx" → Forward PE responses                │ │
 * │    │ Slave:  "DSNOCx" ← PE additional traffic                  │ │
 * │    │                                                            │ │
 * │    │ Master: "DSCACHEx" → Cache #x requests (x=0..3)           │ │
 * │    │ Slave:  "USNOC_CACHEx" ← Cache #x responses (x=0..3)      │ │
 * │    └────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Packet Routing Flow (RNocPacketHandler → execUnicast):**
 * ```
 * When RNocPacket arrives from any source:
 *   ├─ RNocEvent.process() invoked
 *   ├─ Calls NocSim::accept()
 *   └─ Dispatches to NocSim::RNocPacketHandler()
 *
 * RNocPacketHandler(when, pkt) Logic:
 *   ├─ 1. Type validation:
 *   │    dynamic_cast<RNocPacket*>(pkt)
 *   │
 *   ├─ 2. Extract shared container:
 *   │    SharedContainer<DLLRNocFrame>* ptr = rNocPkt->getSharedContainer()
 *   │
 *   ├─ 3. Extract routing information:
 *   │    DLLRoutingInfo* rInfo = ptr->run(0, &DLLRNocFrame::getRoutingInfo)
 *   │
 *   ├─ 4. Determine traffic type:
 *   │    switch (rInfo->getTrafficType()) {
 *   │        case UNICAST:    execUnicast(...)
 *   │        case MULTICAST:  (not implemented)
 *   │        case BROADCAST:  (not implemented)
 *   │    }
 *   │
 *   └─ 5. Route packet to destination
 * ```
 *
 * **Unicast Routing (execUnicast) - Detailed Flow:**
 * ```
 * execUnicast(when, id, rNocPkt, rInfo):
 *   ├─ Extract routing parameters:
 *   │    ├─ destRId = rInfo->getDestRId()      // Destination ID
 *   │    ├─ destDMType = rInfo->getDestDMType()  // Endpoint type
 *   │    └─ pktID = rNocPkt->getID()           // Packet/Transaction ID
 *   │
 *   ├─ Route based on destination type:
 *   │
 *   │  ┌─ Case: TRAFFIC_GENERATOR (→ MCPU) ─────────────────────┐
 *   │  │  ├─ destName = "USMCPU"                                │
 *   │  │  ├─ mcpuSim = getUpStream(destName)                    │
 *   │  │  ├─ Create: MCPUPacket(when + 10)                      │
 *   │  │  ├─ Push to MasterChannelPort "USMCPU"                 │
 *   │  │  └─ Log: "DestDMTypeEnum::TRAFFIC_GENERATOR"           │
 *   │  └────────────────────────────────────────────────────────┘
 *   │
 *   │  ┌─ Case: CACHE (→ Cache Tile) ───────────────────────────┐
 *   │  │  ├─ destName = "DSCACHE" + std::to_string(destRId)     │
 *   │  │  ├─ cacheSim = getDownStream(destName)                 │
 *   │  │  ├─ Determine blackhole mode:                          │
 *   │  │  │    blackhole = (destDMType == BLACKHOLE)            │
 *   │  │  ├─ Create: CacheRNocPacket(when + 10, ptr, blackhole) │
 *   │  │  ├─ Create: CacheRNocEvent(pktID, ...)                 │
 *   │  │  ├─ Wrap: EventPacket(cRNocEvent, currentTick + 10)    │
 *   │  │  ├─ Push to MasterChannelPort "DSCACHE{destRId}"       │
 *   │  │  └─ Log: "DestDMTypeEnum::CACHE"                       │
 *   │  └────────────────────────────────────────────────────────┘
 *   │
 *   │  ┌─ Case: PE (→ Processing Element) ──────────────────────┐
 *   │  │  ├─ destName = "DSPE" + std::to_string(destRId)        │
 *   │  │  ├─ peSim = getDownStream(destName)                    │
 *   │  │  ├─ Create: PERNocPacket(when + 10, ptr)               │
 *   │  │  ├─ Create: PERNocEvent(pktID, ...)                    │
 *   │  │  ├─ Wrap: EventPacket(peRNocEvent, currentTick + 10)   │
 *   │  │  ├─ Push to MasterChannelPort "DSPE{destRId}"          │
 *   │  │  └─ Log: "DestDMTypeEnum::PE"                          │
 *   │  └────────────────────────────────────────────────────────┘
 *   │
 *   └─ Fixed routing latency: +10 ticks for all destinations
 * ```
 *
 * **Routing Examples:**
 * ```
 * Example 1: MCPU → PE #5 (Task Offload)
 *   ├─ Input: RNocPacket(destRId=5, destDMType=PE)
 *   ├─ RNocPacketHandler() extracts routing info
 *   ├─ execUnicast() routes to "DSPE5"
 *   ├─ Creates PERNocPacket + PERNocEvent
 *   ├─ Scheduled for currentTick + 10
 *   └─ Delivered to PESim #5
 *
 * Example 2: PE #3 → MCPU (Result Return)
 *   ├─ Input: RNocPacket(destRId=0, destDMType=TRAFFIC_GENERATOR)
 *   ├─ RNocPacketHandler() extracts routing info
 *   ├─ execUnicast() routes to "USMCPU"
 *   ├─ Creates MCPUPacket
 *   ├─ Scheduled for currentTick + 10
 *   └─ Delivered to MCPUSim
 *
 * Example 3: MCPU → Cache #2 (Memory Access)
 *   ├─ Input: RNocPacket(destRId=2, destDMType=CACHE)
 *   ├─ RNocPacketHandler() extracts routing info
 *   ├─ execUnicast() routes to "DSCACHE2"
 *   ├─ Creates CacheRNocPacket(blackhole=false)
 *   ├─ Scheduled for currentTick + 10
 *   └─ Delivered to CacheSim
 *
 * Example 4: PE #7 → Cache #1 (Test 1 Stress Traffic)
 *   ├─ Input: RNocPacket(destRId=1, destDMType=BLACKHOLE)
 *   ├─ RNocPacketHandler() extracts routing info
 *   ├─ execUnicast() routes to "DSCACHE1"
 *   ├─ Creates CacheRNocPacket(blackhole=true)
 *   ├─ Cache absorbs packet (no response)
 *   └─ Tests high-throughput routing
 * ```
 *
 * **Packet Type Transformations:**
 * ```
 * Input Packet    Routing Decision    Output Packet      Target
 * ────────────────────────────────────────────────────────────────
 * RNocPacket  →   TRAFFIC_GENERATOR → MCPUPacket     →   MCPU
 * RNocPacket  →   PE                → PERNocPacket   →   PE #x
 * RNocPacket  →   CACHE             → CacheRNocPacket→   Cache #x
 * RNocPacket  →   BLACKHOLE         → CacheRNocPacket→   Cache #x
 *                                      (blackhole=true)
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Reactive Architecture:**
 *    - NoC has no init() logic (passive)
 *    - Only responds to incoming packets
 *    - Pure routing function
 *
 * 2. **Routing Latency Model:**
 *    - Fixed 10-tick routing delay
 *    - Applied to all packet types
 *    - Models NoC traversal time
 *
 * 3. **Dynamic Channel Port Management:**
 *    - Multiple downstream ports (DSPEx, DSCACHEx)
 *    - Multiple upstream ports (USPE_NOCx, USNOC_CACHEx)
 *    - Port selection based on routing info
 *
 * 4. **Packet Type Specialization:**
 *    - Generic RNocPacket on input
 *    - Specialized packets on output:
 *      - MCPUPacket (for MCPU)
 *      - PERNocPacket (for PEs)
 *      - CacheRNocPacket (for Cache)
 *
 * 5. **Blackhole Mode Support:**
 *    - Special routing for stress testing
 *    - Cache packets marked as blackhole
 *    - No response generated
 *
 * **Data Structures:**
 * ```cpp
 * class NocSim : public CPPSimBase {
 * public:
 *     // Initialization (empty - passive router)
 *     void init() override;
 *
 *     // Cleanup (empty)
 *     void cleanup() override;
 *
 *     // Main packet routing handler
 *     void RNocPacketHandler(Tick when, SimPacket* pkt);
 *
 *     // Unicast routing implementation
 *     void execUnicast(Tick when, int id, RNocPacket* rNocPkt,
 *                      DLLRoutingInfo* rInfo);
 * };
 * ```
 *
 * **Routing Information Structure:**
 * ```cpp
 * class DLLRoutingInfo {
 *     FlitTypeEnum flitType;      // HEAD/BODY/TAIL
 *     TrafficTypeEnum trafficType; // UNICAST/MULTICAST/BROADCAST
 *     uint32_t destRId;           // Destination router ID
 *     DestDMTypeEnum destDMType;  // PE/CACHE/TRAFFIC_GENERATOR/BLACKHOLE
 * };
 * ```
 *
 * **Usage in System:**
 * ```cpp
 * // In TestAccTop::registerSimulators()
 * nocSim = new NocSim("Noxim");
 * this->addSimulator(nocSim);
 *
 * // Connect to MCPU
 * nocSim->addUpStream(mcpuSim, "USMCPU");
 * mcpuSim->addDownStream(nocSim, "DSNOC");
 *
 * // Connect to PEs (for each PE)
 * nocSim->addDownStream(peSims[peID], "DSPE" + std::to_string(peID));
 * nocSim->addUpStream(peSims[peID], "USPE_NOC" + std::to_string(peID));
 *
 * // Connect to Cache
 * nocSim->addDownStream(cacheSim, "DSCACHE" + std::to_string(cacheID));
 * cacheSim->addUpStream(nocSim, "USNOC_CACHE" + std::to_string(cacheID));
 * ```
 *
 * **Real-World NoC Analogy:**
 * ```
 * NocSim             →  On-chip interconnect fabric
 * execUnicast()      →  Routing table lookup
 * destRId            →  Physical router coordinate
 * 10-tick latency    →  Multi-hop routing delay
 * Channel ports      →  Physical links between routers
 * Packet transform   →  Protocol conversion (e.g., AXI → AHB)
 * BLACKHOLE mode     →  Fire-and-forget transactions
 * ```
 *
 * **Performance Characteristics:**
 * ```
 * Routing Latency:
 *   - Fixed: 10 ticks per packet
 *   - Independent of distance
 *   - Simplified model (no congestion)
 *
 * Scalability:
 *   - Supports 16 PEs + 4 Caches + 1 MCPU
 *   - O(1) routing decision
 *   - Channel-based isolation
 *
 * Throughput:
 *   - Limited by event queue
 *   - No explicit bandwidth model
 *   - Suitable for functional validation
 * ```
 *
 * **Related Files:**
 * - NocEvent.cc: Event processing for NoC packets
 * - NocPacket.cc: Visitor pattern implementation for routing
 * - testAccelerator.cc: System topology and NoC instantiation
 * - MCPUSim.cc: Sends task requests to NoC
 * - PESim.cc: Receives tasks from NoC, sends responses
 * - CacheSim.cc: Receives memory requests from NoC
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see NocEvent.cc for event processing implementation
 * @see NocPacket.cc for visitor pattern routing
 * @see testAccelerator.cc for system architecture overview
 */

#include <chrono>
#include <cstdlib>
#include <thread>
#define sleepus(val) std::this_thread::sleep_for(std::chrono::microseconds(val))

#include "cache/CacheEvent.hh"
#include "cache/CachePacket.hh"
#include "mcpu/MCPUPacket.hh"
#include "noc/NocPacket.hh"
#include "noc/NocSim.hh"
#include "peTile/PEEvent.hh"
#include "peTile/PEPacket.hh"

using namespace acalsim;

void NocSim::init() {}

void NocSim::cleanup() {}

void NocSim::execUnicast(Tick when, int id, RNocPacket* rNocPkt, DLLRoutingInfo* rInfo) {
	CLASS_INFO << "NocSim::execUnicast";
	uint32_t    destRId = rInfo->getDestRId();
	std::string destName;
	int         pktID = rNocPkt->getID();
	switch (rInfo->getDestDMType()) {
		case DestDMTypeEnum::TRAFFIC_GENERATOR: {
			destName            = "USMCPU";
			SimBase*    mcpuSim = this->getUpStream(destName);
			MCPUPacket* mcpuPkt = new MCPUPacket(when + 10);
			this->pushToMasterChannelPort(destName, mcpuPkt);
			CLASS_INFO << "DestDMTypeEnum::TRAFFIC_GENERATOR | " + mcpuSim->getName();
			break;
		}
		case DestDMTypeEnum::CACHE:
		case DestDMTypeEnum::BLACKHOLE: {
			destName                    = "DSCACHE" + std::to_string(destRId);
			SimBase*         cacheSim   = this->getDownStream(destName);
			CacheRNocPacket* cRNocPkt   = new CacheRNocPacket(when + 10, rNocPkt->getSharedContainer(),
			                                                  rInfo->getDestDMType() == DestDMTypeEnum::BLACKHOLE);
			CacheRNocEvent*  cRNocEvent = new CacheRNocEvent(pktID, destName + " ReqPkt", cRNocPkt, (void*)cacheSim);
			EventPacket*     eventPkt   = new EventPacket(cRNocEvent, top->getGlobalTick() + 10);
			this->pushToMasterChannelPort(destName, eventPkt);
			CLASS_INFO << "DestDMTypeEnum::CACHE | " + cacheSim->getName();
			break;
		}
		case DestDMTypeEnum::PE: {
			destName       = "DSPE" + std::to_string(destRId);
			SimBase* peSim = this->getDownStream(destName);
			CLASS_INFO << "DestDMTypeEnum::PE" + std::to_string(peSim == nullptr);
			PERNocPacket* peRNocPkt   = new PERNocPacket(when + 10, rNocPkt->getSharedContainer());
			PERNocEvent*  peRNocEvent = new PERNocEvent(pktID, destName + " ReqPkt", peRNocPkt, (void*)peSim);
			EventPacket*  eventPkt    = new EventPacket(peRNocEvent, top->getGlobalTick() + 10);
			this->pushToMasterChannelPort(destName, eventPkt);
			break;
		}
	}
}

void NocSim::RNocPacketHandler(Tick when, SimPacket* pkt) {
	if (dynamic_cast<RNocPacket*>((SimPacket*)pkt)) {
		auto                                           rNocPkt = dynamic_cast<RNocPacket*>((SimPacket*)pkt);
		std::shared_ptr<SharedContainer<DLLRNocFrame>> ptr     = rNocPkt->getSharedContainer();
		DLLRoutingInfo*                                rInfo   = ptr->run(0, &DLLRNocFrame::getRoutingInfo);

		switch (rInfo->getTrafficType()) {
			case TrafficTypeEnum::UNICAST: this->execUnicast(when, rNocPkt->getID(), rNocPkt, rInfo); break;
			case TrafficTypeEnum::MULTICAST: break;
			case TrafficTypeEnum::BROADCAST: break;
			default: break;
		}
	} else {
		CLASS_ERROR << "Invalid packet type";
	}
}
