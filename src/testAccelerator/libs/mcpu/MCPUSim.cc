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
 * @file MCPUSim.cc
 * @brief Host CPU simulator implementing task generation and response coordination
 *
 * This file implements MCPUSim, the **host CPU** in the accelerator co-design pattern.
 * It serves as the central coordinator that generates computational tasks, offloads them
 * to Processing Elements (PEs) or Cache through the NoC, and collects results. This
 * pattern mirrors real-world CPU-GPU or CPU-accelerator architectures.
 *
 * **MCPUSim Role in System:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────┐
 * │                    MCPUSim (Master CPU / Host)                     │
 * │                                                                    │
 * │  Primary Responsibilities:                                         │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Task Generation (Traffic Injection)                       │ │
 * │  │    - Sequential task creation (IDs 0-19)                     │ │
 * │  │    - 16 tasks → PE tiles (compute offload)                   │ │
 * │  │    - 4 tasks → Cache tiles (memory access)                   │ │
 * │  │                                                              │ │
 * │  │ 2. Task Routing Decision                                     │ │
 * │  │    - destRId: Which PE/Cache to target                       │ │
 * │  │    - destDMType: PE or CACHE endpoint                        │ │
 * │  │                                                              │ │
 * │  │ 3. Response Collection                                       │ │
 * │  │    - Receive completion notifications from PEs/Cache         │ │
 * │  │    - Trigger next task generation                            │ │
 * │  │    - Maintain transaction ordering                           │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Communication Flow:                                               │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │                 ┌─────────────┐                              │ │
 * │  │    injectTraffic() │           │ catchResponse()              │ │
 * │  │                 │           │                              │ │
 * │  │                 ↓           ↑                              │ │
 * │  │            genTraffic()  MCPUPacket                          │ │
 * │  │                 │           ↑                              │ │
 * │  │                 ↓           │                              │ │
 * │  │         MasterChannelPort  SlaveChannelPort                  │ │
 * │  │              "DSNOC"       "DSNOC"                           │ │
 * │  │                 │           ↑                              │ │
 * │  │                 └───────────┘                              │ │
 * │  │                       NoC                                    │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Channel Ports:                                                    │
 * │    ┌────────────────────────────────────────────────────────────┐ │
 * │    │ MasterChannelPort "DSNOC" → Sends task requests to NoC    │ │
 * │    │ SlaveChannelPort "DSNOC" ← Receives responses from NoC    │ │
 * │    └────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Task Generation Flow (injectTraffic → genTraffic):**
 * ```
 * Tick 0: MCPUSim::init()
 *   └─ injectTraffic() called
 *
 * injectTraffic() Logic:
 *   ├─ Get current transactionID (0-19)
 *   ├─ if (id < 20):
 *   │    ├─ if (id <= 15): genTraffic(id, PE)      // Tasks 0-15 → PEs
 *   │    └─ else: genTraffic(id - 16, CACHE)       // Tasks 16-19 → Cache
 *   └─ transactionID++
 *
 * genTraffic(destRId, destDMType) Detailed Steps:
 *   ├─ INPUT:
 *   │    ├─ destRId: Target PE/Cache ID (0-15 for PE, 0-3 for Cache)
 *   │    └─ destDMType: PE or CACHE
 *   │
 *   ├─ 1. Get downstream NoC simulator:
 *   │    nocSim = getDownStream("DSNOC")
 *   │
 *   ├─ 2. Prepare packet metadata:
 *   │    ├─ id = transactionID
 *   │    └─ when = currentTick + 5
 *   │
 *   ├─ 3. Create routing information:
 *   │    DLLRoutingInfo* rInfo = new DLLRoutingInfo(
 *   │        FlitTypeEnum::HEAD,           // Packet type
 *   │        TrafficTypeEnum::UNICAST,     // Point-to-point
 *   │        destRId,                      // Destination ID
 *   │        destDMType                    // PE or CACHE
 *   │    )
 *   │
 *   ├─ 4. Wrap routing info in shared container:
 *   │    SharedContainer<DLLRNocFrame>* ptr = new SharedContainer<...>()
 *   │    ptr->add(id, rInfo)
 *   │
 *   ├─ 5. Create NoC request packet:
 *   │    RNocPacket* rNocPkt = new RNocPacket(when, ptr)
 *   │
 *   ├─ 6. Create routing event:
 *   │    RNocEvent* rNocEvent = new RNocEvent(
 *   │        id,                     // Transaction ID
 *   │        "NOC Request Packet",   // Debug name
 *   │        rNocPkt,                // Packet payload
 *   │        nocSim                  // Target simulator
 *   │    )
 *   │
 *   ├─ 7. Wrap in EventPacket for channel delivery:
 *   │    EventPacket* eventPkt = new EventPacket(rNocEvent, when)
 *   │
 *   ├─ 8. Send to NoC via MasterChannelPort:
 *   │    pushToMasterChannelPort("DSNOC", eventPkt)
 *   │
 *   └─ 9. Log task generation:
 *        CLASS_INFO << "Issue traffic with transaction id: " << id
 * ```
 *
 * **Response Handling (catchResponse):**
 * ```
 * When PE/Cache completes task:
 *   ├─ NoC routes MCPUPacket to MCPUSim
 *   ├─ MCPUSim::accept() receives packet
 *   └─ Invokes MCPUPacket::visit(MCPUSim&)
 *       └─ Calls catchResponse()
 *
 * catchResponse() Logic:
 *   ├─ CLASS_INFO << "catchResponse()" << transactionID
 *   └─ injectTraffic()  // Trigger next task generation
 *       └─ (Cycle continues until transactionID >= 20)
 * ```
 *
 * **Task Distribution Pattern:**
 * ```
 * Transaction ID  |  Destination  |  Routing
 * ───────────────────────────────────────────────────
 *     0           →  PE #0        (id=0, type=PE)
 *     1           →  PE #1        (id=1, type=PE)
 *     ...
 *     15          →  PE #15       (id=15, type=PE)
 *     16          →  Cache #0     (id=0, type=CACHE)
 *     17          →  Cache #1     (id=1, type=CACHE)
 *     18          →  Cache #2     (id=2, type=CACHE)
 *     19          →  Cache #3     (id=3, type=CACHE)
 *     20+         →  (No more tasks generated)
 * ```
 *
 * **Execution Timeline Example:**
 * ```
 * Tick 0:
 *   ├─ MCPUSim::init() called
 *   └─ injectTraffic()
 *       ├─ transactionID = 0
 *       ├─ genTraffic(destRId=0, destDMType=PE)
 *       ├─ Create RNocPacket scheduled for Tick 5
 *       ├─ Push to "DSNOC" channel
 *       └─ transactionID = 1
 *
 * Tick 5:
 *   └─ RNocEvent delivered to NocSim
 *       └─ NoC routes to PE #0
 *
 * Tick 15:
 *   └─ PE #0 sends response back to NoC
 *
 * Tick 25:
 *   ├─ MCPUPacket delivered to MCPUSim
 *   ├─ catchResponse() invoked
 *   │    └─ CLASS_INFO: "catchResponse() 1"
 *   └─ injectTraffic()
 *       ├─ transactionID = 1
 *       ├─ genTraffic(destRId=1, destDMType=PE)
 *       ├─ Create RNocPacket scheduled for Tick 30
 *       └─ transactionID = 2
 *
 * Tick 30:
 *   └─ RNocEvent delivered to NocSim
 *       └─ NoC routes to PE #1
 *
 * ... (Pattern repeats for all 20 tasks)
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Sequential Task Generation:**
 *    - One task at a time (not parallel)
 *    - Next task only after previous response
 *    - Ensures ordering and prevents overwhelming system
 *
 * 2. **Transaction ID Management:**
 *    - Simple counter: 0 → 19
 *    - Used for packet identification
 *    - Enables response tracking
 *
 * 3. **Routing Information Packaging:**
 *    - DLLRoutingInfo: Data-link layer routing metadata
 *    - SharedContainer: Thread-safe shared data structure
 *    - RNocPacket: NoC-level packet wrapper
 *
 * 4. **Channel-Based Communication:**
 *    - MasterChannelPort: Initiates communication
 *    - EventPacket: Timestamped event delivery
 *    - Framework handles event scheduling
 *
 * 5. **Timing Model:**
 *    - Fixed 5-tick injection delay
 *    - when = currentTick + 5
 *    - Models task preparation overhead
 *
 * **Data Structures:**
 * ```cpp
 * class MCPUSim : public CPPSimBase {
 * private:
 *     int transactionID = 0;  // Current task ID (0-19)
 *
 * public:
 *     // Initialization: Starts task generation
 *     void init() override;
 *
 *     // Generate tasks sequentially
 *     void injectTraffic();
 *
 *     // Create and send individual task
 *     void genTraffic(uint32_t destRId, DestDMTypeEnum destDMType);
 *
 *     // Handle task completion, trigger next task
 *     void catchResponse();
 * };
 * ```
 *
 * **DLLRoutingInfo Parameters:**
 * ```cpp
 * enum FlitTypeEnum {
 *     HEAD,    // First packet in multi-flit transfer
 *     BODY,    // Middle packet
 *     TAIL     // Last packet
 * };
 *
 * enum TrafficTypeEnum {
 *     UNICAST,    // Point-to-point (used here)
 *     MULTICAST,  // One-to-many
 *     BROADCAST   // One-to-all
 * };
 *
 * enum DestDMTypeEnum {
 *     PE,                 // Processing Element
 *     CACHE,              // Cache memory
 *     TRAFFIC_GENERATOR,  // MCPU (for responses)
 *     BLACKHOLE          // No response needed
 * };
 * ```
 *
 * **Usage in System:**
 * ```cpp
 * // In TestAccTop::registerSimulators()
 * mcpuSim = new MCPUSim("Master CPU");
 * this->addSimulator(mcpuSim);
 *
 * // Connect to NoC
 * mcpuSim->addDownStream(nocSim, "DSNOC");
 * nocSim->addUpStream(mcpuSim, "USMCPU");
 *
 * // Connect channel ports
 * ChannelPortManager::ConnectPort(mcpuSim, nocSim, "DSNOC", "USMCPU");
 * ChannelPortManager::ConnectPort(nocSim, mcpuSim, "USMCPU", "DSNOC");
 * ```
 *
 * **Real-World CPU Analogy:**
 * ```
 * MCPUSim         →  CPU in GPU system
 * genTraffic()    →  cudaLaunchKernel() / OpenCL enqueue
 * transactionID   →  Command buffer ID
 * PE tiles        →  GPU streaming multiprocessors (SMs)
 * Cache tiles     →  GPU L2 cache
 * NoC             →  PCIe / NVLink interconnect
 * catchResponse() →  Kernel completion callback
 * ```
 *
 * **Performance Characteristics:**
 * ```
 * Task Generation Rate:
 *   - Limited by response latency
 *   - ~20 ticks per task round-trip
 *   - Total time: ~400 ticks for 20 tasks
 *
 * Scalability:
 *   - Sequential generation (not parallel)
 *   - Could be extended for pipelined injection
 *   - Current design ensures simplicity
 * ```
 *
 * **Related Files:**
 * - testAccelerator.cc: Main entry point and system setup
 * - NocSim.cc: Receives and routes task requests
 * - PESim.cc: Processes tasks and returns results
 * - CacheSim.cc: Handles memory access tasks
 * - MCPUPacket.hh: Response packet definition
 * - DLLRoutingInfo.hh: Routing metadata structure
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see NocSim.cc for task routing implementation
 * @see PESim.cc for task processing implementation
 * @see testAccelerator.cc for system architecture overview
 */

#include "mcpu/MCPUSim.hh"

#include <random>

#include "dataLinkLayer/DLLFrame.hh"
#include "dataLinkLayer/DLLHeader.hh"
#include "dataLinkLayer/DLLPayload.hh"
#include "dataLinkLayer/DLLRoutingInfo.hh"
#include "noc/NocEvent.hh"
#include "noc/NocPacket.hh"

void MCPUSim::init() { this->injectTraffic(); }

void MCPUSim::cleanup() {}

void MCPUSim::injectTraffic() {
	CLASS_INFO << "injectTraffic";
	// std::mt19937                            gen(std::random_device());
	// std::uniform_int_distribution<uint32_t> distribution(0, 19);
	// uint32_t                                id = distribution(gen);
	uint32_t id = this->transactionID;

	if (id < 20) {
		id <= 15 ? this->genTraffic(id, DestDMTypeEnum::PE) : this->genTraffic(id - 16, DestDMTypeEnum::CACHE);
		this->transactionID++;
	}
}

void MCPUSim::genTraffic(uint32_t destRId, DestDMTypeEnum destDMType) {
	SimBase* nocSim = this->getDownStream("DSNOC");
	int      id     = this->transactionID;
	Tick     when   = top->getGlobalTick() + 5;

	DLLRoutingInfo* rInfo = new DLLRoutingInfo(FlitTypeEnum::HEAD, TrafficTypeEnum::UNICAST, destRId, destDMType);
	std::shared_ptr<SharedContainer<DLLRNocFrame>> ptr = std::make_shared<SharedContainer<DLLRNocFrame>>();
	ptr->add(id, rInfo);
	RNocPacket*  rNocPkt   = new RNocPacket(when, ptr);
	RNocEvent*   rNocEvent = new RNocEvent(id, "NOC Request Packet", rNocPkt, nocSim);
	EventPacket* eventPkt  = new EventPacket(rNocEvent, when);
	this->pushToMasterChannelPort("DSNOC", eventPkt);
	CLASS_INFO << "Issue traffic with transaction id: " << id << " at Tick=" << top->getGlobalTick();
}
