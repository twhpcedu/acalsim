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
 * @file PESim.cc
 * @brief Processing Element (PE) accelerator simulator implementing task processing
 *
 * This file implements PESim, the **hardware accelerator** component in the host-accelerator
 * co-design pattern. Each PE receives computational tasks from the host CPU (MCPU) via NoC,
 * performs processing, and returns results. This mirrors real-world accelerator architectures
 * like GPU compute units, AI accelerator cores, or DSP tiles.
 *
 * **PESim Role in System:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────┐
 * │                  PESim (Processing Element / Accelerator)          │
 * │                        (16 instances: PE #0 - #15)                 │
 * │                                                                    │
 * │  Primary Responsibilities:                                         │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Task Reception                                            │ │
 * │  │    - Receive PERNocPacket from NoC                           │ │
 * │  │    - Extract task parameters                                 │ │
 * │  │    - Identify transaction ID                                 │ │
 * │  │                                                              │ │
 * │  │ 2. Task Processing (Simulated)                               │ │
 * │  │    - Test 0: Immediate processing                            │ │
 * │  │    - Test 2: Random delay injection (async timing)           │ │
 * │  │                                                              │ │
 * │  │ 3. Result Return                                             │ │
 * │  │    - Create response packet                                  │ │
 * │  │    - Route back to MCPU via NoC                              │ │
 * │  │    - Fixed 10-tick response latency                          │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Communication Flow:                                               │
 * │  ┌──────────────────────────────────────────────────────────────┐ │
 * │  │            ↓ PERNocPacket (Task Request)                     │ │
 * │  │      SlaveChannelPort "DSPEx"                                │ │
 * │  │            │                                                 │ │
 * │  │            ↓                                                 │ │
 * │  │    RNocPacketHandler()                                       │ │
 * │  │            │                                                 │ │
 * │  │            ↓                                                 │ │
 * │  │    [Process Task]                                            │ │
 * │  │            │                                                 │ │
 * │  │            ↓                                                 │ │
 * │  │    Create RNocPacket (Response)                              │ │
 * │  │            │                                                 │ │
 * │  │            ↓                                                 │ │
 * │  │      MasterChannelPort "USNOC_PEx"                           │ │
 * │  │            ↓ RNocPacket → NoC → MCPU                         │ │
 * │  └──────────────────────────────────────────────────────────────┘ │
 * │                                                                    │
 * │  Channel Ports (per PE):                                           │
 * │    ┌────────────────────────────────────────────────────────────┐ │
 * │    │ SlaveChannelPort "DSPEx" ← Receives tasks from NoC        │ │
 * │    │ MasterChannelPort "USNOC_PEx" → Sends responses to NoC    │ │
 * │    │ MasterChannelPort "DSNOC" → (Used in stress test mode)    │ │
 * │    └────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Task Processing Flow (RNocPacketHandler):**
 * ```
 * When NoC delivers PERNocPacket:
 *   ├─ PERNocEvent.process() invoked
 *   ├─ Calls PESim::accept()
 *   └─ Dispatches to PESim::RNocPacketHandler()
 *
 * RNocPacketHandler(when, pkt) Detailed Steps:
 *   ├─ 1. Type check: dynamic_cast<PERNocPacket*>(pkt)
 *   │    └─ Verify packet is PE-specific request
 *   │
 *   ├─ 2. Extract packet information:
 *   │    ├─ peRNocPkt = cast to PERNocPacket
 *   │    ├─ pktID = peRNocPkt->getID()
 *   │    └─ Build destName = "USNOC_PE" + peID
 *   │
 *   ├─ 3. Get upstream NoC simulator:
 *   │    nocSim = getUpStream(destName)
 *   │
 *   ├─ 4. Create response routing info:
 *   │    DLLRoutingInfo* rInfo = new DLLRoutingInfo(
 *   │        FlitTypeEnum::HEAD,
 *   │        TrafficTypeEnum::UNICAST,
 *   │        0,                              // destRId = 0 (MCPU)
 *   │        DestDMTypeEnum::TRAFFIC_GENERATOR  // Back to host
 *   │    )
 *   │
 *   ├─ 5. Package response in shared container:
 *   │    SharedContainer<DLLRNocFrame>* ptr = new SharedContainer<...>()
 *   │    ptr->add(pktID, rInfo)
 *   │
 *   ├─ 6. Create response packet:
 *   │    RNocPacket* rNocPkt = new RNocPacket(
 *   │        currentTick + 10,  // Response scheduled 10 ticks ahead
 *   │        ptr
 *   │    )
 *   │
 *   ├─ 7. Create response event:
 *   │    RNocEvent* rNocEvent = new RNocEvent(
 *   │        pktID,
 *   │        "NOC Request Packet",
 *   │        rNocPkt,
 *   │        nocSim
 *   │    )
 *   │
 *   ├─ 8. Wrap and send via channel:
 *   │    EventPacket* eventPkt = new EventPacket(
 *   │        rNocEvent,
 *   │        currentTick + 10
 *   │    )
 *   │    pushToMasterChannelPort(destName, eventPkt)
 *   │
 *   └─ 9. Test mode handling (if getTestNum() == 2):
 *        sleepus(rand() % 1000)  // Random 0-999μs delay
 *        // Simulates variable processing time
 * ```
 *
 * **Test Mode Behaviors:**
 * ```
 * Test 0: Basic Operation (Default)
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ - PE receives task from MCPU                             │
 *   │ - Immediately processes (no delay)                       │
 *   │ - Returns response after 10 ticks                        │
 *   │ - Clean request-response protocol                        │
 *   └──────────────────────────────────────────────────────────┘
 *
 * Test 1: Traffic Generation (handled in PEEvent.cc)
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ - PE receives initial task from MCPU                     │
 *   │ - Returns response to MCPU (same as Test 0)              │
 *   │ - PEEvent.process() generates 100 additional tasks       │
 *   │ - Additional tasks → Cache (BLACKHOLE mode)              │
 *   │ - Tests high-throughput PE→Cache traffic                 │
 *   └──────────────────────────────────────────────────────────┘
 *
 * Test 2: Asynchronous Processing
 *   ┌──────────────────────────────────────────────────────────┐
 *   │ - PE receives task from MCPU                             │
 *   │ - Inject random delay: sleepus(rand() % 1000)            │
 *   │ - Simulates variable computation time                    │
 *   │ - Returns response after delay + 10 ticks                │
 *   │ - Tests out-of-order completion handling                 │
 *   └──────────────────────────────────────────────────────────┘
 * ```
 *
 * **Execution Timeline Example (Test 0, PE #0):**
 * ```
 * Tick 5:
 *   ├─ PERNocPacket arrives at PE #0
 *   ├─ PERNocEvent.process() invoked
 *   └─ PESim#0::RNocPacketHandler(when=5, pkt=PERNocPacket)
 *       ├─ Extract pktID = 0
 *       ├─ Create response routing: destRId=0, TRAFFIC_GENERATOR
 *       ├─ Create RNocPacket scheduled for Tick 15
 *       ├─ Create RNocEvent
 *       ├─ Push to MasterChannelPort "USNOC_PE0"
 *       └─ Response delivered to NoC
 *
 * Tick 15:
 *   └─ NoC receives response from PE #0
 *       └─ Routes to MCPU
 *
 * Tick 25:
 *   └─ MCPU receives response
 *       └─ Triggers next task generation
 * ```
 *
 * **Execution Timeline Example (Test 2, PE #3):**
 * ```
 * Tick 20:
 *   ├─ PERNocPacket arrives at PE #3
 *   └─ PESim#3::RNocPacketHandler(when=20, pkt=...)
 *       ├─ Create response scheduled for Tick 30
 *       ├─ Push to channel
 *       └─ sleepus(rand() % 1000)  // e.g., 437μs
 *           └─ Wall-clock delay (doesn't affect simulation time)
 *
 * Tick 30:
 *   └─ Response event still scheduled for Tick 30
 *       └─ NoC receives response (simulation time unaffected)
 *
 * Note: sleepus() creates wall-clock delay but preserves simulation timing
 * ```
 *
 * **Multi-PE Parallel Operation:**
 * ```
 * System with 16 PEs (default configuration):
 *
 * PE #0:  Task 0  → Process → Response
 * PE #1:  Task 1  → Process → Response
 * PE #2:  Task 2  → Process → Response
 * ...
 * PE #15: Task 15 → Process → Response
 *
 * Tasks 16-19 go to Cache, not PEs
 *
 * All PEs operate independently:
 *   - Separate channel ports
 *   - Independent event queues
 *   - Can process in parallel (if tasks sent concurrently)
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Per-PE Identification:**
 *    - Each PE has unique peID (0-15)
 *    - Used for channel port naming
 *    - Enables independent operation
 *
 * 2. **Response Latency Model:**
 *    - Fixed 10-tick processing time
 *    - currentTick + 10 → response delivery
 *    - Models accelerator computation delay
 *
 * 3. **Bidirectional Channel Ports:**
 *    - Receive: SlaveChannelPort "DSPEx"
 *    - Send: MasterChannelPort "USNOC_PEx"
 *    - Separate ports for request/response
 *
 * 4. **Test Mode Support:**
 *    - testNum stored per PE instance
 *    - Controls processing behavior
 *    - Enables stress testing scenarios
 *
 * 5. **Packet Type Safety:**
 *    - dynamic_cast ensures PERNocPacket
 *    - Type-safe packet handling
 *    - Visitor pattern integration
 *
 * **Data Structures:**
 * ```cpp
 * class PESim : public CPPSimBase {
 * private:
 *     uint32_t peID;      // PE instance ID (0-15)
 *     int testNum;        // Test mode (0, 1, 2)
 *
 * public:
 *     // Constructor: Initialize PE with ID and test mode
 *     PESim(std::string name, int _peID, int _testNum = 0);
 *
 *     // Initialization (currently empty)
 *     void init() override;
 *
 *     // Cleanup (currently empty)
 *     void cleanup() override;
 *
 *     // Process incoming task requests
 *     void RNocPacketHandler(Tick when, SimPacket* pkt);
 *
 *     // Get test mode
 *     int getTestNum();
 * };
 * ```
 *
 * **Response Packet Structure:**
 * ```cpp
 * // Response always targets MCPU (TRAFFIC_GENERATOR)
 * DLLRoutingInfo(
 *     FlitTypeEnum::HEAD,           // Single-flit packet
 *     TrafficTypeEnum::UNICAST,     // Point-to-point
 *     0,                            // destRId = 0 (MCPU always at ID 0)
 *     DestDMTypeEnum::TRAFFIC_GENERATOR  // Host CPU endpoint
 * )
 * ```
 *
 * **Usage in System:**
 * ```cpp
 * // In TestAccTop::registerSimulators()
 * for (int peID = 0; peID < peCount; ++peID) {
 *     peSims.push_back(new PESim("PETile", peID, testNum));
 *     this->addSimulator(peSims.back());
 * }
 *
 * // Connect to NoC (for each PE)
 * nocSim->addDownStream(peSims[peID], "DSPE" + std::to_string(peID));
 * peSims[peID]->addUpStream(nocSim, "USNOC_PE" + std::to_string(peID));
 *
 * // Connect channel ports
 * ChannelPortManager::ConnectPort(nocSim, peSims[peID],
 *     "DSPE" + std::to_string(peID), "DSPE" + std::to_string(peID));
 * ChannelPortManager::ConnectPort(peSims[peID], nocSim,
 *     "USNOC_PE" + std::to_string(peID), "USNOC_PE" + std::to_string(peID));
 * ```
 *
 * **Real-World Accelerator Analogy:**
 * ```
 * PESim              →  GPU Streaming Multiprocessor (SM)
 * peID               →  SM ID / Core ID
 * RNocPacketHandler  →  Command processor / Instruction dispatch
 * 10-tick latency    →  Kernel execution time
 * Multiple PEs       →  Multi-core GPU (e.g., 16 SMs)
 * Test 2 delay       →  Variable workload processing
 * Response packet    →  Computation result / Status flag
 * ```
 *
 * **Performance Characteristics:**
 * ```
 * Single PE Processing:
 *   - Reception: Instantaneous (event-driven)
 *   - Processing: 10 ticks (fixed latency)
 *   - Response: Immediate creation, delivered in 10 ticks
 *
 * Multi-PE Scalability:
 *   - 16 PEs can process 16 tasks concurrently
 *   - Independent execution (no contention)
 *   - NoC may introduce routing delays
 *
 * Test 2 Impact:
 *   - Wall-clock time increases (sleepus)
 *   - Simulation time unchanged
 *   - Tests framework's timing robustness
 * ```
 *
 * **Related Files:**
 * - PEEvent.cc: PE event processing (handles Test 1 traffic generation)
 * - testAccelerator.cc: System setup and PE instantiation
 * - MCPUSim.cc: Task generation (sends tasks to PEs)
 * - NocSim.cc: Routes packets between MCPU and PEs
 * - PEPacket.hh: PERNocPacket definition
 *
 * @author ACALSim Framework
 * @date 2023-2025
 * @see PEEvent.cc for stress test traffic generation logic
 * @see MCPUSim.cc for task generation implementation
 * @see NocSim.cc for packet routing implementation
 * @see testAccelerator.cc for system architecture overview
 */

#include <chrono>
#include <cstdlib>
#include <thread>
#define sleepus(val) std::this_thread::sleep_for(std::chrono::microseconds(val))

#include "noc/NocEvent.hh"
#include "noc/NocPacket.hh"
#include "peTile/PEPacket.hh"
#include "peTile/PESim.hh"

void PESim::init() {}

void PESim::cleanup() {}

void PESim::RNocPacketHandler(Tick when, SimPacket* pkt) {
	if (dynamic_cast<PERNocPacket*>((SimPacket*)pkt)) {
		auto        peRNocPkt = dynamic_cast<PERNocPacket*>((SimPacket*)pkt);
		std::string destName  = "USNOC_PE" + std::to_string(this->peID);
		SimBase*    nocSim    = this->getUpStream(destName);
		int         pktID     = peRNocPkt->getID();

		DLLRoutingInfo* rInfo =
		    new DLLRoutingInfo(FlitTypeEnum::HEAD, TrafficTypeEnum::UNICAST, 0, DestDMTypeEnum::TRAFFIC_GENERATOR);
		std::shared_ptr<SharedContainer<DLLRNocFrame>> ptr = std::make_shared<SharedContainer<DLLRNocFrame>>();
		ptr->add(pktID, rInfo);
		RNocPacket*  rNocPkt   = new RNocPacket(top->getGlobalTick() + 10, ptr);
		RNocEvent*   rNocEvent = new RNocEvent(pktID, "NOC Request Packet", rNocPkt, nocSim);
		EventPacket* eventPkt  = new EventPacket(rNocEvent, top->getGlobalTick() + 10);
		this->pushToMasterChannelPort(destName, eventPkt);
	}
	if (this->getTestNum() == 2) { sleepus(rand() % 1000); }
}
