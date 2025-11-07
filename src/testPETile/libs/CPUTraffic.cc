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
 * @file CPUTraffic.cc
 * @brief Memory traffic generator module for PE tile testing and benchmarking
 *
 * This file implements the CPUTraffic module (also called PCU - Processing/Computation Unit),
 * which serves as both a traffic generator for memory system testing and a simplified model
 * of a processing element in the tile architecture. All core functionality is implemented
 * inline in the header file (CPUTraffic.hh), making this source file minimal.
 *
 * **CPUTraffic Module Overview:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                       CPUTraffic Module                                │
 * │              (Processing Element / Traffic Generator)                  │
 * │                                                                        │
 * │  Initialization Phase (init()):                                        │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ Inject Memory Requests                                           │ │
 * │  │ ├─ Generate 5 TENSOR_MEM_READ requests (NUM_TEST_REQ)            │ │
 * │  │ ├─ Schedule at ticks: 1, 11, 21, 31, 41                          │ │
 * │  │ ├─ Addresses: 0x0000, 0x1000, 0x2000, 0x3000, 0x4000             │ │
 * │  │ ├─ Sizes: 0, 20, 40, 60, 80 bytes                                │ │
 * │  │ └─ Each with unique transaction ID: 0, 1, 2, 3, 4                │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Request Generation (injectMemRequests()):                             │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ For each request (i = 0 to 4):                                   │ │
 * │  │   ├─ Create MemRespPacket (embedded in request)                  │ │
 * │  │   ├─ Create callback: λ(id, pkt) { MemRespHandler(tid, pkt); }  │ │
 * │  │   ├─ Create MemReqPacket with callback                           │ │
 * │  │   ├─ Create CPUReqEvent(tid=i, bus, callback, memReqPkt)        │ │
 * │  │   └─ Schedule at tick: 1 + i*10                                  │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Response Handling (MemRespHandler()):                                 │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ When response received:                                          │ │
 * │  │   ├─ Log response with transaction ID                            │ │
 * │  │   ├─ Set respReceived flag = true                                │ │
 * │  │   ├─ Free response packet memory                                 │ │
 * │  │   ├─ Increment stats.numResp counter                             │ │
 * │  │   └─ If GTest mode && all responses received:                    │ │
 * │  │      └─ Set test completion bit mask                             │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Statistics Tracking:                                                  │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ struct Stats {                                                   │ │
 * │  │     uint32_t numResp = 0;  // Number of responses received       │ │
 * │  │ }                                                                │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Complete Request-Response Lifecycle:**
 * ```
 * Traffic Generation & Response Flow:
 * ═══════════════════════════════════════════════════════════════════
 *
 * Initialization (Tick 0):
 *   CPUTraffic::init() called
 *     └─> injectMemRequests()
 *         ├─ Create 5 MemReqPackets with callbacks
 *         ├─ Create 5 CPUReqEvents (tid 0-4)
 *         └─ Schedule at ticks 1, 11, 21, 31, 41
 *
 * Request 0 Lifecycle:
 * ────────────────────────────────────────────
 * Tick 1: CPUReqEvent(tid=0) processes
 *   ├─ Creates callback: λ(id, resp) { MemRespHandler(0, resp); }
 *   ├─ Sends MemReqPacket(addr=0x0000, size=0) to AXI Bus
 *   └─ Adds trace record (CpuTrafficTraceRecord)
 *
 * Tick 1 + bus_req_delay: AXI Bus forwards to SRAM
 *   └─> BusReqEvent wraps callback
 *
 * Tick X: SRAM processes request
 *   ├─ Calculates delay = sram_req_delay + (0+1)/256
 *   └─ Schedules SRAMRespEvent
 *
 * Tick Y: Response propagates through bus
 *   └─> BusRespEvent scheduled with bus response delay
 *
 * Tick Z: CPUTraffic receives response
 *   └─> MemRespHandler(id=0, pkt) invoked
 *       ├─ Log: "Receive MemRespPacket with transaction id: 0"
 *       ├─ Set respReceived = true
 *       ├─ Free response packet
 *       └─ stats.numResp = 1
 *
 * [Requests 1-4 follow similar pattern at ticks 11, 21, 31, 41]
 * ```
 *
 * **Traffic Pattern Details:**
 * ```
 * Generated Memory Request Pattern:
 * ═══════════════════════════════════════════════
 *
 * │ TID │ Tick │ Type            │ Address │ Size  │
 * ├─────┼──────┼─────────────────┼─────────┼───────┤
 * │  0  │  1   │ TENSOR_MEM_READ │ 0x0000  │  0 B  │
 * │  1  │  11  │ TENSOR_MEM_READ │ 0x1000  │ 20 B  │
 * │  2  │  21  │ TENSOR_MEM_READ │ 0x2000  │ 40 B  │
 * │  3  │  31  │ TENSOR_MEM_READ │ 0x3000  │ 60 B  │
 * │  4  │  41  │ TENSOR_MEM_READ │ 0x4000  │ 80 B  │
 *
 * Pattern Characteristics:
 *   - Regular 10-tick interval between requests
 *   - Sequential addressing (stride = 0x1000 = 4096 bytes)
 *   - Linearly increasing sizes (0, 20, 40, 60, 80 bytes)
 *   - All read operations (no writes in this example)
 *   - Suitable for testing memory system latency and throughput
 * ```
 *
 * **Callback Mechanism:**
 * ```
 * Callback Creation and Invocation Chain:
 * ════════════════════════════════════════════
 *
 * Step 1: Create Callback (in injectMemRequests()):
 *   std::function<void(int, MemRespPacket*)> callback =
 *     [this, memRespPkt](int id, MemRespPacket* pkt) {
 *       this->MemRespHandler(this->transactionID, memRespPkt);
 *     };
 *
 * Step 2: Attach to Request Packet:
 *   MemReqPacket* memReqPkt = new MemReqPacket(...);
 *   // Callback embedded in CPUReqEvent, not directly in packet yet
 *
 * Step 3: CPUReqEvent::process() sets callback:
 *   auto callback = [this](int id, MemRespPacket* resp) {
 *     this->cpuReqCallback(this->tid, resp);
 *   };
 *   memReqPkt->setCallback(callback);
 *
 * Step 4: AXI Bus wraps callback (in BusReqEvent)
 * Step 5: SRAM invokes wrapped callback (via SRAMRespEvent)
 * Step 6: Bus invokes original callback (via BusRespEvent)
 * Step 7: CPUReqEvent::cpuReqCallback() invoked
 * Step 8: MemRespHandler() finally called
 *   └─> Updates statistics and frees memory
 * ```
 *
 * **Trace Generation:**
 * ```
 * CPU Traffic Trace Records:
 * ════════════════════════════════════════════
 *
 * Each request generates a CpuTrafficTraceRecord:
 *   top->addTraceRecord(
 *     std::make_shared<CpuTrafficTraceRecord>(
 *       tick,           // When request was sent
 *       req_type,       // TENSOR_MEM_READ/WRITE
 *       transaction_id, // Unique ID for correlation
 *       addr,           // Memory address
 *       size            // Transfer size in bytes
 *     ),
 *     "CPUReq"  // Trace category
 *   );
 *
 * Trace Output (JSON format):
 *   {
 *     "tick": 1,
 *     "transaction-id": 0,
 *     "req-type": "TENSOR_MEM_READ",
 *     "addr": "0x0000",
 *     "size": 0
 *   }
 *
 * Use Case:
 *   - Correlate requests with responses
 *   - Analyze memory access patterns
 *   - Measure end-to-end latencies
 *   - Debug transaction flows
 *   - Visualize in chrome://tracing
 * ```
 *
 * **Google Test Integration:**
 * ```
 * Automated Test Verification:
 * ════════════════════════════════════════════
 *
 * if (top->isGTestMode() && stats.numResp == NUM_TEST_REQ) {
 *     top->setGTestBitMask(getSimID(), getID());
 * }
 *
 * Test Criteria:
 *   - Verify all 5 requests received responses
 *   - Set completion bit in test framework
 *   - Enable automated regression testing
 *   - No manual verification required
 *
 * NUM_TEST_REQ = 5 (defined in Test.hh)
 * ```
 *
 * **Module Connectivity:**
 * ```
 * CPUTraffic Port Configuration:
 * ═══════════════════════════════════════════════
 *
 * Downstream Connection (Request Path):
 *   - "Bus": Connected to AXI Bus
 *     └─ Sends memory requests to bus
 *
 * Port Setup (from PETile::registerModules()):
 *   pcu->addDownStream(bus, "Bus")
 *     └─ Establishes CPUTraffic → AXI Bus connection
 *
 * Port Discovery (in injectMemRequests()):
 *   SimModule* bus = this->getDownStream("Bus")
 *     └─ Retrieves bus module for request sending
 *     └─ Asserts bus exists (fail fast if misconfigured)
 * ```
 *
 * **Extension Opportunities:**
 * ```
 * Current: Synthetic Traffic Generator
 * ═══════════════════════════════════════════════
 *   - Fixed pattern (5 reads at regular intervals)
 *   - Simple address sequence
 *   - No compute modeling
 *
 * Future Extensions:
 * ═══════════════════════════════════════════════
 *   1. Trace-Driven Traffic:
 *      - Read real memory traces from file
 *      - Replay application access patterns
 *      - Support variable request types
 *
 *   2. Compute Modeling:
 *      - Add instruction execution simulation
 *      - Model compute-memory overlap
 *      - Track dependencies between operations
 *
 *   3. Multi-Threaded PCU:
 *      - Multiple outstanding requests
 *      - Thread-level parallelism
 *      - Shared/private memory regions
 *
 *   4. Power Modeling:
 *      - Track active/idle cycles
 *      - Energy per request type
 *      - DVFS (voltage/frequency scaling)
 *
 *   5. Stochastic Traffic:
 *      - Poisson arrival process
 *      - Random address generation
 *      - Read/write mix configuration
 * ```
 *
 * **Header-Only Implementation:**
 * ```
 * Why This File is Minimal:
 * ═══════════════════════════════════════════════
 *
 * All CPUTraffic methods are defined inline in CPUTraffic.hh:
 *   - init()             : Calls injectMemRequests()
 *   - injectMemRequests(): Creates and schedules 5 requests
 *   - MemRespHandler()   : Processes responses, updates stats
 *   - accept()           : Not implemented (error if called)
 *   - receivedOrNot()    : Returns respReceived flag
 *
 * Benefits of Header-Only:
 *   - Template-friendly (if needed later)
 *   - Better inlining opportunities
 *   - Simpler build dependencies
 *   - Faster compilation (no separate .cc compilation)
 *
 * This .cc file:
 *   - Exists for consistency with other modules
 *   - May be used for future non-inline implementations
 *   - Satisfies build system expectations
 *   - Contains only include statement
 * ```
 *
 * **Design Patterns:**
 *
 * 1. **Traffic Generator Pattern:**
 *    - Inject requests during initialization
 *    - Responses handled asynchronously via callbacks
 *    - Statistics collected for verification
 *
 * 2. **Callback-Based Communication:**
 *    - Lambda callbacks capture 'this' and transaction ID
 *    - Enables response correlation with original request
 *    - Supports out-of-order response delivery
 *
 * 3. **Event-Driven Architecture:**
 *    - CPUReqEvent scheduled for each request
 *    - No polling or active waiting
 *    - Efficient simulation of concurrent traffic
 *
 * **Related Files:**
 * - Header: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/CPUTraffic.hh
 *   (Contains all implementation)
 * - Event: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUReqEvent.cc
 * - Packets: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/MemReq.cc
 * - AXI Bus: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/AXIBus.cc
 * - PETile: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/PETile.cc
 * - Test Defs: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/Test.hh
 *
 * @author ACALSim Framework Team
 * @date 2023-2025
 */

#include "CPUTraffic.hh"

// All CPUTraffic functionality is implemented inline in the header file (CPUTraffic.hh)
// This source file exists for build system consistency and future non-inline implementations
