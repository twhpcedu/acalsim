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
 * @brief Cache Memory Simulator Implementation
 *
 * @details
 * This file implements the CacheSim class, which serves as the endpoint in the testSimChannel
 * memory subsystem simulation. It models a simplified cache memory that receives requests
 * from the Network-on-Chip (NoC), processes them with realistic timing, and returns data
 * responses. The cache acts as the final destination in the request path and the source
 * of the response path.
 *
 * # Component Role in System Architecture
 *
 * CacheSim represents the bottom of the memory hierarchy:
 *
 * @code{.unparsed}
 *                  TrafficGenerator
 *                         ↓
 *                      NocSim
 *                         ↓ CacheReqPacket
 *              ┌──────────────────────────┐
 *              │ Slave Port: NOC2Cache-s  │ (receives requests)
 *              └──────────┬───────────────┘
 *                         ↓
 *              ┌──────────────────────────┐
 *              │      CacheSim (This)     │
 *              │  ┌────────────────────┐  │
 *              │  │  Request Handler   │  │
 *              │  │  - Extract params  │  │
 *              │  │  - Generate data   │  │
 *              │  │  - Calculate delay │  │
 *              │  └────────────────────┘  │
 *              │  ┌────────────────────┐  │
 *              │  │  Timing Model      │  │
 *              │  │  Latency: 1 tick   │  │
 *              │  │  BW: 32 bytes/tick │  │
 *              │  └────────────────────┘  │
 *              └──────────┬───────────────┘
 *                         ↓
 *              ┌──────────────────────────┐
 *              │ Master Port: Cache2NOC-m │ (sends responses)
 *              └──────────┬───────────────┘
 *                         ↓ CacheRespPacket
 *                      NocSim
 *                         ↓
 *                  TrafficGenerator
 * @endcode
 *
 * # Request Processing Workflow
 *
 * ## handleNOCRequest() Method
 * This is the core function that processes incoming cache requests:
 *
 * ### Input Parameters:
 * - **_cacheReqPkt**: Pointer to CacheReqPacket containing:
 *   - type: CachePktTypeEnum::TEST
 *   - addr: Memory address to access
 *   - size: Transfer size in bytes
 *   - tid: Transaction ID for response matching
 * - **_when**: Tick value when the packet arrived (used for data encoding)
 *
 * ### Processing Steps:
 *
 * 1. **Data Generation**:
 * @code
 * int* data = new int(100 + (int)_when);
 * // Example: If request arrives at tick 11, data = 111
 * @endcode
 * This encoding allows verification of timing correctness in the full system.
 *
 * 2. **Response Packet Creation**:
 * @code
 * auto cacheRespPkt = new CacheRespPacket(
 *     CachePktTypeEnum::TEST,  // Response type
 *     data,                    // Data payload (heap-allocated)
 *     _cacheReqPkt->getTransactionId()  // Copy transaction ID
 * );
 * @endcode
 *
 * 3. **Delay Calculation**:
 * @code
 * auto local_delay = this->getRespDelay(_cacheReqPkt->getSize());
 * // local_delay = cacheMemoryLatency + (size + 1) / cacheMemoryBandwidth
 * // For 256-byte request: 1 + (256+1)/32 = 1 + 8 = 9 ticks
 * @endcode
 *
 * 4. **Response Transmission**:
 * @code
 * this->sendPacketViaChannel(
 *     "Cache2NOC-m",  // Port name
 *     local_delay,    // 9 ticks before packet leaves cache
 *     0,              // 0 ticks remote delay (immediate to NoC)
 *     cacheRespPkt    // Response packet
 * );
 * @endcode
 *
 * # Timing Model
 *
 * ## Configuration Constants (from CacheSim.hh):
 * - **cacheMemoryLatency**: 1 tick (base access time)
 * - **cacheMemoryBandwidth**: 32 bytes/tick (transfer rate)
 *
 * ## Response Delay Formula:
 * @code
 * getRespDelay(int size) {
 *     return latency + ceil(size / bandwidth);
 * }
 * // Implemented as: 1 + (size + 1) / 32
 * @endcode
 *
 * ## Delay Examples:
 *
 * | Request Size | Latency | BW Delay | Total | Calculation           |
 * |--------------|---------|----------|-------|-----------------------|
 * | 32 bytes     | 1 tick  | 1 tick   | 2     | 1 + (32+1)/32 = 2     |
 * | 64 bytes     | 1 tick  | 2 ticks  | 3     | 1 + (64+1)/32 = 3     |
 * | 256 bytes    | 1 tick  | 8 ticks  | 9     | 1 + (256+1)/32 = 9    |
 * | 512 bytes    | 1 tick  | 16 ticks | 17    | 1 + (512+1)/32 = 17   |
 *
 * This models:
 * - **Latency component**: Time to locate data (tag lookup, set selection)
 * - **Bandwidth component**: Time to transfer data on the cache bus
 *
 * # Data Encoding Strategy
 *
 * The cache encodes timing information into response data:
 *
 * @code
 * response_data = 100 + arrival_tick
 * @endcode
 *
 * ## Example Timeline:
 * @code{.unparsed}
 * Tick 0:  TrafficGenerator sends NocReqPacket (tid=0)
 * Tick 11: NocSim forwards as CacheReqPacket to Cache
 * Tick 11: CacheSim receives request (_when=11)
 *          → data = new int(100 + 11) = 111
 *          → local_delay = 9 ticks
 * Tick 20: CacheRespPacket leaves cache (11 + 9 = 20)
 * Tick 20: NocSim receives response
 * Tick 20: TrafficGenerator receives NocRespPacket
 *          → Validates data == 111 ✓
 * @endcode
 *
 * This allows end-to-end verification:
 * - Expected arrival at cache: 11 ticks (1 + 10 for tid=0)
 * - Expected data value: 111
 * - Expected response arrival: 20 ticks (11 + 9)
 *
 * # SimChannel Communication Details
 *
 * ## Port Configuration:
 * - **Incoming Port**: "NOC2Cache-s" (slave port from NoC)
 * - **Outgoing Port**: "Cache2NOC-m" (master port to NoC)
 *
 * ## Packet Sending Details:
 *
 * ### Local Delay (9 ticks for 256-byte request):
 * - Models internal cache processing time
 * - Includes data retrieval and packet preparation
 * - Packet departs cache at: arrival_time + local_delay
 *
 * ### Remote Delay (0 ticks):
 * - Models transmission time to NoC
 * - Set to 0 assuming cache is directly connected to router
 * - Realistic model could add wire delay
 *
 * ## Visitor Pattern Integration:
 * Requests arrive via the visitor pattern:
 * @code
 * // In CacheReqPacket.cc:
 * void CacheReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto* cacheSim = dynamic_cast<CacheSim*>(&simulator)) {
 *         cacheSim->handleNOCRequest(this, when);
 *         // Note: 'when' parameter carries arrival time
 *     }
 * }
 * @endcode
 *
 * # Lifecycle Integration
 *
 * @code{.unparsed}
 * Phase          │ Method Called         │ Action Taken
 * ───────────────┼───────────────────────┼────────────────────────────────
 * Construction   │ CacheSim()            │ Log "Constructing CacheSim"
 * Registration   │ N/A                   │ Added to SimTop via addSimulator()
 * Initialization │ init()                │ No-op (passive component)
 * Simulation     │ step()                │ Verbose logging (mostly idle)
 * Request        │ handleNOCRequest()    │ Generate and send responses
 * Cleanup        │ cleanup()             │ No-op (packets freed by framework)
 * Destruction    │ ~CacheSim()           │ Cleanup resources
 * @endcode
 *
 * # Usage Example in Custom Simulations
 *
 * @code
 * // 1. Create cache simulator
 * SimBase* cacheSim = new CacheSim("L1 Data Cache");
 *
 * // 2. Create NoC for communication
 * SimBase* nocSim = new NocSim("Network Router");
 *
 * // 3. Connect via SimChannel (bidirectional)
 * ChannelPortManager::ConnectPort(
 *     nocSim, cacheSim,
 *     "NOC2Cache-m", "NOC2Cache-s"  // Request path
 * );
 * ChannelPortManager::ConnectPort(
 *     cacheSim, nocSim,
 *     "Cache2NOC-m", "Cache2NOC-s"  // Response path
 * );
 *
 * // 4. Set up dependency chain
 * nocSim->addDownStream(cacheSim, "DSCache");
 * cacheSim->addUpStream(nocSim, "USNOC");
 *
 * // 5. Add to simulation
 * top->addSimulator(nocSim);
 * top->addSimulator(cacheSim);
 * @endcode
 *
 * # Extending for Realistic Cache Simulation
 *
 * ## 1. Adding Cache Line Storage:
 * @code
 * class RealisticCacheSim : public CacheSim {
 * private:
 *     struct CacheLine {
 *         bool valid;
 *         uint64_t tag;
 *         std::vector<uint8_t> data;
 *     };
 *     std::vector<CacheLine> cacheLines;
 *     int lineSize = 64;
 *
 * public:
 *     void handleNOCRequest(CacheReqPacket* req, Tick when) override {
 *         uint64_t addr = req->getAddr();
 *         int index = (addr / lineSize) % cacheLines.size();
 *         uint64_t tag = addr / (lineSize * cacheLines.size());
 *
 *         int* data;
 *         int delay;
 *
 *         if (cacheLines[index].valid && cacheLines[index].tag == tag) {
 *             // Cache hit
 *             data = new int(*(int*)cacheLines[index].data.data());
 *             delay = getRespDelay(req->getSize());
 *         } else {
 *             // Cache miss - fetch from memory
 *             data = new int(fetchFromMemory(addr));
 *             cacheLines[index] = {true, tag, ...};
 *             delay = memoryLatency + getRespDelay(req->getSize());
 *         }
 *
 *         auto resp = new CacheRespPacket(..., data, ...);
 *         sendPacketViaChannel("Cache2NOC-m", delay, 0, resp);
 *     }
 * };
 * @endcode
 *
 * ## 2. Multi-Level Cache Hierarchy:
 * @code
 * class L1Cache : public CacheSim {
 * private:
 *     SimBase* l2Cache;
 *
 * public:
 *     void handleNOCRequest(CacheReqPacket* req, Tick when) override {
 *         if (isHit(req->getAddr())) {
 *             // Respond from L1
 *             auto resp = new CacheRespPacket(...);
 *             sendPacketViaChannel("Cache2NOC-m", l1Delay, 0, resp);
 *         } else {
 *             // Forward to L2
 *             auto l2Req = new CacheReqPacket(...);
 *             sendPacketToL2("L1ToL2-m", 0, 0, l2Req);
 *         }
 *     }
 * };
 * @endcode
 *
 * ## 3. Cache Coherence Protocol:
 * @code
 * class CoherentCache : public CacheSim {
 * private:
 *     enum State { INVALID, SHARED, EXCLUSIVE, MODIFIED };
 *     std::map<uint64_t, State> lineStates;
 *
 * public:
 *     void handleNOCRequest(CacheReqPacket* req, Tick when) override {
 *         uint64_t addr = req->getAddr();
 *         State state = lineStates[addr];
 *
 *         if (req->isWrite()) {
 *             if (state == SHARED || state == EXCLUSIVE) {
 *                 // Send invalidate messages to other caches
 *                 broadcastInvalidate(addr);
 *             }
 *             lineStates[addr] = MODIFIED;
 *         }
 *
 *         // ... continue with normal cache operation
 *     }
 * };
 * @endcode
 *
 * ## 4. Performance Counters:
 * @code
 * class CacheSim : public CPPSimBase {
 * private:
 *     uint64_t totalAccesses = 0;
 *     uint64_t hits = 0;
 *     uint64_t misses = 0;
 *     uint64_t totalLatency = 0;
 *
 * public:
 *     void handleNOCRequest(CacheReqPacket* req, Tick when) override {
 *         totalAccesses++;
 *         bool isHit = checkCache(req->getAddr());
 *         if (isHit) hits++; else misses++;
 *
 *         int delay = isHit ? hitLatency : missLatency;
 *         totalLatency += delay;
 *
 *         // ... send response ...
 *     }
 *
 *     void finish() override {
 *         double hitRate = (double)hits / totalAccesses;
 *         double avgLatency = (double)totalLatency / totalAccesses;
 *         CLASS_INFO << "Hit Rate: " << (hitRate * 100) << "%";
 *         CLASS_INFO << "Average Latency: " << avgLatency << " ticks";
 *     }
 * };
 * @endcode
 *
 * # Design Rationale
 *
 * ## Why Include 'when' Parameter?
 * - Allows timing-dependent data generation
 * - Enables verification of correct packet arrival times
 * - Useful for debugging timing issues in full system
 *
 * ## Why Use Heap-Allocated Data?
 * @code
 * int* data = new int(100 + (int)_when);  // Heap allocation
 * @endcode
 * - Packet may outlive the handleNOCRequest() function
 * - Response sent with delay (packet travels through channels)
 * - Receiver (TrafficGenerator) will use the pointer later
 * - Framework handles deallocation after packet consumption
 *
 * ## Why Separate Local and Remote Delays?
 * - **Local delay**: Cache internal processing (tag check, data retrieval)
 * - **Remote delay**: Link/router delay to NoC (0 in current impl)
 * - Allows independent tuning of cache and interconnect models
 *
 * ## Why No Cache State?
 * This is a **functional testbench**, not a cycle-accurate cache model:
 * - Focuses on SimChannel communication patterns
 * - Demonstrates timing calculations
 * - Provides predictable data for verification
 * - Extensible to full cache implementation (see examples above)
 *
 * @see CacheSim.hh For class declaration and configuration constants
 * @see NocSim For upstream component (request source)
 * @see CachePacket For packet structure definitions
 * @see testSimChannel.cc For complete system integration
 *
 * @author ACAL Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note This is a minimal cache simulator for testing. Production cache models
 *       should include tag arrays, data arrays, replacement policies, coherence
 *       protocols, and prefetchers.
 *
 * @warning Data pointer ownership transfers to the packet. Do not manually delete
 *          the data pointer after creating CacheRespPacket - the framework will
 *          handle cleanup when the packet is consumed.
 */

#include "CacheSim.hh"

void CacheSim::handleNOCRequest(CacheReqPacket* _cacheReqPkt, Tick _when) {
	// Create CacheRespPacket
	int* data         = new int(100 + (int)_when);
	auto cacheRespPkt = new CacheRespPacket(CachePktTypeEnum::TEST, data, _cacheReqPkt->getTransactionId());

	// Setting remote & local delay
	auto local_delay  = this->getRespDelay(_cacheReqPkt->getSize());
	auto remote_delay = 0;

	// Send CacheRespPacket to NoC
	CLASS_INFO << "NocSim sendPacketViaChannel with local latency = " << local_delay
	           << ", remote latency = " << remote_delay;
	this->sendPacketViaChannel("Cache2NOC-m", local_delay, remote_delay, cacheRespPkt);
}
