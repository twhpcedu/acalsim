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
 * @file TrafficGenerator.cc
 * @brief Traffic Generator Implementation for NoC Memory Request Simulation
 *
 * @details
 * This file implements the TrafficGenerator class, which serves as the request initiator
 * in the testSimChannel memory subsystem simulation. It models a simplified CPU or memory
 * controller that generates memory access requests and handles responses from the cache
 * hierarchy via the Network-on-Chip (NoC).
 *
 * # Component Role in System Architecture
 *
 * The TrafficGenerator acts as the top-level component in the memory request chain:
 *
 * @code{.unparsed}
 *     ┌─────────────────────────────────────┐
 *     │      TrafficGenerator (This)        │
 *     │   ┌─────────────────────────────┐   │
 *     │   │  Request Generation Engine  │   │
 *     │   │  - Transaction ID tracking  │   │
 *     │   │  - Timing control           │   │
 *     │   └──────────┬──────────────────┘   │
 *     │              ↓                       │
 *     │   ┌─────────────────────────────┐   │
 *     │   │  SimChannel Master Port     │   │
 *     │   │  "TG2NOC-m" (outgoing)      │   │
 *     │   └──────────┬──────────────────┘   │
 *     └──────────────┼──────────────────────┘
 *                    │ NocReqPacket
 *                    ↓
 *              ┌──────────┐
 *              │  NocSim  │
 *              └─────┬────┘
 *                    ⋮
 *                    ↑ NocRespPacket
 *     ┌──────────────┼──────────────────────┐
 *     │   ┌─────────────────────────────┐   │
 *     │   │  SimChannel Slave Port      │   │
 *     │   │  "NOC2TG-s" (incoming)      │   │
 *     │   └──────────┬──────────────────┘   │
 *     │              ↓                       │
 *     │   ┌─────────────────────────────┐   │
 *     │   │  Response Handler           │   │
 *     │   │  - Data validation          │   │
 *     │   │  - Latency measurement      │   │
 *     │   └─────────────────────────────┘   │
 *     │      TrafficGenerator               │
 *     └─────────────────────────────────────┘
 * @endcode
 *
 * # Request Generation Strategy
 *
 * ## Initialization Phase (init())
 * The traffic generator creates a small batch of memory requests during initialization:
 * - **Request Count**: 2 transactions (configurable via loop)
 * - **Transaction IDs**: 0, 1 (sequential integers)
 * - **Request Type**: TEST type (NocPktTypeEnum::TEST)
 * - **Access Pattern**: Fixed address (0) and size (256 bytes)
 *
 * ## Timing Model
 * Each request uses a dual-delay timing model:
 *
 * ### Local Delay (Transaction-Dependent):
 * @code
 * local_delay = tid × 2 + 1
 * @endcode
 * - tid=0: 1 tick local delay
 * - tid=1: 3 ticks local delay
 * - Simulates variable pipeline stages or instruction scheduling
 *
 * ### Remote Delay (Constant):
 * @code
 * remote_delay = tgRemoteDelay = 10 ticks
 * @endcode
 * - Models network propagation delay to NoC
 * - All requests experience same network latency
 *
 * ### Combined Effect:
 * Request arrives at NocSim at: `current_tick + local_delay + remote_delay`
 *
 * @code{.unparsed}
 * Timeline for tid=0:
 *   Tick 0: sendNoCRequest() called
 *   Tick 1: Packet leaves TrafficGenerator (local_delay=1)
 *   Tick 11: Packet arrives at NocSim (remote_delay=10)
 *
 * Timeline for tid=1:
 *   Tick 0: sendNoCRequest() called
 *   Tick 3: Packet leaves TrafficGenerator (local_delay=3)
 *   Tick 13: Packet arrives at NocSim (remote_delay=10)
 * @endcode
 *
 * # Request Packet Structure
 *
 * Each NocReqPacket contains:
 * - **Type**: NocPktTypeEnum::TEST (placeholder for read/write operations)
 * - **Address**: 0 (memory address to access)
 * - **Size**: 256 bytes (cache line size)
 * - **Transaction ID**: Unique identifier for request-response matching
 *
 * @code
 * // Request creation example
 * auto nocReqPkt = new NocReqPacket(
 *     NocPktTypeEnum::TEST,  // Operation type
 *     0,                     // Memory address
 *     256,                   // Transfer size
 *     tid                    // Transaction ID
 * );
 * @endcode
 *
 * # Response Handling
 *
 * ## handleNoCRespond() Workflow
 * When a NocRespPacket arrives from the NoC:
 *
 * 1. **Thread Safety Logging**: Records which thread processes the response
 * 2. **Transaction ID Extraction**: Identifies which request this responds to
 * 3. **Timing Information**: Logs arrival tick for latency analysis
 * 4. **Data Retrieval**: Extracts returned data payload
 *
 * @code
 * // Response example for tid=0:
 * // Thread 0x12345 executes handleNoCRespond
 * // transaction id: 0
 * // at Tick = 20
 * // get data = 111  (100 + arrival_time@CacheSim)
 * @endcode
 *
 * ## Data Validation Pattern
 * The response data encodes timing information:
 * @code
 * response_data = 100 + tick_when_cache_received_request
 * @endcode
 * This allows verification that:
 * - Requests reached the cache at expected times
 * - Response routing preserved data integrity
 * - No packet corruption occurred during transmission
 *
 * # SimChannel Communication Details
 *
 * ## Port Configuration
 * - **Outgoing Port**: "TG2NOC-m" (master port to NoC)
 * - **Incoming Port**: "NOC2TG-s" (slave port from NoC) - implicitly connected
 *
 * ## Packet Sending API
 * The generator uses the SimBase::sendPacketViaChannel() method:
 * @code
 * this->sendPacketViaChannel(
 *     "TG2NOC-m",      // Port name (master side)
 *     local_delay,     // Ticks before packet leaves this component
 *     remote_delay,    // Ticks for packet to reach destination
 *     nocReqPkt        // Packet pointer (polymorphic SimPacket)
 * );
 * @endcode
 *
 * ## Packet Receiving (Visitor Pattern)
 * Response packets arrive via the visitor pattern:
 * 1. NocRespPacket::visit() is called by the framework
 * 2. visit() method casts the simulator to TrafficGenerator
 * 3. visit() calls handleNoCRespond() with the packet
 *
 * # Simulation Lifecycle Integration
 *
 * @code{.unparsed}
 * Phase          │ Method Called        │ Action Taken
 * ───────────────┼──────────────────────┼──────────────────────────────────
 * Construction   │ TrafficGenerator()   │ Initialize base class (CPPSimBase)
 * Registration   │ N/A                  │ Added to SimTop via addSimulator()
 * Initialization │ init()               │ Generate and send 2 test requests
 * Simulation     │ step()               │ Verbose logging (mostly idle)
 * Response       │ handleNoCRespond()   │ Process returned data
 * Cleanup        │ cleanup()            │ No-op (packets freed by framework)
 * Destruction    │ ~TrafficGenerator()  │ Cleanup resources
 * @endcode
 *
 * # Usage Example in Custom Simulations
 *
 * @code
 * // 1. Create the traffic generator
 * SimBase* trafficGenerator = new TrafficGenerator("CPU Traffic Gen");
 *
 * // 2. Create NoC for communication
 * SimBase* nocSim = new NocSim("Network Router");
 *
 * // 3. Connect via SimChannel
 * ChannelPortManager::ConnectPort(
 *     trafficGenerator, nocSim,
 *     "TG2NOC-m", "TG2NOC-s"
 * );
 *
 * // 4. Set up dependency chain (for simulation ordering)
 * trafficGenerator->addDownStream(nocSim, "DSNOC");
 *
 * // 5. Add to simulation topology
 * top->addSimulator(trafficGenerator);
 * top->addSimulator(nocSim);
 *
 * // 6. Run simulation
 * top->init();   // Triggers TrafficGenerator::init() → sends requests
 * top->run();    // Processes responses as they arrive
 * top->finish(); // Cleanup
 * @endcode
 *
 * # Extending for Advanced Traffic Patterns
 *
 * ## 1. Realistic Memory Access Patterns:
 * @code
 * void init() override {
 *     // Random access pattern
 *     for (int i = 0; i < requestCount; ++i) {
 *         int addr = randomAddressGenerator();
 *         int size = {64, 128, 256}[rand() % 3];  // Variable sizes
 *         sendMemoryRequest(addr, size, i);
 *     }
 * }
 * @endcode
 *
 * ## 2. Time-Based Traffic Generation:
 * @code
 * void step() override {
 *     Tick currentTick = top->getGlobalTick();
 *     if (currentTick % requestInterval == 0) {
 *         sendNoCRequest(nextTransactionID++);
 *     }
 * }
 * @endcode
 *
 * ## 3. Trace-Driven Simulation:
 * @code
 * void init() override {
 *     loadMemoryTrace("traces/spec2017.trace");
 *     for (auto& access : memoryTrace) {
 *         scheduleEvent(access.tick, [this, access]() {
 *             sendRequest(access.addr, access.size, access.tid);
 *         });
 *     }
 * }
 * @endcode
 *
 * # Performance Metrics Collection
 *
 * To add statistics tracking:
 * @code
 * class TrafficGenerator : public CPPSimBase {
 * private:
 *     uint64_t totalRequests = 0;
 *     uint64_t totalResponses = 0;
 *     uint64_t totalLatency = 0;
 *     std::map<int, Tick> requestSentTimes;
 *
 * public:
 *     void sendNoCRequest(int tid) {
 *         requestSentTimes[tid] = top->getGlobalTick();
 *         totalRequests++;
 *         // ... send packet ...
 *     }
 *
 *     void handleNoCRespond(NocRespPacket* resp) {
 *         Tick latency = top->getGlobalTick() - requestSentTimes[resp->getTid()];
 *         totalLatency += latency;
 *         totalResponses++;
 *         CLASS_INFO << "Request latency: " << latency << " ticks";
 *     }
 *
 *     void finish() {
 *         double avgLatency = (double)totalLatency / totalResponses;
 *         CLASS_INFO << "Average latency: " << avgLatency << " ticks";
 *     }
 * };
 * @endcode
 *
 * # Design Rationale
 *
 * ## Why init() for Request Generation?
 * - Ensures requests are sent at tick 0 (simulation start)
 * - All simulators complete initialization before any packets arrive
 * - Predictable timing for verification and debugging
 *
 * ## Why Separate Local and Remote Delays?
 * - **Local delay**: Models internal processing (CPU pipeline, memory controller queue)
 * - **Remote delay**: Models physical transport (wire delay, router hops)
 * - Allows independent tuning of component and network characteristics
 *
 * ## Why Visitor Pattern for Responses?
 * - Type-safe packet routing without manual type casting
 * - Packets encapsulate their handling logic
 * - Easily extensible for new packet types
 *
 * @see TrafficGenerator.hh For class declaration and configuration constants
 * @see NocSim For next component in the request chain
 * @see NocPacket For packet structure definitions
 * @see testSimChannel.cc For complete system integration example
 *
 * @author ACAL Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note This is a minimal traffic generator for testing. Production simulators
 *       should include realistic access patterns, multiple memory types, and
 *       detailed performance statistics.
 *
 * @warning Transaction IDs must be unique for in-flight requests. Reusing IDs
 *          before responses arrive will cause incorrect request-response matching
 *          in the NoC request queue.
 */

#include "TrafficGenerator.hh"

void TrafficGenerator::init() {
	for (Tick i = 0; i < 2; ++i) { sendNoCRequest(i); }
}

void TrafficGenerator::sendNoCRequest(int _tid) {
	CLASS_INFO << "Schedule sendNoCRequest with transaction id : " << _tid << " at Tick = " << top->getGlobalTick();

	// Create NocReqPacket
	int  _size     = 256;
	int  _addr     = 0;
	auto nocReqPkt = new NocReqPacket(NocPktTypeEnum::TEST, _addr, _size, _tid);

	// Setting remote & local delay
	auto local_delay  = _tid * 2 + 1;
	auto remote_delay = this->getRemoteDelay();

	// Send NocReqPacket to NoC
	CLASS_INFO << "TrafficGenerator sendPacketViaChannel with local latency = " << local_delay
	           << ", remote latency = " << remote_delay;
	this->sendPacketViaChannel("TG2NOC-m", local_delay, remote_delay, nocReqPkt);
}

void TrafficGenerator::handleNoCRespond(NocRespPacket* _nocRespPkt) {
	CLASS_INFO << "Thread " << std::this_thread::get_id()
	           << " executes handleNoCRespond with transaction id : " << _nocRespPkt->getTransactionId()
	           << " at Tick = " << top->getGlobalTick() << ", get data = " << *_nocRespPkt->getData();
}
