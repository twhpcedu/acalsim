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
 * @file testChannel.cc
 * @brief Channel-based communication example - TrafficGenerator-NOC-Cache architecture
 *
 * This example demonstrates **channel-based asynchronous communication** in ACALSim through
 * a network-on-chip (NOC) system connecting traffic generators to cache simulators. It showcases
 * lock-free dual-queue ping-pong buffers, event-driven packet routing, callback-based responses,
 * and multi-tick configurable latency modeling.
 *
 * **System Architecture:**
 * ```
 * ┌───────────────────────────┐       ┌───────────────────────────┐       ┌───────────────────────────┐
 * │    TrafficGenerator       │       │         NocSim            │       │        CacheSim           │
 * │                           │       │    (Network-on-Chip)      │       │                           │
 * │  Event-Driven             │       │                           │       │                           │
 * │  Request Generator        │       │  Packet Forwarding        │       │  Request Processing       │
 * │                           │       │  & Routing                │       │  & Response Generation    │
 * │  MasterChannelPort        │       │                           │       │                           │
 * │  (DSNOC - send)           │       │  Bidirectional            │       │  SlaveChannelPort         │
 * │                           │       │  Channel Ports            │       │  (USNOC - receive)        │
 * │  SlaveChannelPort         │       │                           │       │                           │
 * │  (USTrafficGenerator)     │       │  Request Queue            │       │  MasterChannelPort        │
 * │  (receive)                │       │  Tracker                  │       │  (USNOC - send)           │
 * └───────────────────────────┘       └───────────────────────────┘       └───────────────────────────┘
 *             ▲ │                                 ▲ │                                 ▲ │
 *      req ───┘ └─── rsp                   req ──┘ └─── rsp                   req ──┘ └─── rsp
 *   (Channel)    (Channel)              (Channel)    (Channel)              (Channel)    (Channel)
 *
 * Channel Connection Graph:
 *   TrafficGenerator.DSNOC ──────────► NocSim.USTrafficGenerator  (Requests:  TG → NOC)
 *   NocSim.DSCache ───────────────────► CacheSim.USNOC            (Requests:  NOC → Cache)
 *   CacheSim.USNOC ───────────────────► NocSim.DSCache            (Responses: Cache → NOC)
 *   NocSim.USTrafficGenerator ────────► TrafficGenerator.DSNOC    (Responses: NOC → TG)
 * ```
 *
 * **Communication Flow (Request Path):**
 * ```
 * Tick 1-3:
 *   TrafficGenerator:
 *     1. TrafficEvent scheduled (init phase)
 *     2. Create NocReqPacket with addr/size
 *     3. Wrap in NocReqEvent with callback lambda
 *     4. Wrap in EventPacket with target tick (tick + 10)
 *     5. Push to MasterChannelPort "DSNOC"
 *
 *   Framework (Phase 2):
 *     6. ChannelPortManager transfers packet
 *     7. NocSim.USTrafficGenerator SlaveChannelPort receives EventPacket
 *     8. Framework schedules event for target tick
 *
 * Tick 11 (tick + 10):
 *   NocSim:
 *     9. NocReqEvent.process() invoked
 *     10. NocReqPacket.visit(NocSim) called
 *     11. NocSim::handleTGRequest() extracts packet
 *     12. Store request in reqQueue with transaction ID
 *     13. Create CacheReqPacket (unwrapped from NOC layer)
 *     14. Create CacheReqEvent with callback to nocReqCallback
 *     15. Wrap in EventPacket with NOC latency (tick + 1)
 *     16. Push to MasterChannelPort "DSCache"
 *
 *   Framework (Phase 2):
 *     17. CacheSim.USNOC SlaveChannelPort receives EventPacket
 *
 * Tick 12 (tick + 1):
 *   CacheSim:
 *     18. CacheReqEvent.process() invoked
 *     19. CacheReqPacket.visit(CacheSim) called
 *     20. CacheSim::handleNOCRequest() processes request
 *     21. Create NocRespPacket with data
 *     22. Execute caller's callback immediately (nocReqCallback)
 * ```
 *
 * **Communication Flow (Response Path):**
 * ```
 * Tick 12 (continuation):
 *   NocSim (via callback):
 *     23. NocSim::nocReqCallback() executed
 *     24. Retrieve original NocReqPacket from reqQueue
 *     25. Extract caller's callback (TrafficEvent::NocRespHandler)
 *     26. Create NocRespEvent with TG's callback
 *     27. Wrap in EventPacket with NOC latency (tick + 1)
 *     28. Push to MasterChannelPort "USTrafficGenerator"
 *
 *   Framework (Phase 2):
 *     29. TrafficGenerator.DSNOC SlaveChannelPort receives response
 *
 * Tick 13 (tick + 1):
 *   TrafficGenerator:
 *     30. NocRespEvent.process() invoked
 *     31. TrafficEvent::NocRespHandler() executed
 *     32. Create TrafficRespEvent with final data
 *     33. Wrap in EventPacket with TG latency (tick + 1)
 *     34. Push to MasterChannelPort "USTrafficGenerator"
 *
 * Tick 14 (tick + 1):
 *   TrafficGenerator:
 *     35. TrafficRespEvent.process() logs transaction completion
 *     36. Chrome trace records transaction finish
 * ```
 *
 * **Channel vs Port Architecture Comparison:**
 *
 * | Feature                    | Port (testSimPort)           | Channel (testChannel)              |
 * |----------------------------|------------------------------|------------------------------------|
 * | **Communication Model**    | Synchronous push/pop         | Asynchronous message passing       |
 * | **Data Transfer**          | Direct packet transfer       | Event-wrapped packet transfer      |
 * | **Timing Control**         | Immediate (same tick)        | Multi-tick configurable latency    |
 * | **Backpressure**           | Queue full returns false     | Lock-free dual-queue buffers       |
 * | **Synchronization**        | SimPortManager Phase 2       | ChannelPortManager Phase 2         |
 * | **Packet Type**            | SimPacket* (raw pointers)    | EventPacket (wraps events)         |
 * | **Callback Support**       | Manual tracking required     | Built-in callback chaining         |
 * | **Typical Use Case**       | CPU-Bus-Memory (low-level)   | NOC/Network (high-level protocols) |
 * | **Latency Modeling**       | Explicit event scheduling    | Embedded in EventPacket            |
 * | **Memory Management**      | RecycleContainer             | new/delete (event auto-cleanup)    |
 *
 * **Dual-Queue Ping-Pong Buffer Architecture:**
 * ```
 * MasterChannelPort (Sender):                  SlaveChannelPort (Receiver):
 * ┌─────────────────────────┐                  ┌─────────────────────────┐
 * │  pushQueue (write)      │                  │  popQueue (read)        │
 * │  [Pkt1][Pkt2][...]      │ ──────────────►  │  [Pkt1][Pkt2][...]      │
 * └─────────────────────────┘                  └─────────────────────────┘
 *          │                                              │
 *          │ swap() in Phase 2                            │ pop() in Phase 1
 *          ▼                                              ▼
 * ┌─────────────────────────┐                  ┌─────────────────────────┐
 * │  (empty after swap)     │  ◄──────────────  │  (empty after pop)      │
 * └─────────────────────────┘                  └─────────────────────────┘
 *
 * Benefits:
 *   - Lock-free: No mutex required (single writer per queue)
 *   - Zero-copy: Pointer swap instead of data copy
 *   - Decoupled: Sender and receiver operate independently
 *   - Predictable: Deterministic tick-based delivery
 * ```
 *
 * **Key Features Demonstrated:**
 *
 * 1. **Channel-Based Communication:**
 *    - MasterChannelPort for asynchronous sending
 *    - SlaveChannelPort for asynchronous receiving
 *    - EventPacket wrapper for tick-precise delivery
 *    - Lock-free dual-queue architecture
 *
 * 2. **Event-Driven Packet Routing:**
 *    - TrafficEvent generates requests
 *    - NocReqEvent/NocRespEvent handle routing
 *    - CacheReqEvent processes memory operations
 *    - TrafficRespEvent completes transactions
 *
 * 3. **Callback-Based Response Handling:**
 *    - Lambda callbacks capture transaction context
 *    - Callback chaining through packet layers (TG → NOC → Cache → NOC → TG)
 *    - Asynchronous response propagation
 *
 * 4. **Multi-Tick Latency Modeling:**
 *    - TrafficGenerator: 10-tick request generation + 1-tick response
 *    - NOC: 1-tick routing latency (bidirectional)
 *    - Cache: Dynamic latency based on packet size
 *
 * 5. **Visitor Pattern Packet Processing:**
 *    - NocReqPacket::visit(NocSim) → handleTGRequest()
 *    - CacheReqPacket::visit(CacheSim) → handleNOCRequest()
 *    - Type-safe dynamic dispatch
 *
 * 6. **Outstanding Request Tracking:**
 *    - NocSim::reqQueue stores in-flight requests
 *    - Transaction ID matching for responses
 *    - UnorderedRequestQueue container
 *
 * **ChannelPortManager Connection API:**
 * ```cpp
 * // Bidirectional connection setup (four separate channels):
 * ChannelPortManager::ConnectPort(trafficGenerator, nocSim, "DSNOC", "USTrafficGenerator");
 *   // Creates: TG.DSNOC (Master) → NOC.USTrafficGenerator (Slave)
 *
 * ChannelPortManager::ConnectPort(nocSim, cacheSim, "DSCache", "USNOC");
 *   // Creates: NOC.DSCache (Master) → Cache.USNOC (Slave)
 *
 * ChannelPortManager::ConnectPort(cacheSim, nocSim, "USNOC", "DSCache");
 *   // Creates: Cache.USNOC (Master) → NOC.DSCache (Slave)
 *
 * ChannelPortManager::ConnectPort(nocSim, trafficGenerator, "USTrafficGenerator", "DSNOC");
 *   // Creates: NOC.USTrafficGenerator (Master) → TG.DSNOC (Slave)
 *
 * // Note: Each ConnectPort() creates a unidirectional channel
 * //       Bidirectional communication requires two ConnectPort() calls
 * ```
 *
 * **Command-Line Options:**
 * ```bash
 * # Run with default parameters
 * ./testChannel
 *
 * # Combine with framework options
 * ./testChannel --max-tick 10000 --threadmanager V3 --verbose
 *
 * # Enable trace output
 * ./testChannel --trace-enable --trace-file trace.log
 *
 * # Use Chrome tracing for visualization
 * ./testChannel --chrome-trace trace.json
 * # Then open chrome://tracing in Chrome browser
 * ```
 *
 * **Expected Output:**
 * ```
 * [TrafficGenerator] TrafficEvent Processed.
 * [TrafficGenerator] Traffic Event Processed and sendNocEvent with transaction id: 1 at Tick=1
 * [NocSim] Process NocReqEvent with transaction id: 1 at Tick=11
 * [NocSim] Thread 0x... executes NocReqEvent::nocReqCallback() transaction id: 1 at Tick=12
 * [CacheSim] Process CacheReqEvent with transaction id: 0 at Tick=12
 * [NocSim] Process NocRespEvent with transaction id: 1 at Tick=13
 * [TrafficGenerator] Thread 0x... executes NocRespHandler with transaction id: 1 at Tick=13
 * [TrafficGenerator] Transaction Finish ! Data = 100
 * ```
 *
 * **Learning Outcomes:**
 * - Understand channel-based vs port-based communication models
 * - Implement asynchronous message passing with callbacks
 * - Use EventPacket for tick-precise event delivery
 * - Chain callbacks across multiple simulator layers
 * - Model network latencies with configurable delays
 * - Apply visitor pattern for type-safe packet routing
 * - Manage outstanding requests with transaction IDs
 *
 * **Performance Characteristics:**
 * - Total latency per request: 10 (TG) + 1 (NOC) + 1 (Cache) + 1 (NOC) + 1 (TG) = 14 ticks
 * - Asynchronous pipelining: Multiple requests can overlap
 * - Lock-free operation: No mutex contention
 * - Zero-copy channel transfer: Pointer swapping only
 *
 * **Extending This Example:**
 * 1. Add multiple traffic generators with different patterns
 * 2. Implement NOC routing with mesh topology
 * 3. Add cache coherence protocol (MSI/MESI)
 * 4. Support multiple cache levels (L1/L2/L3)
 * 5. Implement credit-based flow control
 * 6. Add deadlock detection and recovery
 *
 * @see TrafficGenerator For event-driven request generation
 * @see NocSim For packet routing and forwarding
 * @see CacheSim For memory request processing
 * @see TrafficEvent For request generation events
 * @see NocReqEvent/NocRespEvent For NOC routing events
 * @see CacheReqEvent For cache processing events
 * @see NocReqPacket/NocRespPacket For NOC packet structures
 * @see CacheReqPacket For cache packet structures
 */

#include "ACALSim.hh"
using namespace acalsim;

// Step 1 include header files of the simulator classes
#include "CacheSim.hh"
#include "NocSim.hh"
#include "TrafficGenerator.hh"

// Step 2. Inherit SimTop to create your own top-level simulation class
class TestChannel : public SimTop {
public:
	TestChannel() : SimTop() {
		this->traceCntr.run(0, &SimTraceContainer::setFilePath, "trace", "src/testChannel/trace");
		this->traceCntr.run(1, &SimTraceContainer::setFilePath, "chrome-trace", "src/testChannel/trace");
	}

	void registerSimulators() override {
		// Create simulators
		SimBase* trafficGenerator = (SimBase*)new TrafficGenerator("Traffic Generator");
		SimBase* nocSim           = (SimBase*)new NocSim("Noxim");
		SimBase* cacheSim         = (SimBase*)new CacheSim("Cache Simulator");

		// register Simulators
		this->addSimulator(trafficGenerator);
		this->addSimulator(nocSim);
		this->addSimulator(cacheSim);

		ChannelPortManager::ConnectPort(trafficGenerator, nocSim, "DSNOC", "USTrafficGenerator");
		ChannelPortManager::ConnectPort(nocSim, cacheSim, "DSCache", "USNOC");
		ChannelPortManager::ConnectPort(cacheSim, nocSim, "USNOC", "DSCache");
		ChannelPortManager::ConnectPort(nocSim, trafficGenerator, "USTrafficGenerator", "DSNOC");

		// connect simulators
		trafficGenerator->addDownStream(nocSim, "DSNOC");
		nocSim->addDownStream(cacheSim, "DSCache");
		cacheSim->addUpStream(nocSim, "USNOC");
		nocSim->addUpStream(trafficGenerator, "USTrafficGenerator");
	}
};

int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	top = std::make_shared<TestChannel>();
	top->init(argc, argv);
	top->run();
	top->finish();
	return 0;
}
