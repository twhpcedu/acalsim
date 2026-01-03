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
 * @file testSimChannel.cc
 * @brief SimChannel-based NoC-Cache Communication System Test
 *
 * @details
 * This file demonstrates an advanced usage of the ACALSim framework featuring a complete
 * Network-on-Chip (NoC) and cache memory simulation system using SimChannel for inter-component
 * communication. It showcases bidirectional packet routing, transaction tracking, and
 * multi-layer memory hierarchy simulation.
 *
 * # Architecture Overview
 *
 * The system simulates a typical computer architecture's memory subsystem with three main
 * components communicating through SimChannel ports:
 *
 * @code{.unparsed}
 *     ┌──────────────────┐
 *     │ TrafficGenerator │  (Generates memory requests)
 *     │   (CPU Model)    │
 *     └────────┬─────────┘
 *              │ TG2NOC (NocReqPacket)
 *              ↓
 *     ┌────────────────┐ NOC2Cache (CacheReqPacket)
 *     │    NocSim      │──────────────────────────────────┐
 *     │  (NoC Router)  │                                  ↓
 *     └────────┬───────┘                          ┌───────────────┐
 *              ↑                                  │   CacheSim    │
 *              │ NOC2TG (NocRespPacket)           │ (Cache Memory)│
 *              │                                  └───────┬───────┘
 *              └──────────────────────────────────────────┘
 *                            Cache2NOC (CacheRespPacket)
 * @endcode
 *
 * # Communication Flow
 *
 * ## Request Path (Forward):
 * 1. **TrafficGenerator → NocSim**: Sends NocReqPacket via "TG2NOC-m" port
 *    - Contains: address, size, transaction ID
 *    - Timing: Local delay = (tid * 2 + 1), Remote delay = 10 ticks
 *
 * 2. **NocSim → CacheSim**: Forwards as CacheReqPacket via "NOC2Cache-m" port
 *    - NoC stores original request in reqQueue (indexed by transaction ID)
 *    - Packet transformation: NocReqPacket → CacheReqPacket
 *    - Timing: Immediate forwarding (0 local, 0 remote delay)
 *
 * ## Response Path (Backward):
 * 3. **CacheSim → NocSim**: Returns CacheRespPacket via "Cache2NOC-m" port
 *    - Contains: data payload (100 + arrival time), transaction ID
 *    - Timing: Latency = 1 + (size + 1) / 32 ticks
 *
 * 4. **NocSim → TrafficGenerator**: Returns NocRespPacket via "NOC2TG-m" port
 *    - NoC retrieves original request from reqQueue
 *    - Packet transformation: CacheRespPacket → NocRespPacket
 *    - Timing: Immediate return (0 local, 0 remote delay)
 *
 * # Key Features Demonstrated
 *
 * ## 1. Bidirectional Communication
 * Unlike testChannel which shows unidirectional flow, this example demonstrates:
 * - Forward request path (TrafficGenerator → NoC → Cache)
 * - Backward response path (Cache → NoC → TrafficGenerator)
 * - Transaction ID tracking to match requests with responses
 *
 * ## 2. Packet Transformation
 * NoC acts as a protocol converter:
 * - **Request transformation**: NocReqPacket → CacheReqPacket
 * - **Response transformation**: CacheRespPacket → NocRespPacket
 * - Maintains transaction state in UnorderedRequestQueue
 *
 * ## 3. Advanced Timing Models
 * Each component models realistic hardware timing:
 * - **TrafficGenerator**: Variable request injection (tid-dependent)
 * - **NocSim**: Network routing latency (bandwidth-limited)
 * - **CacheSim**: Memory access time (latency + bandwidth model)
 *
 * ## 4. SimChannel Port Naming Convention
 * Ports follow a directional naming scheme:
 * - **Master ports** (suffix "-m"): Initiator side (sends first)
 * - **Slave ports** (suffix "-s"): Responder side (receives first)
 * - Example: "TG2NOC-m" (TrafficGenerator's master) ↔ "TG2NOC-s" (NoC's slave)
 *
 * # Comparison with testChannel
 *
 * | Feature                  | testChannel          | testSimChannel (This)     |
 * |--------------------------|----------------------|---------------------------|
 * | Communication Pattern    | Unidirectional       | Bidirectional             |
 * | Component Count          | 3 (A→B→C)            | 3 (TG↔NoC↔Cache)          |
 * | Packet Types             | Single type          | 4 types (Req/Resp × 2)    |
 * | Transaction Tracking     | None                 | Queue-based with IDs      |
 * | Timing Complexity        | Simple delays        | Multi-stage timing model  |
 * | Protocol Conversion      | No                   | Yes (NoC as middleware)   |
 * | Real-world Model         | Abstract pipeline    | Memory subsystem          |
 * | Upstream/Downstream      | Linear chain         | Cyclic dependency graph   |
 *
 * # Upstream/Downstream Topology
 *
 * The component connectivity uses both upstream and downstream relationships to enable
 * bidirectional communication:
 *
 * @code{.unparsed}
 *   TrafficGenerator
 *         ↓ downstream (DSNOC)
 *       NocSim
 *         ↓ downstream (DSCache)
 *      CacheSim
 *         ↑ upstream (USNOC)
 *       NocSim
 *         ↑ upstream (USTrafficGenerator)
 *   TrafficGenerator
 * @endcode
 *
 * This forms a request-response cycle where:
 * - Downstream connections carry requests (forward path)
 * - Upstream connections carry responses (backward path)
 *
 * # Transaction Lifecycle Example
 *
 * @code
 * // Tick 0: TrafficGenerator initializes and sends 2 requests
 * init() {
 *     sendNoCRequest(tid=0);  // local_delay=1, remote_delay=10
 *     sendNoCRequest(tid=1);  // local_delay=3, remote_delay=10
 * }
 *
 * // Tick 11: NocSim receives tid=0 request
 * handleTGRequest(nocReqPkt) {
 *     reqQueue->add(0, nocReqPkt);  // Store for matching
 *     send CacheReqPacket to Cache;
 * }
 *
 * // Tick 11: CacheSim receives request
 * handleNOCRequest(cacheReqPkt, when=11) {
 *     data = new int(100 + 11);  // data = 111
 *     local_delay = 1 + (256+1)/32 = 9;
 *     send CacheRespPacket with delay=9;
 * }
 *
 * // Tick 20: NocSim receives cache response
 * handleCacheRespond(cacheRespPkt) {
 *     nocReqPkt = reqQueue->get(0);  // Retrieve original request
 *     send NocRespPacket to TrafficGenerator;
 * }
 *
 * // Tick 20: TrafficGenerator receives response
 * handleNoCRespond(nocRespPkt) {
 *     // tid=0, data=111
 * }
 * @endcode
 *
 * # Usage Example
 *
 * Build and run the test:
 * @code{.sh}
 * # Build the test
 * cd acalsim-workspace
 * make testSimChannel
 *
 * # Run the simulation
 * ./build/testSimChannel
 *
 * # Expected output shows:
 * # - TrafficGenerator sending 2 requests (tid 0, 1)
 * # - NocSim forwarding requests to cache
 * # - CacheSim responding with data
 * # - NocSim routing responses back
 * # - TrafficGenerator receiving responses
 * @endcode
 *
 * # Performance Characteristics
 *
 * ## Bandwidth Modeling:
 * - **NoC Bandwidth**: 32 bytes/tick (nocBandwidth constant)
 * - **Cache Memory Bandwidth**: 32 bytes/tick (cacheMemoryBandwidth)
 * - **Cache Latency**: 1 tick base + bandwidth-dependent transfer time
 *
 * ## Latency Breakdown (256-byte request):
 * 1. TrafficGenerator local: (tid × 2 + 1) ticks
 * 2. TrafficGenerator remote: 10 ticks
 * 3. NoC routing: ~0 ticks (immediate)
 * 4. Cache access: 1 + (256+1)/32 = 9 ticks
 * 5. NoC return: ~0 ticks
 * Total for tid=0: 1 + 10 + 0 + 9 + 0 = 20 ticks
 *
 * # Implementation Guidelines
 *
 * ## When to Use This Pattern:
 * - Building memory hierarchy simulators (caches, DRAM controllers)
 * - Modeling network protocols with request-response semantics
 * - Simulating systems requiring transaction tracking
 * - Testing multi-hop packet routing with protocol conversion
 *
 * ## Key Design Patterns:
 * 1. **Transaction Queue**: Store requests to match with responses
 * 2. **Visitor Pattern**: Packets know how to handle themselves (visit() methods)
 * 3. **Port Pairing**: Master-slave port pairs for bidirectional channels
 * 4. **Timing Separation**: Local vs. remote delays for accurate modeling
 *
 * @see TrafficGenerator For traffic generation and request handling
 * @see NocSim For network routing and packet transformation
 * @see CacheSim For cache memory modeling
 * @see NocPacket For NoC packet structures
 * @see CachePacket For cache packet structures
 * @see testChannel.cc For comparison with simpler unidirectional pattern
 *
 * @author ACAL Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note This example focuses on SimChannel communication patterns. For production
 *       cache simulations, add cache line management, coherence protocols, and
 *       replacement policies.
 *
 * @warning Transaction IDs must be unique across all in-flight requests to prevent
 *          response mismatches in the NoC request queue.
 */

/* --------------------------------------------------------------------------------------
 *  A test template to demonstrate how to create your own simulation using this framework
 *  Step 1. Inherit SimBase to create your own simulator classes
 *  Step 2. Inherit SimTop to create your own top-level simulation class
 *          Add all the simulators one by one using the SimTop::addSimulation() API
 *  Step 3. instantiate a top-level simulation instance and call the following APIs in turn
 *          1) SimTop::init(); //Pre-Simulation Initialization
 *          2) SimTop::run();  //Simulation main loop
 *          3) SimTop::finish(); // Post-Simulation cleanup
 * --------------------------------------------------------------------------------------*/

#include "ACALSim.hh"
using namespace acalsim;

// Step 1 include header files of the simulator classes
#include "CacheSim.hh"
#include "NocSim.hh"
#include "TrafficGenerator.hh"

// Step 2. Inherit SimTop to create your own top-level simulation class
class TestSimChannel : public SimTop {
public:
	TestSimChannel() : SimTop() {}

	void registerSimulators() override {
		// Create simulators
		SimBase* trafficGenerator = (SimBase*)new TrafficGenerator("Traffic Generator");
		SimBase* nocSim           = (SimBase*)new NocSim("Noc Simulator");
		SimBase* cacheSim         = (SimBase*)new CacheSim("Cache Simulator");

		// register Simulators
		this->addSimulator(trafficGenerator);
		this->addSimulator(nocSim);
		this->addSimulator(cacheSim);

		std::string tg2noc_port    = "TG2NOC";
		std::string noc2cache_port = "NOC2Cache";
		std::string cache2noc_port = "Cache2NOC";
		std::string noc2tg_port    = "NOC2TG";

		// Connect components with SimChannel
		ChannelPortManager::ConnectPort(trafficGenerator, nocSim, tg2noc_port + "-m", tg2noc_port + "-s");
		ChannelPortManager::ConnectPort(nocSim, cacheSim, noc2cache_port + "-m", noc2cache_port + "-s");
		ChannelPortManager::ConnectPort(cacheSim, nocSim, cache2noc_port + "-m", cache2noc_port + "-s");
		ChannelPortManager::ConnectPort(nocSim, trafficGenerator, noc2tg_port + "-m", noc2tg_port + "-s");

		// connect simulators
		trafficGenerator->addDownStream(nocSim, "DSNOC");
		nocSim->addDownStream(cacheSim, "DSCache");
		cacheSim->addUpStream(nocSim, "USNOC");
		nocSim->addUpStream(trafficGenerator, "USTrafficGenerator");
	}
};

int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	top = std::make_shared<TestSimChannel>();
	top->init(argc, argv);
	top->run();
	top->finish();
	return 0;
}
