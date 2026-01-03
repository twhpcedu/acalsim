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
 * @file test.cc
 * @brief Basic event-driven simulation example - TrafficGenerator-NOC-Cache architecture
 *
 * This example demonstrates a **basic event-driven simulation framework** in ACALSim through
 * a three-tier architecture connecting traffic generators, network-on-chip (NOC) simulators,
 * and cache simulators. It showcases the fundamental simulation lifecycle, event scheduling,
 * simulator registration, and bidirectional communication paths using downstream/upstream
 * connections without channel-based communication.
 *
 * **System Architecture:**
 * ```
 * ┌───────────────────────────┐       ┌───────────────────────────┐       ┌───────────────────────────┐
 * │    TrafficGenerator       │       │         NocSim            │       │        CacheSim           │
 * │                           │       │    (Network-on-Chip)      │       │                           │
 * │  Event-Driven             │       │                           │       │                           │
 * │  Request Generator        │       │  Event Forwarding         │       │  Event Processing         │
 * │                           │       │  & Routing                │       │                           │
 * │  - Generates traffic      │       │                           │       │                           │
 * │    events at init         │       │  - Routes events          │       │  - Processes cache        │
 * │  - Schedules 9 events     │       │  - Intermediate layer     │       │    events                 │
 * │    (tick 3,5,7,...19)     │       │  - Schedules 9 events     │       │  - Schedules 9 events     │
 * │                           │       │    (tick 3,5,7,...19)     │       │    (tick 3,5,7,...19)     │
 * │  Downstream: DSNOC        │       │                           │       │                           │
 * │  Upstream: USTrafficGen   │       │  DS: DSCache              │       │  Upstream: USNOC          │
 * │                           │       │  US: USTrafficGenerator   │       │                           │
 * └───────────────────────────┘       └───────────────────────────┘       └───────────────────────────┘
 *             ↓                                   ↓                                   ↑
 *        (DSNOC)                             (DSCache)                           (USNOC)
 *             ↓                                   ↓                                   ↑
 *             └──────────► NocSim ───────────────► CacheSim ─────────────────────────┘
 *                    (USTrafficGenerator)                            (upstream path)
 *
 * Connection Topology:
 *   TrafficGenerator ─DSNOC────────────► NocSim (USTrafficGenerator)
 *   NocSim ───────────DSCache──────────► CacheSim (USNOC)
 *   CacheSim ─────────USNOC────────────► NocSim (DSCache)
 *   NocSim ───────USTrafficGenerator───► TrafficGenerator (DSNOC)
 * ```
 *
 * **Simulation Flow (Event-Driven Execution):**
 * ```
 * Initialization Phase (SimTop::init):
 *   TestTop:
 *     1. Create three simulator instances:
 *        - TrafficGenerator("Traffic Generator")
 *        - NocSim("Noxim")
 *        - CacheSim("Cache Simulator")
 *     2. Register simulators with SimTop::addSimulator()
 *     3. Establish bidirectional connections:
 *        - TrafficGenerator.downstream["DSNOC"] = NocSim
 *        - NocSim.downstream["DSCache"] = CacheSim
 *        - CacheSim.upstream["USNOC"] = NocSim
 *        - NocSim.upstream["USTrafficGenerator"] = TrafficGenerator
 *
 *   Framework Initialization:
 *     4. Call each simulator's init() method
 *     5. Each simulator schedules its initial events:
 *        - TrafficGenerator: 9 TrafficEvents at ticks 3,5,7,9,11,13,15,17,19
 *        - NocSim: 9 NocEvents at ticks 3,5,7,9,11,13,15,17,19
 *        - CacheSim: 9 CacheEvents at ticks 3,5,7,9,11,13,15,17,19
 *
 * Simulation Phase (SimTop::run):
 *   Tick 1-2: No events scheduled
 *
 *   Tick 3:
 *     6. Process TrafficEvent (id=1):
 *        - Log: "TrafficEvent Processed"
 *        - Create NocEvent with callback
 *        - Get downstream NOC via getDownStream("DSNOC")
 *        - Schedule NocEvent at tick 13 (current + 10)
 *     7. Process NocEvent (id=1):
 *        - Log: "NocEvent Processed"
 *        - Execute callback if present
 *     8. Process CacheEvent (id=1):
 *        - Log: "CacheEvent Processed"
 *
 *   Tick 5:
 *     9. Process TrafficEvent (id=2) - schedule NocEvent at tick 15
 *     10. Process NocEvent (id=2)
 *     11. Process CacheEvent (id=2)
 *
 *   Tick 7-19:
 *     12. Continue processing events in temporal order
 *
 *   Tick 13:
 *     13. Process NocEvent created by TrafficEvent (id=1)
 *     14. Execute callback: TrafficEvent::callback()
 *     15. Log: "TrafficEvent Callback()"
 *     16. Release TrafficEvent via releaseImpl()
 *
 *   ... (similar pattern for remaining events)
 *
 * Cleanup Phase (SimTop::finish):
 *   17. Call each simulator's cleanup() method
 *   18. Release remaining resources
 *   19. Clear event queues
 * ```
 *
 * **Event Scheduling Pattern:**
 * ```
 * Each simulator independently schedules events during initialization:
 *
 * for (Tick i = 1; i < 10; ++i) {
 *     Event* event = new Event(...);
 *     scheduleEvent(event, i * 2 + 1);  // Ticks: 3, 5, 7, 9, 11, 13, 15, 17, 19
 * }
 *
 * Timeline visualization:
 * Tick:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
 * TG:       *  X  *  X  *  X  *  X  *  X  *  X  *  X  *  X  *  X  *
 * NOC:      *  X  *  X  *  X  *  X  *  X  *  X  *  X  *  X  *  X  *
 * Cache:    *  X  *  X  *  X  *  X  *  X  *  X  *  X  *  X  *  X  *
 *
 * Legend: * = event processed, X = no event
 *
 * Note: TrafficEvent at tick 3 creates additional NocEvent at tick 13
 * ```
 *
 * **Inter-Simulator Communication:**
 * ```
 * Request Path (Downstream):
 *   TrafficGenerator::process()
 *     └─► getDownStream("DSNOC") returns NocSim pointer
 *         └─► NocSim->scheduleEvent(nocEvent, tick+10)
 *             └─► NocEvent scheduled in NocSim's event queue
 *
 * The downstream/upstream connections enable:
 * 1. Dynamic event routing between simulators
 * 2. Flexible topology reconfiguration
 * 3. Simulator decoupling (each simulator only knows connection names)
 * 4. Support for hierarchical simulation architectures
 * ```
 *
 * **Event Lifecycle Management:**
 * ```
 * Event Creation:
 *   TrafficEvent* event = new TrafficEvent(this, id, name);
 *   - Constructor clears Managed flag: clearFlags(Managed)
 *   - Manual memory management required
 *
 * Event Scheduling:
 *   scheduleEvent(event, targetTick);
 *   - Framework inserts event into priority queue
 *   - Events processed in temporal order
 *
 * Event Processing:
 *   event->process();
 *   - Virtual method dispatched based on event type
 *   - Can schedule new events (cross-simulator)
 *
 * Event Cleanup:
 *   this->releaseImpl();
 *   - Manual release in callback
 *   - Prevents memory leaks for unmanaged events
 * ```
 *
 * **Key Design Patterns:**
 *
 * 1. **Simulator Registration Pattern:**
 *    ```cpp
 *    SimBase* sim = new ConcreteSimulator("name");
 *    this->addSimulator(sim);
 *    ```
 *    Registers simulator with framework for lifecycle management
 *
 * 2. **Connection Establishment Pattern:**
 *    ```cpp
 *    srcSim->addDownStream(dstSim, "connectionName");
 *    dstSim->addUpStream(srcSim, "connectionName");
 *    ```
 *    Creates bidirectional communication paths
 *
 * 3. **Cross-Simulator Event Routing:**
 *    ```cpp
 *    SimBase* target = sim->getDownStream("DSNOC");
 *    target->scheduleEvent(event, futureTicke);
 *    ```
 *    Routes events to connected simulators
 *
 * 4. **Callback-Based Event Completion:**
 *    ```cpp
 *    NocEvent* event = new NocEvent(id, name, [this] { callback(); });
 *    ```
 *    Lambda captures enable deferred event completion
 *
 * **Comparison with Channel-Based Communication:**
 *
 * This example uses direct event scheduling:
 * - Simple event passing via scheduleEvent()
 * - No packet wrapping (EventPacket)
 * - No channel ports (MasterChannelPort/SlaveChannelPort)
 * - No ping-pong buffer mechanism
 * - Immediate event delivery (Phase 2 not involved)
 *
 * For channel-based communication with buffering:
 * @see testChannel.cc Channel-based asynchronous communication
 * @see testCommunication.cc Advanced packet-based communication
 *
 * **Usage Example:**
 * ```bash
 * # Build the test
 * cd build
 * cmake ..
 * make test
 *
 * # Run the simulation
 * ./test
 *
 * # Expected output pattern:
 * [TrafficGenerator] TrafficEvent Processed.
 * [Noxim] NocEvent Processed.
 * [Cache Simulator] CacheEvent Processed.
 * ... (repeating pattern)
 * [TrafficGenerator] TrafficEvent Callback()
 * ```
 *
 * **Extending This Example:**
 *
 * 1. Add realistic traffic patterns:
 *    - Poisson arrival processes
 *    - Bursty traffic generation
 *    - Trace-driven traffic
 *
 * 2. Implement NOC routing logic:
 *    - XY routing algorithm
 *    - Adaptive routing
 *    - Deadlock avoidance
 *
 * 3. Add cache functionality:
 *    - Cache hit/miss simulation
 *    - Replacement policies (LRU, LFU)
 *    - Coherence protocols (MESI, MOESI)
 *
 * 4. Implement performance metrics:
 *    - Latency histograms
 *    - Throughput measurements
 *    - Queue occupancy tracking
 *
 * 5. Upgrade to channel-based communication:
 *    - Add ChannelPorts for buffered communication
 *    - Implement packet structures
 *    - Use EventPacket for timestamped delivery
 *
 * **Performance Considerations:**
 * - 27 total events scheduled (9 per simulator)
 * - Additional events created dynamically (TrafficEvent → NocEvent)
 * - Event queue uses priority queue (O(log n) insertion/removal)
 * - String-based connection lookup (consider optimization for large-scale)
 *
 * **Framework Concepts Demonstrated:**
 * - Event-driven simulation fundamentals
 * - Simulator lifecycle (init/run/finish)
 * - Dynamic event scheduling
 * - Inter-simulator communication
 * - Bidirectional connection graphs
 * - Callback-based event completion
 *
 * @see TrafficGenerator.cc Traffic generation implementation
 * @see NocSim.cc Network-on-chip simulation implementation
 * @see CacheSim.cc Cache simulation implementation
 * @see TrafficEvent.cc Traffic event processing and NOC event creation
 * @see NocEvent.cc NOC event processing with callbacks
 * @see CacheEvent.cc Cache event processing
 * @see SimBase Base class for all simulators
 * @see SimTop Top-level simulation orchestrator
 * @see SimEvent Base class for all events
 */

#include "ACALSim.hh"
using namespace acalsim;

// Step 1 include header files of the simulator classes
#include "CacheSim.hh"
#include "NocSim.hh"
#include "TrafficGenerator.hh"

// Step 2. Inherit SimTop to create your own top-level simulation class
class TestTop : public SimTop {
public:
	TestTop() : SimTop() {}

	void registerSimulators() override {
		// Create simulators
		SimBase* trafficGenerator = (SimBase*)new TrafficGenerator("Traffic Generator");
		SimBase* nocSim           = (SimBase*)new NocSim("Noxim");
		SimBase* cacheSim         = (SimBase*)new CacheSim("Cache Simulator");

		// register Simulators
		this->addSimulator(trafficGenerator);
		this->addSimulator(nocSim);
		this->addSimulator(cacheSim);

		// connect simulators
		trafficGenerator->addDownStream(nocSim, "DSNOC");
		nocSim->addDownStream(cacheSim, "DSCache");
		cacheSim->addUpStream(nocSim, "USNOC");
		nocSim->addUpStream(trafficGenerator, "USTrafficGenerator");
	}
};

int main(int argc, char** argv) {
	// Step 3. instantiate a top-level simulation instance
	top = std::make_shared<TestTop>();
	top->init(argc, argv);
	top->run();
	top->finish();

	return 0;
}
