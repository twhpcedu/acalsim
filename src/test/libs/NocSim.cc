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
 * @brief Network-on-Chip (NOC) simulator implementation for event routing and forwarding
 *
 * This file implements the **NocSim** simulator component, which serves as the intermediate
 * routing layer in the test architecture. It demonstrates network-on-chip behavior including
 * event forwarding, bidirectional communication between traffic generators and cache simulators,
 * and multi-port connectivity patterns.
 *
 * **Role in System Architecture:**
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                             NocSim                                       │
 * │                  (Network-on-Chip Simulator)                             │
 * │                      (CPPSimBase derived)                                │
 * │                                                                           │
 * │  Responsibilities:                                                        │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Route events between TrafficGenerator and CacheSim             │ │
 * │  │ 2. Manage multiple connection ports (upstream/downstream)          │ │
 * │  │ 3. Schedule independent NOC events for testing                     │ │
 * │  │ 4. Forward events across simulator boundaries                      │ │
 * │  │ 5. Provide intermediate layer for event transformation             │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Connection Topology:                                                     │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │                           ▲                                         │ │
 * │  │                           │ Upstream                                │ │
 * │  │               ┌───────────┴───────────┐                             │ │
 * │  │               │                       │                             │ │
 * │  │     "USTrafficGenerator"       "DSCache"                            │ │
 * │  │          (from TG)              (to Cache)                          │ │
 * │  │               │                       │                             │ │
 * │  │               │                       │                             │ │
 * │  │         ┌─────▼─────┐           ┌────▼─────┐                       │ │
 * │  │         │   Upstream │           │ Downstream│                      │ │
 * │  │         │    Port    │           │   Port   │                       │ │
 * │  │         │            │           │          │                       │ │
 * │  │         │ Receives   │           │  Sends   │                       │ │
 * │  │         │ requests   │           │ requests │                       │ │
 * │  │         │ from TG    │           │ to Cache │                       │ │
 * │  │         └────────────┘           └──────────┘                       │ │
 * │  │                                                                     │ │
 * │  │  Independent Event Queue:                                           │ │
 * │  │  [NocEvent_1@tick3] [NocEvent_2@tick5] ... [NocEvent_9@tick19]    │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Routing Behavior:                                                        │
 * │  - Receives TrafficEvents (via TrafficEvent::process())                  │
 * │  - Forwards to CacheSim via downstream connection                        │
 * │  - Maintains own event queue for NOC-specific operations                 │
 * │  - Provides latency modeling (event scheduling delays)                   │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Communication Flow:
 *
 * Request Path:
 *   TrafficGenerator ──► NocSim (USTrafficGenerator port)
 *                        └─► Process/Route
 *                            └─► CacheSim (via DSCache port)
 *
 * Response Path (if implemented):
 *   CacheSim ──► NocSim (DSCache port)
 *               └─► Process/Route
 *                   └─► TrafficGenerator (via USTrafficGenerator port)
 * ```
 *
 * **NOC Simulation Flow:**
 * ```
 * Initialization (init() called by framework):
 *   Step 1: Loop i = 1 to 9
 *     ├─► Create NocEvent(0, std::to_string(i))
 *     │   - ID = 0 (placeholder for routing ID)
 *     │   - Name suffix: "1", "2", "3", ..., "9"
 *     │   - Full event names: "NocEvent_1", "NocEvent_2", etc.
 *     │
 *     ├─► Calculate target tick: i * 2 + 1
 *     │   - Same scheduling pattern as TrafficGenerator
 *     │   - Ticks: 3, 5, 7, 9, 11, 13, 15, 17, 19
 *     │   - Simulates independent NOC operations
 *     │
 *     └─► scheduleEvent(noc_event, targetTick)
 *         - Framework inserts event into NOC's event queue
 *         - These events run independently of routed traffic
 *
 * Simulation Phase (dual event processing):
 *   Type 1: Independent NOC Events (scheduled in init)
 *     Tick 3:
 *       └─► NocEvent_1::process() invoked
 *           - Logs: "NocEvent Processed."
 *           - Executes callback if provided
 *           - Represents NOC internal operations
 *
 *   Type 2: Routed Traffic Events (from TrafficGenerator)
 *     Tick 13 (example):
 *       └─► NocEvent created by TrafficEvent_1::process()
 *           - Different from independent NOC events
 *           - Contains callback to TrafficEvent::callback()
 *           - Logs: "NocEvent Processed."
 *           - Executes callback() to notify TrafficGenerator
 *
 * Cleanup Phase (cleanup() called by framework):
 *   - Release dynamic memory allocations
 *   - Clear event queue
 *   - Free routing tables (if implemented)
 * ```
 *
 * **Event Scheduling Timeline:**
 * ```
 * NOC Event Types on Timeline:
 *
 * Tick:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
 *
 * NOC Init Events:
 *              ●     ●     ●     ●     ●     ●     ●     ●     ●
 *            (self-scheduled during init phase)
 *
 * Routed Traffic:
 *                             ●     ●     ●     ●     ●     ●     ●     ●     ●
 *                       (from TrafficGenerator, delayed by 10 ticks)
 *
 * Combined View:
 *              ●     ●     ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
 *              │     │     │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
 *         Init │     │     │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
 *              └─────┴─────┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──
 *                                 │  │  │  │  │  │  │  │  │
 *                        Routed   │  │  │  │  │  │  │  │  │
 *                                 └──┴──┴──┴──┴──┴──┴──┴──┴──
 *
 * Note: At tick 13+, both event types may coexist in the queue
 * ```
 *
 * **NocEvent Creation Patterns:**
 * ```cpp
 * // Pattern 1: Self-scheduled events (in init())
 * NocEvent* noc_event = new NocEvent(
 *     0,                    // Routing ID (0 = broadcast or local)
 *     std::to_string(i)     // Event name suffix
 * );
 * scheduleEvent(noc_event, i * 2 + 1);
 *
 * // Pattern 2: Routed events (from TrafficEvent::process())
 * NocEvent* nocEvent = new NocEvent(
 *     0,                           // Routing ID
 *     "TestEventFromPE2NOC",       // Descriptive name
 *     [this] { callback(); }       // Completion callback
 * );
 * noc->scheduleEvent(nocEvent, top->getGlobalTick() + 10);
 *
 * // Key Differences:
 * // - Self-scheduled: No callback, pure NOC operations
 * // - Routed: Has callback for end-to-end acknowledgment
 * ```
 *
 * **Connection Port Management:**
 * ```
 * Port Configuration (in test.cc):
 *
 * Upstream Connections (receives events from):
 *   1. "USTrafficGenerator" ← TrafficGenerator
 *      - Receives traffic requests from generator
 *      - Entry point for request path
 *
 *   (Future: could add upstream from CacheSim for responses)
 *
 * Downstream Connections (sends events to):
 *   1. "DSCache" → CacheSim
 *      - Forwards requests to cache simulator
 *      - Exit point for request path
 *
 *   (Future: could add downstream to other NOCs for multi-hop routing)
 *
 * Port Lookup Example:
 *   SimBase* cache = this->getDownStream("DSCache");
 *   if (cache) {
 *       CacheEvent* event = new CacheEvent("data");
 *       cache->scheduleEvent(event, currentTick + nocLatency);
 *   }
 * ```
 *
 * **NOC Routing Functionality (Placeholder for Extension):**
 * ```
 * Current Implementation:
 *   - Simple event forwarding (no actual routing)
 *   - Fixed latency model (constant delay)
 *   - No congestion modeling
 *   - No buffer management
 *
 * Production NOC Simulator Would Include:
 *
 * 1. Routing Algorithms:
 *    - XY Routing: Route in X dimension, then Y dimension
 *    - Adaptive Routing: Choose least congested path
 *    - Source Routing: Path determined at source
 *
 * 2. Flow Control:
 *    - Credit-based flow control
 *    - Wormhole switching
 *    - Virtual channel allocation
 *
 * 3. Network Topology:
 *    - 2D/3D Mesh topology
 *    - Torus topology
 *    - Hierarchical topology
 *
 * 4. Performance Modeling:
 *    - Link traversal latency
 *    - Router pipeline stages
 *    - Contention delays
 *    - Serialization latency
 * ```
 *
 * **Extension Example: XY Routing Implementation:**
 * ```cpp
 * class NocSim : public CPPSimBase {
 * private:
 *     struct RouterNode {
 *         int x, y;                    // Router coordinates
 *         std::queue<NocEvent*> inputBuffer[4];  // N, S, E, W ports
 *         int creditCount[4];          // Available credits per port
 *     };
 *
 *     std::vector<std::vector<RouterNode>> topology;
 *     int meshWidth = 4, meshHeight = 4;
 *
 * public:
 *     void routeEvent(NocEvent* event, int srcX, int srcY,
 *                     int dstX, int dstY) {
 *         // XY Routing: Route in X first, then Y
 *         int currentX = srcX, currentY = srcY;
 *         Tick currentTick = top->getGlobalTick();
 *
 *         // X-dimension routing
 *         while (currentX != dstX) {
 *             int direction = (dstX > currentX) ? EAST : WEST;
 *             currentX += (dstX > currentX) ? 1 : -1;
 *             currentTick += hopLatency;
 *         }
 *
 *         // Y-dimension routing
 *         while (currentY != dstY) {
 *             int direction = (dstY > currentY) ? NORTH : SOUTH;
 *             currentY += (dstY > currentY) ? 1 : -1;
 *             currentTick += hopLatency;
 *         }
 *
 *         // Schedule event at destination
 *         scheduleEvent(event, currentTick);
 *     }
 * };
 * ```
 *
 * **Extension Example: Congestion Modeling:**
 * ```cpp
 * class NocSim : public CPPSimBase {
 * private:
 *     struct LinkState {
 *         int occupancy;           // Current buffer occupancy
 *         int maxCapacity;         // Buffer capacity
 *         Tick nextAvailable;      // Next available time
 *     };
 *
 *     std::map<std::string, LinkState> linkStates;
 *
 * public:
 *     Tick calculateLinkLatency(std::string linkId) {
 *         LinkState& link = linkStates[linkId];
 *
 *         // Base latency
 *         Tick latency = baseLinkLatency;
 *
 *         // Contention delay
 *         if (link.occupancy >= link.maxCapacity) {
 *             latency = link.nextAvailable - top->getGlobalTick();
 *         }
 *
 *         // Serialization delay
 *         latency += (packetSize / linkBandwidth);
 *
 *         // Update link state
 *         link.occupancy++;
 *         link.nextAvailable = top->getGlobalTick() + latency;
 *
 *         return latency;
 *     }
 * };
 * ```
 *
 * **Integration with Traffic Generator:**
 * ```
 * Event Flow Example:
 *
 * Tick 3:
 *   TrafficGenerator:
 *     TrafficEvent_1::process()
 *       ├─► Create NocEvent("TestEventFromPE2NOC", callback)
 *       ├─► Get NOC via getDownStream("DSNOC")
 *       └─► noc->scheduleEvent(nocEvent, tick + 10)
 *
 * Tick 13:
 *   NocSim:
 *     NocEvent::process()
 *       ├─► Log: "NocEvent Processed."
 *       └─► Execute callback()
 *           └─► TrafficEvent::callback()
 *               └─► Log: "TrafficEvent Callback()"
 *
 * This demonstrates end-to-end event flow with NOC intermediate layer
 * ```
 *
 * **Performance Metrics (Future Enhancement):**
 * ```cpp
 * class NocSim : public CPPSimBase {
 * private:
 *     struct NocMetrics {
 *         uint64_t totalPackets;       // Total packets routed
 *         uint64_t totalHops;          // Total hop count
 *         uint64_t totalLatency;       // Cumulative latency
 *         uint64_t droppedPackets;     // Packets dropped due to congestion
 *         std::map<int, uint64_t> hopDistribution;  // Histogram
 *     } metrics;
 *
 * public:
 *     void cleanup() override {
 *         double avgHops = (double)metrics.totalHops / metrics.totalPackets;
 *         double avgLatency = (double)metrics.totalLatency / metrics.totalPackets;
 *
 *         std::cout << "NOC Statistics:" << std::endl;
 *         std::cout << "  Total Packets: " << metrics.totalPackets << std::endl;
 *         std::cout << "  Average Hops: " << avgHops << std::endl;
 *         std::cout << "  Average Latency: " << avgLatency << std::endl;
 *         std::cout << "  Dropped Packets: " << metrics.droppedPackets << std::endl;
 *     }
 * };
 * ```
 *
 * **Comparison: Simple Forwarding vs. Realistic NOC:**
 *
 * | Feature                | This Example | Realistic NOC           |
 * |------------------------|--------------|-------------------------|
 * | Routing Algorithm      | None         | XY, Adaptive, Source    |
 * | Topology               | N/A          | 2D/3D Mesh, Torus       |
 * | Flow Control           | None         | Credit-based, Wormhole  |
 * | Buffer Management      | None         | FIFO queues, VC buffers |
 * | Congestion Modeling    | None         | Link utilization        |
 * | Latency Model          | Fixed        | Hop-based, Contention   |
 * | Performance Metrics    | None         | Throughput, Latency     |
 * | Deadlock Handling      | None         | Turn model, VCs         |
 *
 * **Usage in Test Framework:**
 * ```cpp
 * // In test.cc:
 * SimBase* nocSim = (SimBase*)new NocSim("Noxim");
 * this->addSimulator(nocSim);
 *
 * // Connect TrafficGenerator → NocSim
 * trafficGenerator->addDownStream(nocSim, "DSNOC");
 * nocSim->addUpStream(trafficGenerator, "USTrafficGenerator");
 *
 * // Connect NocSim → CacheSim
 * nocSim->addDownStream(cacheSim, "DSCache");
 * cacheSim->addUpStream(nocSim, "USNOC");
 *
 * // Framework calls:
 * // 1. init()    - schedules 9 independent NOC events
 * // 2. run()     - processes both self-scheduled and routed events
 * // 3. cleanup() - releases resources
 * ```
 *
 * **Design Rationale:**
 * - Demonstrates multi-port connectivity (2 upstream, 2 downstream potential)
 * - Shows event forwarding between simulators
 * - Provides intermediate layer for latency injection
 * - Serves as template for realistic NOC implementation
 * - Independent event queue shows NOC internal operations
 *
 * @see test.cc Main test framework and system topology
 * @see NocEvent.cc NOC event processing implementation
 * @see NocEvent.hh NOC event class definition
 * @see TrafficGenerator.cc Upstream traffic source
 * @see CacheSim.cc Downstream cache simulator
 * @see TrafficEvent.cc Event routing from TrafficGenerator
 * @see CPPSimBase Base class for C++ simulators
 * @see SimEvent Base class for all events
 */

#include "NocSim.hh"

#include "NocEvent.hh"

void NocSim::init() {
	// TODO: Should schedule the events into event queue.
	for (Tick i = 1; i < 10; ++i) {
		// Schedule the event for testing.
		NocEvent* noc_event = new NocEvent(0, std::to_string(i));
		scheduleEvent(noc_event, i * 2 + 1);
	}
}

void NocSim::cleanup() {
	// TODO: Release the dynamic memory, clean up the event queue, ...etc.

	// clean up the event queue
}
