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
 * @brief Traffic generator simulator implementation for event-driven memory traffic generation
 *
 * This file implements the **TrafficGenerator** simulator component, which serves as the
 * traffic injection source in the test architecture. It demonstrates event-based traffic
 * generation patterns, periodic event scheduling, and cross-simulator event routing to
 * downstream components (NOC simulator).
 *
 * **Role in System Architecture:**
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                          TrafficGenerator                                │
 * │                       (CPPSimBase derived)                               │
 * │                                                                           │
 * │  Responsibilities:                                                        │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Initialize traffic generation pattern (init phase)              │ │
 * │  │ 2. Schedule TrafficEvents at regular intervals                     │ │
 * │  │ 3. Generate 9 events at ticks: 3, 5, 7, 9, 11, 13, 15, 17, 19     │ │
 * │  │ 4. Each TrafficEvent creates downstream NocEvent                   │ │
 * │  │ 5. Manage event lifecycle and memory cleanup                       │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Connection Points:                                                       │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ Downstream: "DSNOC" ──────────► NocSim                             │ │
 * │  │   - Sends traffic requests to NOC                                  │ │
 * │  │   - TrafficEvent creates NocEvent via downstream connection        │ │
 * │  │                                                                     │ │
 * │  │ Upstream: "USTrafficGenerator" ◄────── NocSim                      │ │
 * │  │   - Receives responses/callbacks from NOC                          │ │
 * │  │   - Handles event completion notifications                         │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Event Queue:                                                             │
 * │  [TrafficEvent_1@tick3] [TrafficEvent_2@tick5] ... [TrafficEvent_9@19]  │
 * └─────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Traffic Generation Flow:**
 * ```
 * Initialization (init() called by framework):
 *   Step 1: Loop i = 1 to 9
 *     ├─► Create TrafficEvent(this, i, std::to_string(i))
 *     │   - Pass simulator pointer (this)
 *     │   - Assign unique ID (i)
 *     │   - Generate event name: "TrafficEvent_1", "TrafficEvent_2", etc.
 *     │
 *     ├─► Calculate target tick: i * 2 + 1
 *     │   - Tick mapping: 1→3, 2→5, 3→7, 4→9, 5→11, 6→13, 7→15, 8→17, 9→19
 *     │   - Creates periodic pattern with 2-tick intervals
 *     │
 *     └─► scheduleEvent(traffic_event, targetTick)
 *         - Framework inserts event into global priority queue
 *         - Event will trigger at specified tick during simulation phase
 *
 * Simulation Phase (events processed by framework):
 *   Tick 3:
 *     └─► TrafficEvent_1::process() invoked
 *         - Logs: "TrafficEvent Processed."
 *         - Creates NocEvent for downstream NOC simulator
 *         - Schedules NocEvent at tick 13 (current + 10)
 *         - See TrafficEvent.cc for detailed processing
 *
 *   Tick 5:
 *     └─► TrafficEvent_2::process() invoked
 *         (similar pattern)
 *
 *   ... (continues for all 9 events)
 *
 * Cleanup Phase (cleanup() called by framework):
 *   - Release dynamic memory allocations
 *   - Clear event queue (framework handles remaining events)
 *   - Free simulator-specific resources
 * ```
 *
 * **Event Scheduling Pattern Details:**
 * ```
 * Loop iteration and tick calculation:
 *
 * i=1:  tick = 1*2 + 1 = 3   → TrafficEvent_1 scheduled at tick 3
 * i=2:  tick = 2*2 + 1 = 5   → TrafficEvent_2 scheduled at tick 5
 * i=3:  tick = 3*2 + 1 = 7   → TrafficEvent_3 scheduled at tick 7
 * i=4:  tick = 4*2 + 1 = 9   → TrafficEvent_4 scheduled at tick 9
 * i=5:  tick = 5*2 + 1 = 11  → TrafficEvent_5 scheduled at tick 11
 * i=6:  tick = 6*2 + 1 = 13  → TrafficEvent_6 scheduled at tick 13
 * i=7:  tick = 7*2 + 1 = 15  → TrafficEvent_7 scheduled at tick 15
 * i=8:  tick = 8*2 + 1 = 17  → TrafficEvent_8 scheduled at tick 17
 * i=9:  tick = 9*2 + 1 = 19  → TrafficEvent_9 scheduled at tick 19
 *
 * Timeline visualization:
 * Tick:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
 * TG:             ●     ●     ●     ●     ●     ●     ●     ●     ●
 *                 │     │     │     │     │     │     │     │     │
 *                 └─┐   └─┐   └─┐   └─┐   └─┐   └─┐   └─┐   └─┐   └─┐
 *                   │     │     │     │     │     │     │     │     │
 *                   ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
 * NOC:                   ●     ●     ●     ●     ●     ●     ●     ●     ●
 *               (tick+10 delay for each event)
 *
 * Legend: ● = TrafficEvent processed, creates NocEvent scheduled 10 ticks later
 * ```
 *
 * **TrafficEvent Creation and Properties:**
 * ```cpp
 * // Constructor signature from TrafficEvent.hh:
 * TrafficEvent(SimBase* _sim, int _id, std::string name)
 *
 * // Creation in init():
 * TrafficEvent* traffic_event = new TrafficEvent(
 *     this,              // Simulator pointer for downstream routing
 *     i,                 // Unique ID for event tracking (1-9)
 *     std::to_string(i)  // Event name suffix: "TrafficEvent_1", etc.
 * );
 *
 * // Key properties:
 * - Unmanaged memory (clearFlags(Managed) in constructor)
 * - Requires manual release via releaseImpl()
 * - Contains callback mechanism for event completion
 * - Can create and schedule cross-simulator events
 * ```
 *
 * **Simulator Lifecycle Integration:**
 * ```
 * Framework Phases:
 *
 * 1. Construction (TestTop::registerSimulators):
 *    TrafficGenerator* tg = new TrafficGenerator("Traffic Generator");
 *    - Object created
 *    - Constructor calls CPPSimBase("Traffic Generator")
 *    - Registered with framework via addSimulator()
 *
 * 2. Connection Establishment (TestTop::registerSimulators):
 *    tg->addDownStream(nocSim, "DSNOC");
 *    tg->addUpStream(nocSim, "USTrafficGenerator");
 *    - Downstream path to NOC established
 *    - Upstream path from NOC established
 *
 * 3. Initialization (SimTop::init):
 *    tg->init();
 *    - THIS FILE: TrafficGenerator::init() called
 *    - Schedules 9 TrafficEvents
 *    - Event queue populated before simulation starts
 *
 * 4. Simulation (SimTop::run):
 *    [Framework processes events chronologically]
 *    - TrafficEvents trigger at scheduled ticks
 *    - Each event creates downstream NocEvent
 *    - Callbacks executed when NOC processing completes
 *
 * 5. Cleanup (SimTop::finish):
 *    tg->cleanup();
 *    - THIS FILE: TrafficGenerator::cleanup() called
 *    - Release remaining resources
 *    - Framework destroys simulator object
 * ```
 *
 * **Cross-Simulator Event Routing Example:**
 * ```cpp
 * // In TrafficEvent::process() (see TrafficEvent.cc):
 *
 * // Step 1: Create event for downstream NOC
 * NocEvent* nocEvent = new NocEvent(0, "TestEventFromPE2NOC",
 *                                   [this] { callback(); });
 *
 * // Step 2: Get downstream NOC simulator via connection name
 * SimBase* noc = sim->getDownStream("DSNOC");
 * CLASS_ASSERT(noc);  // Verify connection exists
 *
 * // Step 3: Schedule event in NOC's event queue
 * noc->scheduleEvent((SimEvent*)nocEvent, top->getGlobalTick() + 10);
 * // Event will execute 10 ticks after current global time
 * ```
 *
 * **Memory Management Strategy:**
 * ```
 * Event Allocation:
 *   - Dynamic allocation: new TrafficEvent(...)
 *   - Unmanaged flag set in TrafficEvent constructor
 *   - Caller responsible for cleanup
 *
 * Event Cleanup:
 *   - Option 1: Framework auto-cleanup (if Managed flag set)
 *   - Option 2: Manual release via event->releaseImpl()
 *   - This example uses manual release in callback
 *
 * Cleanup Phase:
 *   - cleanup() called after all events processed
 *   - Release any remaining dynamic allocations
 *   - Clear simulator-specific data structures
 * ```
 *
 * **Extension Points:**
 *
 * 1. **Realistic Traffic Patterns:**
 *    ```cpp
 *    // Poisson arrival process
 *    double lambda = 0.5;  // Average arrival rate
 *    std::exponential_distribution<> dist(lambda);
 *    Tick nextArrival = currentTick + dist(rng);
 *
 *    // Bursty traffic
 *    if (burstActive) {
 *        scheduleEvent(event, currentTick + 1);  // High frequency
 *    } else {
 *        scheduleEvent(event, currentTick + 100);  // Low frequency
 *    }
 *    ```
 *
 * 2. **Address Pattern Generation:**
 *    ```cpp
 *    // Sequential access
 *    uint64_t addr = baseAddr + (i * cacheLineSize);
 *
 *    // Random access
 *    std::uniform_int_distribution<> dist(0, memorySize - 1);
 *    uint64_t addr = dist(rng);
 *
 *    // Strided access
 *    uint64_t addr = baseAddr + (i * stride);
 *    ```
 *
 * 3. **Trace-Driven Traffic:**
 *    ```cpp
 *    void init() override {
 *        loadTraceFile("memory_trace.txt");
 *        for (auto& access : trace) {
 *            TrafficEvent* event = new TrafficEvent(this, access.id,
 *                                                    access.type);
 *            scheduleEvent(event, access.timestamp);
 *        }
 *    }
 *    ```
 *
 * 4. **Multiple Traffic Streams:**
 *    ```cpp
 *    void init() override {
 *        // Stream 1: Read-intensive
 *        generateReadTraffic(1, 100, 1);
 *
 *        // Stream 2: Write-intensive
 *        generateWriteTraffic(101, 200, 5);
 *
 *        // Stream 3: Mixed access
 *        generateMixedTraffic(201, 300);
 *    }
 *    ```
 *
 * 5. **Performance Monitoring:**
 *    ```cpp
 *    class TrafficGenerator : public CPPSimBase {
 *    private:
 *        uint64_t totalRequests = 0;
 *        uint64_t totalLatency = 0;
 *        std::map<int, Tick> requestStartTime;
 *
 *    public:
 *        void recordRequest(int id, Tick startTime) {
 *            requestStartTime[id] = startTime;
 *            totalRequests++;
 *        }
 *
 *        void recordResponse(int id, Tick endTime) {
 *            Tick latency = endTime - requestStartTime[id];
 *            totalLatency += latency;
 *        }
 *
 *        double getAverageLatency() {
 *            return (double)totalLatency / totalRequests;
 *        }
 *    };
 *    ```
 *
 * **Comparison with Real-World Traffic Generators:**
 *
 * This simplified implementation vs. production traffic generators:
 *
 * | Feature                  | This Example | Production TG         |
 * |--------------------------|--------------|------------------------|
 * | Traffic Pattern          | Periodic     | Poisson/Trace/Custom  |
 * | Address Generation       | None         | Sequential/Random/Hot |
 * | Request Types            | Generic      | Read/Write/RMW        |
 * | Dependency Modeling      | No           | Load-store deps       |
 * | Outstanding Requests     | Unlimited    | Configurable limit    |
 * | Flow Control             | No           | Credit-based/backpres |
 * | Statistics Collection    | No           | Detailed metrics      |
 *
 * **Usage in Test Framework:**
 * ```cpp
 * // In test.cc:
 * SimBase* trafficGenerator = (SimBase*)new TrafficGenerator("Traffic Generator");
 * this->addSimulator(trafficGenerator);
 * trafficGenerator->addDownStream(nocSim, "DSNOC");
 * trafficGenerator->addUpStream(nocSim, "USTrafficGenerator");
 *
 * // Framework automatically calls:
 * // 1. init()    - schedules 9 events
 * // 2. run()     - processes events
 * // 3. cleanup() - releases resources
 * ```
 *
 * **Design Rationale:**
 * - Simple periodic pattern demonstrates basic event scheduling
 * - Fixed number of events (9) keeps output manageable for testing
 * - 2-tick interval provides clear temporal separation
 * - Cross-simulator event creation shows communication patterns
 * - Manual memory management demonstrates lifecycle control
 *
 * @see test.cc Main test framework and system topology
 * @see TrafficEvent.cc Traffic event processing implementation
 * @see TrafficEvent.hh Traffic event class definition
 * @see NocSim.cc Downstream NOC simulator
 * @see NocEvent.cc NOC event processing
 * @see CPPSimBase Base class for C++ simulators
 * @see SimEvent Base class for all events
 */

#include "TrafficGenerator.hh"

#include "TrafficEvent.hh"

void TrafficGenerator::init() {
	// TODO: Should schedule the events into event queue.
	for (Tick i = 1; i < 10; ++i) {
		// Schedule the event for testing.
		TrafficEvent* traffic_event = new TrafficEvent(this, i, std::to_string(i));
		scheduleEvent(traffic_event, i * 2 + 1);
	}
}

void TrafficGenerator::cleanup() {
	// TODO: Release the dynamic memory, clean up the event queue, ...etc.

	// clean up the event queue
}
