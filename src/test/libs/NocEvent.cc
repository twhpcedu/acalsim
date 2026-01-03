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
 * @file NocEvent.cc
 * @brief NOC event implementation with callback support for network-on-chip simulation
 *
 * This file implements the **NocEvent** class, which represents events processed by the
 * network-on-chip (NOC) simulator. It demonstrates **callback-based event notification**,
 * **dual-purpose event handling** (self-scheduled vs. routed), and serves as the intermediate
 * event type in the TrafficGenerator → NOC → Cache event flow.
 *
 * **Event Role and Architecture:**
 * ```
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                           NocEvent                                       │
 * │                      (SimEvent derived)                                  │
 * │                                                                           │
 * │  Purpose:                                                                 │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Represent NOC operations and packet routing                     │ │
 * │  │ 2. Support callback mechanism for event completion notification    │ │
 * │  │ 3. Handle both self-scheduled and externally-created events        │ │
 * │  │ 4. Provide routing ID for network topology                         │ │
 * │  │ 5. Serve as intermediate event in multi-simulator flows            │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Event Types:                                                             │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ Type 1: Self-Scheduled (created in NocSim::init())                 │ │
 * │  │   - No callback function                                           │ │
 * │  │   - Simulates internal NOC operations                              │ │
 * │  │   - Independent of traffic flow                                    │ │
 * │  │                                                                     │ │
 * │  │ Type 2: Routed Traffic (created in TrafficEvent::process())       │ │
 * │  │   - Has callback function (lambda from TrafficEvent)               │ │
 * │  │   - Represents traffic flowing through NOC                         │ │
 * │  │   - Notifies upstream on completion                                │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                           │
 * │  Key Members:                                                             │
 * │  - id: Routing/PE identifier (0 in simple example)                       │
 * │  - _name: Event name (e.g., "NocEvent_1" or "TestEventFromPE2NOC")      │
 * │  - callback: Optional completion callback (std::function<void(void)>)    │
 * └─────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Event Processing Flow:**
 * ```
 * Type 1: Self-Scheduled NocEvent (from NocSim::init()):
 *
 * Tick 3 (example):
 *   Step 1: Framework invokes process()
 *     └─► NocEvent::process() called
 *
 *   Step 2: Log processing
 *     └─► CLASS_INFO << "NocEvent Processed."
 *     └─► Output: [Noxim] NocEvent Processed.
 *
 *   Step 3: Check and execute callback
 *     └─► if (callback) { callback(); }
 *     └─► No callback for self-scheduled events
 *     └─► Processing complete
 *
 * Type 2: Routed NocEvent (from TrafficEvent::process()):
 *
 * Tick 13 (example - 10 ticks after TrafficEvent_1):
 *   Step 1: Framework invokes process()
 *     └─► NocEvent::process() called
 *     └─► This is the NocEvent created by TrafficEvent_1
 *
 *   Step 2: Log processing
 *     └─► CLASS_INFO << "NocEvent Processed."
 *     └─► Output: [Noxim] NocEvent Processed.
 *
 *   Step 3: Execute callback (notifies TrafficEvent)
 *     └─► if (callback) { callback(); }
 *     └─► Callback exists (lambda from TrafficEvent)
 *     └─► Invokes: [this] { callback(); } from TrafficEvent
 *     └─► Executes: TrafficEvent::callback()
 *         ├─► Log: "TrafficEvent Callback()"
 *         └─► Release TrafficEvent memory via releaseImpl()
 *
 * Flow Diagram:
 *   Self-Scheduled:
 *     NocSim::init()
 *       └─► Create NocEvent(id, name)    [no callback]
 *           └─► scheduleEvent(event, tick)
 *               └─► process()
 *                   └─► Log only
 *
 *   Routed Traffic:
 *     TrafficEvent::process()
 *       └─► Create NocEvent(id, name, callback)
 *           └─► Schedule in NOC @ tick+10
 *               └─► process()
 *                   ├─► Log "NocEvent Processed"
 *                   └─► callback()  [invokes TrafficEvent::callback()]
 * ```
 *
 * **process() Method Implementation:**
 * ```cpp
 * void NocEvent::process() {
 *     // Step 1: Log event processing
 *     CLASS_INFO << "NocEvent Processed.";
 *     // Logs: [Noxim] NocEvent Processed.
 *     // [Noxim] is the simulator name from NocSim construction
 *
 *     // Step 2: Execute callback if provided
 *     if (callback) {
 *         callback();
 *         // Invokes std::function<void(void)> callback
 *         // For routed events: calls TrafficEvent::callback()
 *         // For self-scheduled: callback is nullptr, skips this
 *     }
 * }
 *
 * // Simple but powerful:
 * // - Logs all NOC event processing
 * // - Supports optional callback mechanism
 * // - Works for both self-scheduled and routed events
 * // - Enables cross-simulator event notification
 * ```
 *
 * **Callback Mechanism Details:**
 * ```
 * Callback Storage:
 *   std::function<void(void)> callback;
 *   - Type: std::function wrapper for any callable
 *   - Supports: lambdas, function pointers, functors
 *   - Optional: can be nullptr
 *   - Flexibility: captures context from caller
 *
 * Callback Creation (in TrafficEvent::process()):
 *   NocEvent* nocEvent = new NocEvent(
 *       0,
 *       "TestEventFromPE2NOC",
 *       [this] { callback(); }  // Lambda with capture
 *   );
 *
 *   Lambda breakdown:
 *     [this]              - Capture TrafficEvent instance
 *     { callback(); }     - Call TrafficEvent::callback()
 *     Stored as:          std::function<void(void)>
 *
 * Callback Execution:
 *   When NocEvent::process() runs:
 *     if (callback)       - Check if callback exists
 *         callback();     - Invoke the stored function
 *
 *   This calls the lambda:
 *     [this] { callback(); }
 *       └─► Resolves to: TrafficEvent::callback()
 *           └─► Logs "TrafficEvent Callback()"
 *           └─► Calls this->releaseImpl()
 *
 * Why std::function?
 *   - Type erasure: caller doesn't need to know TrafficEvent type
 *   - Flexibility: supports any callable with matching signature
 *   - Standard: part of C++11 standard library
 *   - Efficient: small object optimization for lambdas
 * ```
 *
 * **Constructor Variations:**
 * ```cpp
 * // From NocEvent.hh:
 * NocEvent(int _id, std::string name,
 *          std::function<void(void)> _callback = nullptr)
 *     : SimEvent(),
 *       id(_id),
 *       _name("NocEvent_" + name),
 *       callback(_callback)
 * {}
 *
 * Usage Pattern 1: Self-Scheduled (no callback)
 *   NocEvent* event = new NocEvent(0, "1");
 *   // callback defaults to nullptr
 *   // _name becomes "NocEvent_1"
 *
 * Usage Pattern 2: Routed with Callback
 *   NocEvent* event = new NocEvent(
 *       0,
 *       "TestEventFromPE2NOC",
 *       [this] { callback(); }  // Completion notification
 *   );
 *   // _name becomes "NocEvent_TestEventFromPE2NOC"
 *   // callback stores lambda for later invocation
 *
 * Usage Pattern 3: With Routing ID
 *   NocEvent* event = new NocEvent(
 *       15,                     // PE/Router ID
 *       "RouteToCache",
 *       [this] { onComplete(); }
 *   );
 *   // id = 15 (could represent destination router)
 *   // Enables routing decisions based on ID
 * ```
 *
 * **Event Name Construction:**
 * ```
 * Name Format: "NocEvent_" + supplied_name
 *
 * Examples:
 *   Input: "1"                  → Output: "NocEvent_1"
 *   Input: "TestEventFromPE2NOC" → Output: "NocEvent_TestEventFromPE2NOC"
 *   Input: "RouteToCache"       → Output: "NocEvent_RouteToCache"
 *
 * Naming Conventions:
 *   Self-scheduled:   Simple numbers ("1", "2", "3", ...)
 *   Routed traffic:   Descriptive names ("TestEventFromPE2NOC")
 *   Future routing:   "PE5_to_PE12", "Read_0x1000", etc.
 *
 * Name Usage:
 *   - Debugging and trace analysis
 *   - Event filtering in logs
 *   - Performance profiling
 *   - Visualization of event flows
 * ```
 *
 * **Routing ID Usage (Future Extension):**
 * ```cpp
 * // Current: id is placeholder (0)
 * // Future: id represents routing information
 *
 * // Example: Source/Destination Encoding
 * class NocEvent : public SimEvent {
 * private:
 *     int id;  // Encodes src and dst
 *
 *     int getSourcePE() const {
 *         return (id >> 16) & 0xFFFF;  // Upper 16 bits
 *     }
 *
 *     int getDestPE() const {
 *         return id & 0xFFFF;  // Lower 16 bits
 *     }
 *
 * public:
 *     void process() override {
 *         CLASS_INFO << "Routing packet from PE"
 *                    << getSourcePE() << " to PE" << getDestPE();
 *
 *         // Determine routing path
 *         std::vector<int> route = calculateRoute(
 *             getSourcePE(),
 *             getDestPE()
 *         );
 *
 *         // Process through NOC topology
 *         for (int hop : route) {
 *             processHop(hop);
 *         }
 *
 *         if (callback) {
 *             callback();  // Notify sender of completion
 *         }
 *     }
 * };
 *
 * // Usage:
 * int src = 5, dst = 12;
 * int routeId = (src << 16) | dst;  // Encode as 0x0005000C
 * NocEvent* event = new NocEvent(routeId, "RoutePacket", callback);
 * ```
 *
 * **Extending for Realistic NOC Simulation:**
 * ```cpp
 * class NocEvent : public SimEvent {
 * private:
 *     // Routing information
 *     int sourceId;
 *     int destId;
 *     std::vector<int> route;  // Hops through NOC
 *     int currentHop;
 *
 *     // Packet information
 *     uint64_t address;        // Memory address
 *     uint32_t size;           // Packet size in flits
 *     PacketType type;         // READ_REQ, WRITE_REQ, ACK, etc.
 *     uint8_t* payload;        // Data payload
 *
 *     // Timing
 *     Tick injectionTime;      // When packet entered NOC
 *     Tick headLatency;        // Head flit latency
 *     Tick bodyLatency;        // Body flit latency
 *
 *     // Statistics
 *     int hopCount;
 *     bool contentionEncountered;
 *
 * public:
 *     void process() override {
 *         CLASS_INFO << "Processing " << packetTypeString(type)
 *                    << " from PE" << sourceId
 *                    << " to PE" << destId;
 *
 *         // Calculate routing path if not already done
 *         if (route.empty()) {
 *             route = calculateXYRoute(sourceId, destId);
 *         }
 *
 *         // Process current hop
 *         if (currentHop < route.size()) {
 *             int nextRouter = route[currentHop];
 *             Tick hopLatency = calculateHopLatency(nextRouter);
 *
 *             // Check for contention
 *             if (isLinkBusy(nextRouter)) {
 *                 contentionEncountered = true;
 *                 hopLatency += contentionDelay;
 *             }
 *
 *             currentHop++;
 *             hopCount++;
 *
 *             // Schedule next hop or completion
 *             if (currentHop < route.size()) {
 *                 scheduleEvent(this, top->getGlobalTick() + hopLatency);
 *             } else {
 *                 // Reached destination
 *                 recordStatistics();
 *                 if (callback) {
 *                     callback();
 *                 }
 *             }
 *         }
 *     }
 *
 *     void recordStatistics() {
 *         Tick totalLatency = top->getGlobalTick() - injectionTime;
 *         nocStats.recordPacket(hopCount, totalLatency, contentionEncountered);
 *     }
 * };
 * ```
 *
 * **Callback Patterns:**
 * ```cpp
 * // Pattern 1: Simple Notification
 * NocEvent* event = new NocEvent(id, name,
 *     [this] {
 *         CLASS_INFO << "NOC operation completed";
 *         this->releaseImpl();
 *     }
 * );
 *
 * // Pattern 2: Data Passing
 * NocEvent* event = new NocEvent(id, name,
 *     [this, responseData] {
 *         processResponse(responseData);
 *         this->releaseImpl();
 *     }
 * );
 *
 * // Pattern 3: Performance Tracking
 * Tick startTime = top->getGlobalTick();
 * NocEvent* event = new NocEvent(id, name,
 *     [this, startTime] {
 *         Tick latency = top->getGlobalTick() - startTime;
 *         recordLatency(latency);
 *         this->releaseImpl();
 *     }
 * );
 *
 * // Pattern 4: Chain of Events
 * NocEvent* event = new NocEvent(id, name,
 *     [this, nextSim] {
 *         // Create next event in chain
 *         CacheEvent* cacheEvent = new CacheEvent("data");
 *         nextSim->scheduleEvent(cacheEvent,
 *                               top->getGlobalTick() + 1);
 *         this->releaseImpl();
 *     }
 * );
 * ```
 *
 * **Comparison: With Callback vs. Without:**
 *
 * | Aspect               | Without Callback      | With Callback        |
 * |----------------------|-----------------------|----------------------|
 * | Event Source         | Self-scheduled        | External creator     |
 * | Completion Notice    | None                  | Invokes callback     |
 * | Use Case             | Independent ops       | Request-response     |
 * | Memory Management    | Auto or manual        | Manual (in callback) |
 * | Cross-Sim Comm       | No                    | Yes                  |
 * | Example              | NOC internal          | Traffic routing      |
 *
 * **Integration with Event Flow:**
 * ```
 * Complete Multi-Simulator Flow:
 *
 * TrafficGenerator          NocSim                CacheSim
 *       │                      │                       │
 *   init() called              │                       │
 *       │                      │                       │
 *   Schedule                   │                       │
 *   TrafficEvent @tick3        │                       │
 *       │                      │                       │
 *   ┌───▼───┐                  │                       │
 *   │ Tick 3│                  │                       │
 *   └───┬───┘                  │                       │
 *       │                      │                       │
 *   TrafficEvent::process()    │                       │
 *       ├─► Create NocEvent    │                       │
 *       │   with callback      │                       │
 *       └─► Schedule @tick13 ──┼─► [Event Queue]      │
 *                              │                       │
 *                          ┌───▼───┐                   │
 *                          │Tick 13│                   │
 *                          └───┬───┘                   │
 *                              │                       │
 *                   NocEvent::process()                │
 *                              ├─► Log "Processed"     │
 *                              └─► callback() ────────►│
 *                                         │            │
 *     ◄─────────────────────────────────┘            │
 *     │                                                │
 *  callback()                                          │
 *     ├─► Log "Callback()"                            │
 *     └─► releaseImpl()                                │
 * ```
 *
 * **Design Rationale:**
 * - Simple process() keeps NOC event handling clean
 * - Optional callback enables flexible notification
 * - std::function provides type-safe callback storage
 * - Supports both independent and routed event types
 * - Minimal overhead for self-scheduled events
 *
 * @see test.cc Main test framework and topology
 * @see NocEvent.hh NocEvent class definition
 * @see NocSim.cc NOC simulator creating self-scheduled events
 * @see TrafficEvent.cc Traffic events creating routed NocEvents with callbacks
 * @see TrafficEvent.hh TrafficEvent with callback() method
 * @see SimEvent Base class for all events
 */

#include "NocEvent.hh"

void NocEvent::process() {
	CLASS_INFO << "NocEvent Processed.";
	if (callback) { callback(); }
}
