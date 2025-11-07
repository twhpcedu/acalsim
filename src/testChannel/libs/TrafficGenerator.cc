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
 * @brief Event-driven traffic generation for channel-based NOC communication
 *
 * This file implements the TrafficGenerator simulator, which initiates request transactions
 * in the channel-based communication example. It demonstrates event-driven request generation,
 * asynchronous channel communication using MasterChannelPort/SlaveChannelPort, callback-based
 * response handling, and Chrome trace integration for visualization.
 *
 * **TrafficGenerator Role in System:**
 * ```
 * ┌────────────────────────────────────────────────────────────────┐
 * │                      TrafficGenerator                          │
 * │                                                                │
 * │  Initialization Phase:                                         │
 * │  ┌─────────────────────────────────────────────────────────┐  │
 * │  │ init()                                                   │  │
 * │  │   1. Schedule TrafficEvent(id=1) at tick 3               │  │
 * │  │   2. Add Chrome trace record for visualization           │  │
 * │  └─────────────────────────────────────────────────────────┘  │
 * │                                                                │
 * │  Request Generation (via TrafficEvent):                        │
 * │  ┌─────────────────────────────────────────────────────────┐  │
 * │  │ TrafficEvent::process()                                  │  │
 * │  │   3. Create NocReqPacket (addr=0, size=256)              │  │
 * │  │   4. Create NocReqEvent with callback lambda             │  │
 * │  │   5. Wrap in EventPacket (target tick = current + 10)    │  │
 * │  │   6. Push to MasterChannelPort "DSNOC"                   │  │
 * │  │      → Sends to NocSim.USTrafficGenerator                │  │
 * │  └─────────────────────────────────────────────────────────┘  │
 * │                                                                │
 * │  Response Handling (via callback):                             │
 * │  ┌─────────────────────────────────────────────────────────┐  │
 * │  │ TrafficEvent::NocRespHandler()                           │  │
 * │  │   7. Receive NocRespPacket from NOC                      │  │
 * │  │   8. Create TrafficRespEvent with response data          │  │
 * │  │   9. Wrap in EventPacket (target tick = current + 1)     │  │
 * │  │   10. Push to MasterChannelPort "USTrafficGenerator"     │  │
 * │  │       → Self-delivery for final processing               │  │
 * │  └─────────────────────────────────────────────────────────┘  │
 * │                                                                │
 * │  Final Processing:                                             │
 * │  ┌─────────────────────────────────────────────────────────┐  │
 * │  │ TrafficRespEvent::process()                              │  │
 * │  │   11. Log transaction completion                         │  │
 * │  │   12. Add Chrome trace instant event                     │  │
 * │  │   13. Print response data (e.g., "Data = 100")           │  │
 * │  └─────────────────────────────────────────────────────────┘  │
 * │                                                                │
 * │  Channel Ports:                                                │
 * │    MasterChannelPort "DSNOC" → Sends requests to NOC          │
 * │    SlaveChannelPort "USTrafficGenerator" → Receives responses │
 * └────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Traffic Generation Pattern:**
 * ```
 * Timeline (ticks):
 *
 * Tick 0: init() called
 *   ├─ Schedule TrafficEvent(id=1) at tick 3
 *   └─ Add Chrome trace record
 *
 * Tick 3: TrafficEvent::process()
 *   ├─ Create NocReqPacket (TEST type, addr=0, size=256)
 *   ├─ Create callback lambda: [this](Tick, int, NocRespPacket*, SimBase*) { NocRespHandler(...) }
 *   ├─ Create NocReqEvent with callback
 *   ├─ Wrap in EventPacket (target tick = 3 + 10 = 13)
 *   ├─ Push to DSNOC MasterChannelPort
 *   └─ Chrome trace: "NocReqEvent" duration 10 ticks
 *
 * Tick 4 (Phase 2):
 *   └─ ChannelPortManager transfers EventPacket to NocSim.USTrafficGenerator
 *
 * Tick 13: NocSim processes request (see NocSim.cc for details)
 *
 * Tick 14: Cache processes request (see CacheSim.cc)
 *
 * Tick 15: NocSim sends response back
 *   └─ TrafficEvent::NocRespHandler() invoked via callback
 *
 * Tick 15: NocRespHandler()
 *   ├─ Receive NocRespPacket with data=100
 *   ├─ Create TrafficRespEvent
 *   ├─ Wrap in EventPacket (target tick = 15 + 1 = 16)
 *   ├─ Push to USTrafficGenerator MasterChannelPort (self-delivery)
 *   └─ Chrome trace: "NocSim2TrafficGenerator" duration 1 tick
 *
 * Tick 16: TrafficRespEvent::process()
 *   ├─ Log: "Transaction Finish ! Data = 100"
 *   └─ Chrome trace instant event: "Transaction Finish"
 * ```
 *
 * **MasterChannelPort vs SlaveChannelPort Usage:**
 * ```cpp
 * // MasterChannelPort - For sending packets asynchronously
 * MasterChannelPort* sendPort = getMasterChannelPort("DSNOC");
 * EventPacket* eventPkt = new EventPacket(event, targetTick);
 * *sendPort << eventPkt;  // Operator<< pushes to channel
 * // Alternative: pushToMasterChannelPort("DSNOC", eventPkt);
 *
 * // SlaveChannelPort - For receiving packets (handled automatically by framework)
 * // Framework pops from SlaveChannelPort and schedules events
 * // No manual pop required - events are auto-delivered at target tick
 * ```
 *
 * **Callback Chaining Pattern:**
 * ```
 * Step 1: TrafficGenerator creates callback
 *   auto callback = [this](Tick when, int id, NocRespPacket* pkt, SimBase* sim) {
 *       this->NocRespHandler(when, id, pkt, sim);
 *   };
 *
 * Step 2: Callback embedded in NocReqEvent
 *   NocReqEvent* nocReqEvent = new NocReqEvent(id, "tg2Noc", callback, noc, nocReqPkt);
 *
 * Step 3: NocSim extracts callback from NocReqPacket
 *   nocReqPkt->setCallback(callerCallback);  // Store for later use
 *
 * Step 4: Cache invokes callback after processing
 *   callerCallback(when, id, nocRespPkt, this);  // Calls NocSim::nocReqCallback()
 *
 * Step 5: NocSim forwards to original callback
 *   auto callerCallback = reqPkt->getCallback();  // Get TrafficGenerator's callback
 *   callerCallback(when, id, nocRespPkt, this);   // Calls TrafficEvent::NocRespHandler()
 *
 * Step 6: TrafficGenerator completes transaction
 *   NocRespHandler() creates final TrafficRespEvent
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Event-Driven Request Generation:**
 *    - init() schedules TrafficEvent at specific ticks
 *    - TrafficEvent::process() creates and sends requests
 *    - Avoids polling - purely event-driven
 *
 * 2. **Asynchronous Channel Communication:**
 *    - MasterChannelPort "DSNOC" for sending to NOC
 *    - SlaveChannelPort "USTrafficGenerator" for receiving from NOC
 *    - EventPacket wrapper contains target tick and event
 *    - Lock-free dual-queue transfer (no blocking)
 *
 * 3. **Callback-Based Response Handling:**
 *    - Lambda captures transaction context (id, this pointer)
 *    - Callback chained through NOC and Cache layers
 *    - Asynchronous notification when response ready
 *    - No manual polling or request tracking needed
 *
 * 4. **Chrome Trace Integration:**
 *    - Complete events show duration (e.g., NocReqEvent)
 *    - Instant events mark milestones (e.g., Transaction Finish)
 *    - Enables visual debugging in chrome://tracing
 *
 * 5. **Multi-Tick Latency Modeling:**
 *    - Request generation: 10-tick latency (see TrafficEvent.cc)
 *    - Response processing: 1-tick latency
 *    - Total contribution: 11 ticks per transaction
 *
 * **Initialization Phase:**
 * ```cpp
 * void TrafficGenerator::init() {
 *     // Schedule N traffic events with staggered timing
 *     for (Tick i = 1; i < 2; ++i) {
 *         Tick eventTick = i * 2 + 1;  // Tick 3, 5, 7, ...
 *
 *         // Add Chrome trace for visualization
 *         top->addChromeTraceRecord(
 *             ChromeTraceRecord::createCompleteEvent(
 *                 "TrafficGenerator", "TrafficEvent",
 *                 top->getGlobalTick(), eventTick
 *             )
 *         );
 *
 *         // Create and schedule traffic event
 *         TrafficEvent* traffic_event = new TrafficEvent(
 *             (SimBase*)this, i, std::to_string(i)
 *         );
 *         scheduleEvent(traffic_event, eventTick);
 *     }
 * }
 * ```
 *
 * **Cleanup Phase:**
 * ```cpp
 * void TrafficGenerator::cleanup() {
 *     // Event queue automatically cleaned by framework
 *     // No manual memory management needed for events
 *     // (Events auto-delete after process() completes)
 * }
 * ```
 *
 * **Configuration Parameters:**
 * ```cpp
 * class TrafficGenerator : public CPPSimBase {
 * protected:
 *     static const int tgRespDelay = 1;  // Response processing latency
 * };
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * // In TestChannel::registerSimulators()
 * SimBase* trafficGenerator = new TrafficGenerator("Traffic Generator");
 * this->addSimulator(trafficGenerator);
 *
 * // Connect channels (bidirectional)
 * ChannelPortManager::ConnectPort(
 *     trafficGenerator, nocSim,
 *     "DSNOC", "USTrafficGenerator"  // TG → NOC (requests)
 * );
 * ChannelPortManager::ConnectPort(
 *     nocSim, trafficGenerator,
 *     "USTrafficGenerator", "DSNOC"  // NOC → TG (responses)
 * );
 *
 * // Add downstream/upstream relationships
 * trafficGenerator->addDownStream(nocSim, "DSNOC");
 * nocSim->addUpStream(trafficGenerator, "USTrafficGenerator");
 * ```
 *
 * **Expected Output:**
 * ```
 * [TrafficGenerator] TrafficEvent Processed.
 * [TrafficGenerator] Traffic Event Processed and sendNocEvent with transaction id: 1 at Tick=3
 * [TrafficGenerator] Thread 0x7000012af000 executes NocRespHandler with transaction id: 1 at Tick=15
 * [TrafficGenerator] Transaction Finish ! Data = 100
 * ```
 *
 * **Performance Characteristics:**
 * - Request generation rate: 1 request per 2 ticks (configurable)
 * - Request latency contribution: 10 ticks
 * - Response latency contribution: 1 tick
 * - No blocking operations (fully asynchronous)
 * - Lock-free channel communication
 *
 * **Extending Traffic Patterns:**
 * 1. Add burst traffic generation (multiple requests per event)
 * 2. Implement Poisson arrival process for realistic workloads
 * 3. Support read/write/prefetch request types
 * 4. Add address generation patterns (sequential, random, stride)
 * 5. Implement outstanding request limits for flow control
 * 6. Add trace-driven traffic replay from files
 *
 * @see TrafficEvent For request generation event implementation
 * @see TrafficRespEvent For response processing event
 * @see NocReqPacket For request packet structure
 * @see NocRespPacket For response packet structure
 * @see NocSim For NOC routing and forwarding
 * @see CacheSim For cache request processing
 */

#include "TrafficGenerator.hh"

#include "TrafficEvent.hh"
#include "container/ChromeTraceRecord.hh"

/**
 * @brief Initialize TrafficGenerator by scheduling traffic generation events
 *
 * This method is called once during simulator initialization (before the main
 * simulation loop). It schedules TrafficEvent instances at specific ticks to
 * generate request traffic to the NOC.
 *
 * **Scheduling Pattern:**
 * - Event ID 1: Scheduled at tick 3 (1 * 2 + 1)
 * - Event ID 2: Scheduled at tick 5 (2 * 2 + 1) [if enabled]
 * - Event ID N: Scheduled at tick (N * 2 + 1)
 *
 * **Chrome Trace Integration:**
 * Each scheduled event gets a corresponding Chrome trace record for visualization
 * in chrome://tracing. The trace shows the complete event duration.
 *
 * @note Currently only 1 event is scheduled (loop: i < 2)
 * @note Increase loop limit to generate more traffic
 *
 * @see TrafficEvent::process() For request generation logic
 * @see ChromeTraceRecord::createCompleteEvent() For trace visualization
 */
void TrafficGenerator::init() {
	// TODO: Should schedule the events into event queue.
	for (Tick i = 1; i < 2; ++i) {
		top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createCompleteEvent("TrafficGenerator", "TrafficEvent",
		                                                                          top->getGlobalTick(), i * 2 + 1));
		// Schedule the event for testing.
		TrafficEvent* traffic_event = new TrafficEvent((SimBase*)this, i, std::to_string(i));
		scheduleEvent(traffic_event, i * 2 + 1);
	}
}

/**
 * @brief Clean up TrafficGenerator resources at simulation end
 *
 * This method is called once after the simulation loop completes. It performs
 * final cleanup of any dynamically allocated resources.
 *
 * **Automatic Cleanup:**
 * - Event queue automatically cleared by framework
 * - Events auto-delete after process() completes
 * - Channel ports managed by ChannelPortManager
 * - No manual memory management required
 *
 * @note Currently no manual cleanup needed (placeholder for future extensions)
 * @see SimBase::cleanup() For base class cleanup behavior
 */
void TrafficGenerator::cleanup() {
	// TODO: Release the dynamic memory, clean up the event queue, ...etc.

	// clean up the event queue
}
