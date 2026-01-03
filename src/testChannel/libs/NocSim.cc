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
 * @brief Network-on-Chip (NOC) router simulator with bidirectional channel communication
 *
 * This file implements the NocSim simulator, which acts as a packet routing intermediary
 * between TrafficGenerator and CacheSim. It demonstrates bidirectional channel communication,
 * packet forwarding with callback preservation, outstanding request tracking, and multi-tick
 * network latency modeling.
 *
 * **NocSim Role in System:**
 * ```
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │                            NocSim (Router)                               │
 * │                                                                          │
 * │  Request Path (TrafficGenerator → Cache):                               │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Receive NocReqPacket from TrafficGenerator                      │ │
 * │  │    via SlaveChannelPort "USTrafficGenerator"                       │ │
 * │  │                                                                    │ │
 * │  │ 2. handleTGRequest() invoked by visitor pattern:                  │ │
 * │  │    ├─ Extract packet fields (addr, size, callback)                │ │
 * │  │    ├─ Store in reqQueue with transaction ID                       │ │
 * │  │    ├─ Unwrap: NocReqPacket → CacheReqPacket                       │ │
 * │  │    └─ Create CacheReqEvent with nocReqCallback                    │ │
 * │  │                                                                    │ │
 * │  │ 3. Forward to Cache:                                              │ │
 * │  │    ├─ Wrap in EventPacket (tick + nocRespDelay = 1)               │ │
 * │  │    └─ Push to MasterChannelPort "DSCache"                         │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                          │
 * │  Response Path (Cache → TrafficGenerator):                              │
 * │  ┌────────────────────────────────────────────────────────────────────┐ │
 * │  │ 4. nocReqCallback() invoked by Cache:                             │ │
 * │  │    ├─ Receive NocRespPacket from Cache                            │ │
 * │  │    ├─ Retrieve original NocReqPacket from reqQueue                │ │
 * │  │    ├─ Extract caller's callback (TrafficEvent::NocRespHandler)    │ │
 * │  │    └─ Create NocRespEvent with TG's callback                      │ │
 * │  │                                                                    │ │
 * │  │ 5. Forward to TrafficGenerator:                                   │ │
 * │  │    ├─ Wrap in EventPacket (tick + nocRespDelay = 1)               │ │
 * │  │    └─ Push to MasterChannelPort "USTrafficGenerator"              │ │
 * │  └────────────────────────────────────────────────────────────────────┘ │
 * │                                                                          │
 * │  Channel Ports (Bidirectional):                                         │
 * │    ┌─────────────────────────────────────────────────────────────────┐ │
 * │    │ SlaveChannelPort "USTrafficGenerator" ← TG requests             │ │
 * │    │ MasterChannelPort "USTrafficGenerator" → TG responses           │ │
 * │    │ MasterChannelPort "DSCache" → Cache requests                    │ │
 * │    │ SlaveChannelPort "DSCache" ← Cache responses                    │ │
 * │    └─────────────────────────────────────────────────────────────────┘ │
 * │                                                                          │
 * │  Outstanding Request Queue:                                             │
 * │    ┌───────────────────────────────────────────────────────────────┐   │
 * │    │ reqQueue: UnorderedRequestQueue<SimPacket*>                   │   │
 * │    │   ├─ Key: Transaction ID (int)                                │   │
 * │    │   ├─ Value: NocReqPacket* (with original callback)            │   │
 * │    │   └─ Purpose: Match responses to requests                     │   │
 * │    └───────────────────────────────────────────────────────────────┘   │
 * └──────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Packet Routing Flow:**
 * ```
 * Request Path:
 * ════════════
 *
 * Tick 11: NocReqEvent.process()
 *   ├─ NocReqPacket::visit(NocSim&) called
 *   └─ Dispatches to handleTGRequest()
 *
 * Tick 11: handleTGRequest(NocReqPacket* nocReqPkt, int id, Tick when)
 *   ├─ INPUT: nocReqPkt (addr=0, size=256, callback=λ)
 *   ├─ Store: reqQueue->add(id, nocReqPkt)  // Save for response matching
 *   ├─ Unwrap: Create CacheReqPacket(TEST, addr=0, size=256)
 *   ├─ Callback: callback = [this, id](...) { nocReqCallback(when, id, ...) }
 *   ├─ Create: CacheReqEvent(id, "Noc2Cache", callback, cache, cacheReqPkt)
 *   ├─ Wrap: EventPacket(cacheReqEvent, when + nocRespDelay)
 *   ├─ Chrome trace: "CacheReqEvent" duration 1 tick
 *   └─ Send: *getMasterChannelPort("DSCache") << eventPkt
 *
 * Tick 12 (Phase 2):
 *   └─ CacheSim.USNOC receives EventPacket
 *
 * Response Path:
 * ═════════════
 *
 * Tick 12: CacheSim processes request and invokes nocReqCallback()
 *
 * Tick 12: nocReqCallback(Tick when, int id, NocRespPacket* nocRespPkt, SimBase* sim)
 *   ├─ INPUT: nocRespPkt (data=100), id=transaction_id, sim=CacheSim
 *   ├─ Retrieve: reqPkt = (NocReqPacket*)reqQueue->get(id)
 *   ├─ Extract: callerCallback = reqPkt->getCallback()
 *   │           // This is TrafficEvent::NocRespHandler
 *   ├─ Create: NocRespEvent(id, "noc2tg", callerCallback, nocRespPkt, this)
 *   ├─ Wrap: EventPacket(nocRespEvent, when + nocRespDelay)
 *   ├─ Chrome trace: "NocRespEvent" duration 1 tick
 *   └─ Send: *sim->getMasterChannelPort("USNOC") << eventPkt
 *            // Note: sim points to NocSim, not CacheSim
 *
 * Tick 13 (Phase 2):
 *   └─ TrafficGenerator.DSNOC receives EventPacket
 *
 * Tick 13: TrafficEvent::NocRespHandler() invoked
 *   └─ Transaction completes
 * ```
 *
 * **Callback Preservation Pattern:**
 * ```
 * The NOC acts as a transparent router that preserves callbacks across layers:
 *
 * Step 1: TrafficGenerator creates callback
 *   auto tgCallback = [this](Tick, int, NocRespPacket*, SimBase*) {
 *       this->NocRespHandler(...);
 *   };
 *
 * Step 2: TrafficGenerator embeds callback in NocReqPacket
 *   nocReqPkt->setCallback(tgCallback);  // Done by NocReqEvent
 *
 * Step 3: NOC stores NocReqPacket in reqQueue
 *   reqQueue->add(id, nocReqPkt);  // Preserves original callback
 *
 * Step 4: NOC creates new callback for Cache
 *   auto nocCallback = [this, id](...) {
 *       this->nocReqCallback(when, id, ...);  // NOC's own callback
 *   };
 *
 * Step 5: Cache invokes NOC's callback
 *   nocCallback(when, id, nocRespPkt, this);  // Calls nocReqCallback()
 *
 * Step 6: NOC retrieves original callback and forwards response
 *   auto tgCallback = reqPkt->getCallback();  // Get TG's callback
 *   tgCallback(when, id, nocRespPkt, this);   // Invoke TG's callback
 *
 * This pattern enables callback chaining: TG → NOC → Cache → NOC → TG
 * ```
 *
 * **Outstanding Request Queue:**
 * ```cpp
 * // Request storage structure
 * class NocSim {
 * private:
 *     UnorderedRequestQueue<SimPacket*>* reqQueue;
 * };
 *
 * // Store request when forwarding to Cache
 * reqQueue->add(transactionID, nocReqPkt);
 *
 * // Retrieve request when response arrives
 * NocReqPacket* reqPkt = (NocReqPacket*)reqQueue->get(transactionID);
 *
 * // Extract original callback
 * auto callerCallback = reqPkt->getCallback();
 *
 * // Benefits:
 * //   - Decouples request and response handling
 * //   - Preserves transaction context across ticks
 * //   - Enables asynchronous response matching
 * //   - Supports multiple outstanding requests
 * ```
 *
 * **Bidirectional Channel Port Usage:**
 * ```
 * Direction 1: TrafficGenerator → NOC → Cache (Requests)
 * ═══════════════════════════════════════════════════════
 *
 * TrafficGenerator:
 *   ├─ Push to MasterChannelPort "DSNOC"
 *   └─ Connected to: NocSim.USTrafficGenerator (Slave)
 *
 * NocSim:
 *   ├─ Receive from SlaveChannelPort "USTrafficGenerator"
 *   ├─ Process and forward
 *   ├─ Push to MasterChannelPort "DSCache"
 *   └─ Connected to: CacheSim.USNOC (Slave)
 *
 * Direction 2: Cache → NOC → TrafficGenerator (Responses)
 * ════════════════════════════════════════════════════════
 *
 * CacheSim:
 *   ├─ Invoke callback (nocReqCallback) directly
 *   └─ No explicit port push (callback-based)
 *
 * NocSim:
 *   ├─ nocReqCallback() creates response event
 *   ├─ Push to MasterChannelPort "USTrafficGenerator"
 *   └─ Connected to: TrafficGenerator.DSNOC (Slave)
 *
 * Note: Each direction uses a separate channel connection
 *       Total of 4 ConnectPort() calls for bidirectional communication
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Bidirectional Channel Communication:**
 *    - Four channel ports (2 Master, 2 Slave)
 *    - Separate paths for requests and responses
 *    - Lock-free asynchronous transfer
 *
 * 2. **Packet Unwrapping/Wrapping:**
 *    - Unwrap: NocReqPacket → CacheReqPacket
 *    - Wrap: CacheRespPkt → NocRespPacket (implicitly)
 *    - Layer-specific packet types
 *
 * 3. **Callback Chain Management:**
 *    - Store original callback in reqQueue
 *    - Create intermediate callback for Cache
 *    - Forward response to original caller
 *
 * 4. **Network Latency Modeling:**
 *    - Request latency: nocRespDelay (1 tick)
 *    - Response latency: nocRespDelay (1 tick)
 *    - Total NOC contribution: 2 ticks
 *
 * 5. **Outstanding Request Tracking:**
 *    - UnorderedRequestQueue for fast lookup
 *    - Transaction ID for request-response matching
 *    - Supports multiple in-flight requests
 *
 * **Configuration Parameters:**
 * ```cpp
 * class NocSim : public CPPSimBase {
 * private:
 *     static const int nocRespDelay = 1;     // Routing latency (ticks)
 *     static const int nocBandwidth = 32;    // Bandwidth (bytes/tick)
 * };
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * // In TestChannel::registerSimulators()
 * SimBase* nocSim = new NocSim("Noxim");
 * this->addSimulator(nocSim);
 *
 * // Connect channels (4 connections for bidirectional communication)
 * ChannelPortManager::ConnectPort(trafficGenerator, nocSim, "DSNOC", "USTrafficGenerator");
 * ChannelPortManager::ConnectPort(nocSim, cacheSim, "DSCache", "USNOC");
 * ChannelPortManager::ConnectPort(cacheSim, nocSim, "USNOC", "DSCache");
 * ChannelPortManager::ConnectPort(nocSim, trafficGenerator, "USTrafficGenerator", "DSNOC");
 *
 * // Add relationships
 * trafficGenerator->addDownStream(nocSim, "DSNOC");
 * nocSim->addDownStream(cacheSim, "DSCache");
 * cacheSim->addUpStream(nocSim, "USNOC");
 * nocSim->addUpStream(trafficGenerator, "USTrafficGenerator");
 * ```
 *
 * **Expected Output:**
 * ```
 * [NocSim] Process NocReqEvent with transaction id: 1 at Tick=11
 * [NocSim] Thread 0x7000012af000 executes NocReqEvent::nocReqCallback() transaction id: 1 at Tick=12
 * [NocSim] Process NocRespEvent with transaction id: 1 at Tick=13
 * ```
 *
 * **Performance Characteristics:**
 * - Request latency: 1 tick (nocRespDelay)
 * - Response latency: 1 tick (nocRespDelay)
 * - Total NOC latency: 2 ticks per round-trip
 * - No blocking (asynchronous callbacks)
 * - Supports pipelining (multiple outstanding requests)
 *
 * **Extending NOC Features:**
 * 1. Add mesh/torus topology routing algorithms
 * 2. Implement virtual channels for deadlock avoidance
 * 3. Add credit-based flow control
 * 4. Support multicast/broadcast packets
 * 5. Implement priority-based arbitration
 * 6. Add congestion-aware adaptive routing
 * 7. Model router pipeline stages (SA, VA, ST)
 * 8. Implement wormhole/virtual cut-through switching
 *
 * @see NocReqEvent For request event processing
 * @see NocRespEvent For response event processing
 * @see NocReqPacket For request packet structure
 * @see NocRespPacket For response packet structure
 * @see CacheReqEvent For downstream cache events
 * @see TrafficEvent For upstream traffic generator events
 * @see UnorderedRequestQueue For outstanding request tracking
 */

#include "NocSim.hh"

#include "NocEvent.hh"
#include "container/ChromeTraceRecord.hh"

/**
 * @brief Initialize NocSim (currently no-op for passive router)
 *
 * The NOC simulator operates as a passive router that only reacts to incoming
 * requests. No initialization events are scheduled because the NOC waits for
 * TrafficGenerator to send requests.
 *
 * **Design Choice:**
 * Unlike TrafficGenerator which actively generates traffic, NocSim is reactive:
 * - No self-scheduled events
 * - All processing triggered by incoming packets
 * - Request queue initialized in constructor
 *
 * @note Commented code shows how to schedule NOC-initiated events if needed
 * @see handleTGRequest() For request processing entry point
 * @see nocReqCallback() For response forwarding entry point
 */
void NocSim::init() {
	// TODO: Should schedule the events into event queue.
	// for (Tick i = 1; i < 10; ++i) {
	// 	// Schedule the event for testing.
	// 	NocEvent* noc_event = new NocEvent(0, std::to_string(i));
	// 	scheduleEvent(noc_event, i * 2 + 1);
	// }
}

/**
 * @brief Clean up NocSim resources at simulation end
 *
 * This method is called once after the simulation loop completes.
 *
 * **Automatic Cleanup:**
 * - reqQueue automatically deleted by destructor
 * - Channel ports managed by ChannelPortManager
 * - Events auto-delete after process() completes
 *
 * @note Currently no manual cleanup needed (placeholder for future extensions)
 * @see SimBase::cleanup() For base class cleanup behavior
 */
void NocSim::cleanup() {
	// TODO: Release the dynamic memory, clean up the event queue, ...etc.

	// clean up the event queue
}

/**
 * @brief Handle incoming request from TrafficGenerator and forward to Cache
 *
 * This method is invoked by the visitor pattern when NocReqPacket arrives from
 * TrafficGenerator. It demonstrates the request forwarding path with callback
 * preservation and packet unwrapping.
 *
 * **Processing Steps:**
 * 1. Retrieve downstream Cache simulator reference
 * 2. Store request in reqQueue for response matching
 * 3. Create lambda callback to capture transaction context
 * 4. Unwrap NOC packet → Create Cache packet
 * 5. Create CacheReqEvent with NOC's callback
 * 6. Wrap in EventPacket with target tick
 * 7. Push to DSCache MasterChannelPort
 *
 * **Callback Lambda:**
 * ```cpp
 * auto callback = [this, id](Tick when, int _id, NocRespPacket* pkt, SimBase* sim) {
 *     this->nocReqCallback(when, id, pkt, sim);
 * };
 * ```
 * Captures: this (NocSim*), id (transaction ID)
 * Purpose: Forward response back to TrafficGenerator when Cache responds
 *
 * **Packet Unwrapping:**
 * ```
 * NocReqPacket (addr=0, size=256, callback=TrafficEvent::NocRespHandler)
 *       ↓ Unwrap
 * CacheReqPacket (TEST, addr=0, size=256)
 * ```
 *
 * **Request Queue Storage:**
 * ```
 * reqQueue->add(nocReqPkt->getID(), nocReqPkt);
 * ```
 * Purpose: Preserve original callback for response forwarding
 *
 * @param nocReqPkt Request packet from TrafficGenerator
 * @param id Transaction ID (simulator ID, not packet ID)
 * @param when Current tick when request arrived
 *
 * @note Uses nocReqPkt->getID() for reqQueue key (packet's transaction ID)
 * @note Uses 'id' parameter for CacheReqEvent (simulator ID)
 * @note Chrome trace shows "CacheReqEvent" with duration = nocRespDelay
 *
 * @see NocReqPacket::visit() For visitor pattern invocation
 * @see CacheReqEvent For downstream event type
 * @see nocReqCallback() For response handling
 */
void NocSim::handleTGRequest(NocReqPacket* nocReqPkt, int id, Tick when) {
	SimBase* cache = this->getDownStream("DSCache");
	reqQueue->add(nocReqPkt->getID(), (SimPacket*)nocReqPkt);
	auto callback = [this, id](Tick when, int _id, NocRespPacket* pkt, SimBase* sim) {
		this->nocReqCallback(when, id, pkt, sim);
	};
	top->addChromeTraceRecord(
	    acalsim::ChromeTraceRecord::createCompleteEvent("NocSim", "CacheReqEvent", when, this->getRespDelay()));
	CacheReqPacket* cacheReqPkt =
	    new CacheReqPacket(CachePktTypeEnum::TEST, nocReqPkt->getAddr(), nocReqPkt->getSize());
	CacheReqEvent* cacheReqEvent = new CacheReqEvent(id, "Noc2Cache", callback, cache, cacheReqPkt);
	EventPacket*   eventPkt      = new EventPacket(cacheReqEvent, when + this->getRespDelay());
	*(this->getMasterChannelPort("DSCache")) << eventPkt;
}

/**
 * @brief Callback invoked by Cache to forward response back to TrafficGenerator
 *
 * This method is invoked by CacheSim after processing a request. It retrieves the
 * original request from reqQueue, extracts the TrafficGenerator's callback, and
 * forwards the response upstream.
 *
 * **Processing Steps:**
 * 1. Retrieve original NocReqPacket from reqQueue using transaction ID
 * 2. Extract caller's callback (TrafficEvent::NocRespHandler)
 * 3. Validate callback exists
 * 4. Create NocRespEvent with caller's callback
 * 5. Wrap in EventPacket with target tick
 * 6. Push to USTrafficGenerator MasterChannelPort
 *
 * **Callback Chain:**
 * ```
 * CacheSim invokes: nocReqCallback(when, id, nocRespPkt, NocSim)
 *       ↓
 * NocSim retrieves: callerCallback = TrafficEvent::NocRespHandler
 *       ↓
 * NocSim forwards: NocRespEvent wraps callerCallback
 *       ↓
 * TrafficGenerator receives: NocRespEvent.process() → callerCallback()
 * ```
 *
 * **Request Queue Lookup:**
 * ```cpp
 * NocReqPacket* reqPkt = (NocReqPacket*)reqQueue->get(_id);
 * if (reqPkt) {
 *     auto callerCallback = reqPkt->getCallback();  // Get TG's callback
 *     // Forward response...
 * }
 * ```
 *
 * **Error Handling:**
 * If reqPkt not found in queue:
 * - Logs: "Packet not found !"
 * - Response dropped (no forwarding)
 * - Indicates bug (request never stored or already removed)
 *
 * **Port Selection:**
 * ```cpp
 * *(sim->getMasterChannelPort("USNOC")) << eventPkt;
 * ```
 * Note: 'sim' parameter points to NocSim (this), not CacheSim
 *       Uses NocSim's "USNOC" port to reach TrafficGenerator
 *
 * @param when Tick when response should be delivered
 * @param _id Transaction ID (from downstream module)
 * @param _nocRespPkt Response packet from Cache (wrapped in NocRespPacket)
 * @param sim Pointer to simulator (should be 'this' NocSim)
 *
 * @note Chrome trace shows "NocRespEvent" with duration = nocRespDelay
 * @note Logs thread ID for debugging parallel execution
 *
 * @see handleTGRequest() For request path that stores in reqQueue
 * @see NocRespEvent For response event type
 * @see TrafficEvent::NocRespHandler() For ultimate callback destination
 */
void NocSim::nocReqCallback(Tick when, int _id, /* transaction ID to the downstream module */
                            NocRespPacket* _nocRespPkt, SimBase* sim) {
	NocReqPacket* reqPkt = ((NocReqPacket*)reqQueue->get(_id));
	if (reqPkt) {
		auto callerCallback = reqPkt->getCallback();
		if (callerCallback) {  // callerCallback = TrafficEvent::NocRespHandler
			CLASS_INFO << "Thread " << std::this_thread::get_id()
			           << "executes NocReqEvent::nocReqCallback()  transaction id: " << _id
			           << " at Tick=" << top->getGlobalTick();
			top->addChromeTraceRecord(
			    acalsim::ChromeTraceRecord::createCompleteEvent("NocSim", "NocRespEvent", when, this->getRespDelay()));
			NocRespEvent* nocRespEvent = new NocRespEvent(_id, "noc2tg", callerCallback, _nocRespPkt, this);
			EventPacket*  eventPkt     = new EventPacket(nocRespEvent, when + this->getRespDelay());
			*(sim->getMasterChannelPort("USNOC")) << eventPkt;
		}
	} else {
		CLASS_INFO << "Packet not found !";
	}
}
