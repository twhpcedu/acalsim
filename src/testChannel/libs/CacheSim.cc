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
 * @brief Cache memory simulator with callback-based response generation
 *
 * This file implements the CacheSim simulator, which serves as the endpoint in the
 * channel-based communication chain. It demonstrates synchronous request processing,
 * callback-based response delivery (without channel ports), and size-dependent latency
 * modeling for memory operations.
 *
 * **CacheSim Role in System:**
 * ```
 * ┌────────────────────────────────────────────────────────────────┐
 * │                       CacheSim (Memory)                        │
 * │                                                                │
 * │  Request Reception:                                            │
 * │  ┌──────────────────────────────────────────────────────────┐ │
 * │  │ 1. Receive CacheReqPacket from NocSim                    │ │
 * │  │    via SlaveChannelPort "USNOC"                          │ │
 * │  │                                                          │ │
 * │  │ 2. CacheReqEvent.process() invoked by framework         │ │
 * │  │    ├─ CacheReqPacket::visit(CacheSim&) called           │ │
 * │  │    └─ Dispatches to handleNOCRequest()                  │ │
 * │  └──────────────────────────────────────────────────────────┘ │
 * │                                                                │
 * │  Request Processing (handleNOCRequest):                        │
 * │  ┌──────────────────────────────────────────────────────────┐ │
 * │  │ 3. Extract request parameters:                           │ │
 * │  │    ├─ addr (memory address)                              │ │
 * │  │    ├─ size (transfer size in bytes)                      │ │
 * │  │    └─ callback (NocSim::nocReqCallback)                  │ │
 * │  │                                                          │ │
 * │  │ 4. Generate pseudo data (data = 100)                     │ │
 * │  │                                                          │ │
 * │  │ 5. Create NocRespPacket with data                        │ │
 * │  │                                                          │ │
 * │  │ 6. Calculate response latency:                           │ │
 * │  │    latency = cacheMemoryLatency + size / bandwidth       │ │
 * │  │            = 1 + (256+1) / 32 = 9 ticks                  │ │
 * │  │                                                          │ │
 * │  │ 7. Invoke caller's callback immediately:                 │ │
 * │  │    callerCallback(when + latency, id, nocRespPkt, this)  │ │
 * │  │    (Calls NocSim::nocReqCallback)                        │ │
 * │  └──────────────────────────────────────────────────────────┘ │
 * │                                                                │
 * │  Channel Ports:                                                │
 * │    ┌────────────────────────────────────────────────────────┐ │
 * │    │ SlaveChannelPort "USNOC" ← Receives requests from NOC  │ │
 * │    │ MasterChannelPort "USNOC" → Sends via callback only    │ │
 * │    │   (Not used for responses - callback-based delivery)   │ │
 * │    └────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Request Processing Flow:**
 * ```
 * Tick 12: CacheReqEvent.process()
 *   ├─ CacheReqPacket::visit(CacheSim&) called
 *   └─ Dispatches to handleNOCRequest()
 *
 * Tick 12: handleNOCRequest(CacheReqPacket* cacheReqPkt, int id, Tick when)
 *   ├─ INPUT: cacheReqPkt (TEST, addr=0, size=256, callback=λ)
 *   ├─ Generate: data = 100 (pseudo response data)
 *   ├─ Create: NocRespPacket(TEST, data=100)
 *   ├─ Calculate: latency = getRespDelay(256)
 *   │             = cacheMemoryLatency + (size+1) / cacheMemoryBandwidth
 *   │             = 1 + 257 / 32 = 1 + 8 = 9 ticks
 *   ├─ Extract: callerCallback = cacheReqPkt->getCallback()
 *   │           // This is NocSim::nocReqCallback
 *   ├─ Chrome trace: "Cache2NocSim" duration 9 ticks
 *   └─ Invoke: callerCallback(when + 9, id, nocRespPkt, this)
 *              // Immediately calls NocSim::nocReqCallback()
 *
 * Tick 12 (same tick): NocSim::nocReqCallback() invoked
 *   └─ Response forwarded to TrafficGenerator
 * ```
 *
 * **Callback-Based Response vs Channel-Based:**
 * ```
 * Traditional Channel Approach (NOT used here):
 * ═══════════════════════════════════════════════
 *   CacheSim::handleNOCRequest() {
 *       // Create response packet
 *       NocRespPacket* respPkt = new NocRespPacket(...);
 *
 *       // Create response event
 *       CacheRespEvent* respEvent = new CacheRespEvent(...);
 *
 *       // Wrap and send via channel
 *       EventPacket* eventPkt = new EventPacket(respEvent, when + latency);
 *       *getMasterChannelPort("USNOC") << eventPkt;
 *   }
 *   → Response delivered at future tick via channel
 *
 * Actual Callback Approach (USED here):
 * ══════════════════════════════════════
 *   CacheSim::handleNOCRequest() {
 *       // Create response packet
 *       NocRespPacket* respPkt = new NocRespPacket(...);
 *
 *       // Extract callback from request
 *       auto callback = cacheReqPkt->getCallback();
 *
 *       // Invoke callback immediately (on same tick)
 *       callback(when + latency, id, respPkt, this);
 *   }
 *   → Callback invoked synchronously (same stack frame)
 *   → Latency modeled by 'when + latency' parameter
 *   → NocSim schedules response event at correct tick
 *
 * Benefits of Callback Approach:
 *   - Simpler: No event/packet wrapping needed
 *   - Faster: One less event scheduling overhead
 *   - Flexible: Callback can do anything (not just send events)
 *   - Same result: Latency preserved via 'when' parameter
 * ```
 *
 * **Latency Calculation:**
 * ```cpp
 * Tick getRespDelay(int size) {
 *     return (Tick)(cacheMemoryLatency + (size + 1) / cacheMemoryBandwidth);
 * }
 *
 * Example:
 *   size = 256 bytes
 *   cacheMemoryLatency = 1 tick (base latency)
 *   cacheMemoryBandwidth = 32 bytes/tick
 *
 *   latency = 1 + (256 + 1) / 32
 *           = 1 + 257 / 32
 *           = 1 + 8  (integer division)
 *           = 9 ticks
 *
 * Components:
 *   - Base latency (1 tick): Tag lookup, coherence check, etc.
 *   - Transfer time (8 ticks): Data transfer from cache array
 *   - Bandwidth model: Larger requests take proportionally longer
 * ```
 *
 * **Key Implementation Details:**
 *
 * 1. **Callback-Based Response Delivery:**
 *    - No response events created
 *    - No channel port used for responses
 *    - Callback invoked synchronously
 *    - Latency modeled via 'when' parameter
 *
 * 2. **Visitor Pattern Request Handling:**
 *    - CacheReqPacket::visit(CacheSim&) invokes handleNOCRequest()
 *    - Type-safe dynamic dispatch
 *    - Decouples packet types from simulator logic
 *
 * 3. **Size-Dependent Latency:**
 *    - Base latency + transfer time
 *    - Bandwidth-limited data transfer
 *    - Realistic memory timing model
 *
 * 4. **Passive Simulator:**
 *    - No self-scheduled events
 *    - Only reacts to incoming requests
 *    - Similar to NocSim (reactive architecture)
 *
 * 5. **Pseudo Data Generation:**
 *    - Constant data value (100)
 *    - Real implementation would:
 *      - Access cache array
 *      - Handle cache misses
 *      - Update replacement policy
 *
 * **Configuration Parameters:**
 * ```cpp
 * class CacheSim : public CPPSimBase {
 * private:
 *     static const int cacheMemoryLatency = 1;    // Base latency (ticks)
 *     static const int cacheMemoryBandwidth = 32; // Bandwidth (bytes/tick)
 * };
 * ```
 *
 * **Usage Example:**
 * ```cpp
 * // In TestChannel::registerSimulators()
 * SimBase* cacheSim = new CacheSim("Cache Simulator");
 * this->addSimulator(cacheSim);
 *
 * // Connect channels
 * ChannelPortManager::ConnectPort(nocSim, cacheSim, "DSCache", "USNOC");
 * ChannelPortManager::ConnectPort(cacheSim, nocSim, "USNOC", "DSCache");
 *
 * // Add relationships
 * nocSim->addDownStream(cacheSim, "DSCache");
 * cacheSim->addUpStream(nocSim, "USNOC");
 * ```
 *
 * **Expected Output:**
 * ```
 * [CacheSim] Process CacheReqEvent with transaction id: 0 at Tick=12
 * ```
 *
 * **Performance Characteristics:**
 * - Base latency: 1 tick
 * - Transfer latency: size-dependent (8 ticks for 256 bytes)
 * - Total latency: 9 ticks (for 256-byte request)
 * - No channel overhead (callback-based response)
 * - Synchronous callback invocation
 *
 * **Extending Cache Features:**
 * 1. Implement cache array and tag lookup
 * 2. Add cache hit/miss detection
 * 3. Support write-back/write-through policies
 * 4. Implement replacement algorithms (LRU, FIFO, Random)
 * 5. Add cache coherence protocol (MSI, MESI, MOESI)
 * 6. Model cache bank conflicts
 * 7. Support multiple cache levels (L1/L2/L3)
 * 8. Add prefetching logic
 * 9. Implement MSHR (Miss Status Holding Registers)
 * 10. Model cache line fill buffers
 *
 * @see CacheReqEvent For request event processing
 * @see CacheReqPacket For request packet structure
 * @see NocRespPacket For response packet structure
 * @see NocSim::nocReqCallback() For response callback destination
 */

#include "CacheSim.hh"

#include "CacheEvent.hh"
#include "container/ChromeTraceRecord.hh"

/**
 * @brief Initialize CacheSim (currently no-op for passive memory)
 *
 * The cache simulator operates as a passive memory that only reacts to incoming
 * requests from the NOC. No initialization events are scheduled.
 *
 * **Design Choice:**
 * CacheSim is purely reactive:
 * - No self-scheduled events
 * - All processing triggered by incoming CacheReqPacket
 * - Cache array would be initialized here (if implemented)
 *
 * @note Commented code shows how to schedule cache-initiated events if needed
 * @see handleNOCRequest() For request processing entry point
 */
void CacheSim::init() {
	// TODO: Should schedule the events into event queue.
	// for (Tick i = 1; i < 10; ++i) {
	// 	// Schedule the event for testing.
	// 	CacheEvent* cache_event = new CacheEvent(std::to_string(i));
	// 	scheduleEvent(cache_event, i * 2 + 1);
	// }
}

/**
 * @brief Clean up CacheSim resources at simulation end
 *
 * This method is called once after the simulation loop completes.
 *
 * **Automatic Cleanup:**
 * - Channel ports managed by ChannelPortManager
 * - Events auto-delete after process() completes
 * - No dynamic allocations in this simple implementation
 *
 * @note Real cache would free cache array and tag array here
 * @see SimBase::cleanup() For base class cleanup behavior
 */
void CacheSim::cleanup() {
	// TODO: Release the dynamic memory, clean up the event queue, ...etc.

	// clean up the event queue
}

/**
 * @brief Process cache request and invoke callback with response
 *
 * This method is invoked by the visitor pattern when CacheReqPacket arrives from
 * NocSim. Unlike traditional channel-based response, this method uses callback-based
 * delivery for simplicity and efficiency.
 *
 * **Processing Steps:**
 * 1. Generate pseudo response data (data = 100)
 * 2. Create NocRespPacket with data
 * 3. Calculate size-dependent latency
 * 4. Add Chrome trace record for visualization
 * 5. Extract callback from request packet
 * 6. Invoke callback with response data and target tick
 *
 * **Latency Calculation:**
 * ```
 * latency = cacheMemoryLatency + (size + 1) / cacheMemoryBandwidth
 *
 * Example (size=256):
 *   = 1 + (256 + 1) / 32
 *   = 1 + 8
 *   = 9 ticks
 *
 * Components:
 *   - Base latency (1): Tag lookup, coherence check
 *   - Transfer time (8): Data transfer at 32 bytes/tick
 * ```
 *
 * **Callback Invocation:**
 * ```cpp
 * auto callerCallback = cacheReqPkt->getCallback();
 * // callerCallback is NocSim::nocReqCallback
 *
 * callerCallback(when + latency, id, nocRespPkt, this);
 * // Immediately calls NocSim::nocReqCallback() on same stack
 * // 'when + latency' models the response delivery time
 * // 'this' provides CacheSim pointer to callback
 * ```
 *
 * **Why Callback Instead of Channel?**
 * - Simpler: No event/packet wrapping needed
 * - Efficient: No event scheduling overhead
 * - Flexible: Callback can perform complex logic
 * - Equivalent: Latency still modeled via 'when' parameter
 *
 * **Pseudo Data Generation:**
 * Current implementation returns constant data (100).
 * Real cache would:
 * - Check tags for hit/miss
 * - Read from cache array on hit
 * - Initiate miss handling on miss
 * - Update replacement policy state
 *
 * @param cacheReqPkt Request packet from NOC
 * @param id Transaction ID (simulator ID)
 * @param when Tick when request arrived
 *
 * @note Callback invoked synchronously (same tick as handleNOCRequest)
 * @note Latency modeled by 'when + getRespDelay(size)' parameter
 * @note Chrome trace shows "Cache2NocSim" with actual latency duration
 *
 * @see CacheReqPacket::visit() For visitor pattern invocation
 * @see NocSim::nocReqCallback() For callback destination
 * @see getRespDelay() For latency calculation
 */
void CacheSim::handleNOCRequest(CacheReqPacket* cacheReqPkt, int id, Tick when) {
	int            data       = 100;
	NocRespPacket* nocRespPkt = new NocRespPacket(NocPktTypeEnum::TEST, data);
	top->addChromeTraceRecord(acalsim::ChromeTraceRecord::createCompleteEvent("CacheSim", "Cache2NocSim", when,
	                                                                          getRespDelay(cacheReqPkt->getSize())));
	auto callerCallback =
	    cacheReqPkt->getCallback();  // callerCallback = CacheReqPacket::callback = NocSim::nocReqCallback
	callerCallback(when + getRespDelay(cacheReqPkt->getSize()), id, nocRespPkt, this);
}
