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
 * @brief Network-on-Chip (NoC) Simulator Implementation
 *
 * @details
 * This file implements the NocSim class, which serves as the central routing and protocol
 * translation component in the testSimChannel memory subsystem. The NoC acts as an
 * intermediary between the TrafficGenerator (CPU model) and CacheSim (cache memory),
 * performing packet transformation, transaction tracking, and bidirectional routing.
 *
 * # Component Role in System Architecture
 *
 * The NocSim functions as a network router and protocol converter:
 *
 * @code{.unparsed}
 *                    TrafficGenerator
 *                          ↓ NocReqPacket
 *              ┌───────────────────────────┐
 *              │  Slave Port: TG2NOC-s     │ (receives requests)
 *              └───────────┬───────────────┘
 *                          ↓
 *              ┌───────────────────────────┐
 *              │      NocSim (Router)      │
 *              │  ┌─────────────────────┐  │
 *              │  │  Request Queue      │  │
 *              │  │  (Transaction DB)   │  │
 *              │  │  tid → NocReqPacket │  │
 *              │  └─────────────────────┘  │
 *              │  ┌─────────────────────┐  │
 *              │  │ Packet Transformer  │  │
 *              │  │ NocReq ↔ CacheReq   │  │
 *              │  │ CacheResp ↔ NocResp │  │
 *              │  └─────────────────────┘  │
 *              └───────────┬───────────────┘
 *                          ↓
 *              ┌───────────────────────────┐
 *              │ Master Port: NOC2Cache-m  │ (sends to cache)
 *              └───────────┬───────────────┘
 *                          ↓ CacheReqPacket
 *                     CacheSim
 *                          ↑ CacheRespPacket
 *              ┌───────────────────────────┐
 *              │ Slave Port: Cache2NOC-s   │ (receives responses)
 *              └───────────┬───────────────┘
 *                          ↓
 *              ┌───────────────────────────┐
 *              │      NocSim (Router)      │
 *              │  [Lookup in reqQueue]     │
 *              │  [Create NocRespPacket]   │
 *              └───────────┬───────────────┘
 *                          ↓
 *              ┌───────────────────────────┐
 *              │  Master Port: NOC2TG-m    │ (sends responses)
 *              └───────────┬───────────────┘
 *                          ↓ NocRespPacket
 *                    TrafficGenerator
 * @endcode
 *
 * # Key Responsibilities
 *
 * ## 1. Request Path (Forward Routing)
 * - **Receive**: NocReqPacket from TrafficGenerator
 * - **Store**: Original request in reqQueue (indexed by transaction ID)
 * - **Transform**: NocReqPacket → CacheReqPacket (protocol conversion)
 * - **Forward**: CacheReqPacket to CacheSim
 *
 * ## 2. Response Path (Backward Routing)
 * - **Receive**: CacheRespPacket from CacheSim
 * - **Lookup**: Original request in reqQueue using transaction ID
 * - **Transform**: CacheRespPacket → NocRespPacket (protocol conversion)
 * - **Forward**: NocRespPacket to TrafficGenerator
 *
 * ## 3. Transaction State Management
 * - Maintains a mapping of transaction ID → original NocReqPacket
 * - Ensures request-response matching for out-of-order completion
 * - Prevents lost transactions through queue-based tracking
 *
 * # Packet Transformation Details
 *
 * ## Forward Path: NocReqPacket → CacheReqPacket
 *
 * Input packet (from TrafficGenerator):
 * @code
 * NocReqPacket {
 *     type: NocPktTypeEnum::TEST
 *     addr: 0x0000
 *     size: 256 bytes
 *     tid:  0
 * }
 * @endcode
 *
 * Transformation in handleTGRequest():
 * @code
 * void handleTGRequest(NocReqPacket* nocReqPkt) {
 *     // 1. Store original request for later matching
 *     int tid = nocReqPkt->getTransactionId();
 *     reqQueue->add(tid, nocReqPkt);
 *
 *     // 2. Extract request parameters
 *     int addr = nocReqPkt->getAddr();  // 0x0000
 *     int size = nocReqPkt->getSize();  // 256
 *
 *     // 3. Create cache-protocol packet
 *     auto cacheReqPkt = new CacheReqPacket(
 *         CachePktTypeEnum::TEST,
 *         addr, size, tid
 *     );
 *
 *     // 4. Forward immediately (zero delays)
 *     pushToMasterChannelPort("NOC2Cache-m", cacheReqPkt);
 * }
 * @endcode
 *
 * Output packet (to CacheSim):
 * @code
 * CacheReqPacket {
 *     type: CachePktTypeEnum::TEST
 *     addr: 0x0000
 *     size: 256 bytes
 *     tid:  0
 * }
 * @endcode
 *
 * ## Backward Path: CacheRespPacket → NocRespPacket
 *
 * Input packet (from CacheSim):
 * @code
 * CacheRespPacket {
 *     type: CachePktTypeEnum::TEST
 *     data: 111 (pointer to int)
 *     tid:  0
 * }
 * @endcode
 *
 * Transformation in handleCacheRespond():
 * @code
 * void handleCacheRespond(CacheRespPacket* cacheRespPkt) {
 *     // 1. Extract transaction ID
 *     int tid = cacheRespPkt->getTransactionId();
 *
 *     // 2. Retrieve original request from queue
 *     NocReqPacket* origReq = (NocReqPacket*)reqQueue->get(tid);
 *
 *     if (origReq) {
 *         // 3. Extract response data
 *         int* data = cacheRespPkt->getData();
 *
 *         // 4. Create NoC-protocol response
 *         auto nocRespPkt = new NocRespPacket(
 *             NocPktTypeEnum::TEST,
 *             data, tid
 *         );
 *
 *         // 5. Route back to traffic generator
 *         pushToMasterChannelPort("NOC2TG-m", nocRespPkt);
 *     }
 * }
 * @endcode
 *
 * Output packet (to TrafficGenerator):
 * @code
 * NocRespPacket {
 *     type: NocPktTypeEnum::TEST
 *     data: 111 (pointer to int)
 *     tid:  0
 * }
 * @endcode
 *
 * # Transaction Queue (reqQueue) Management
 *
 * The NoC uses an UnorderedRequestQueue to track in-flight transactions:
 *
 * ## Queue Structure:
 * @code
 * UnorderedRequestQueue<SimPacket*>* reqQueue;
 * // Maps: int (transaction ID) → SimPacket* (original NocReqPacket)
 * @endcode
 *
 * ## Queue Operations:
 *
 * ### Adding Requests (on receive from TrafficGenerator):
 * @code
 * reqQueue->add(tid, (SimPacket*)nocReqPkt);
 * // tid=0 → NocReqPacket{addr=0, size=256, ...}
 * // tid=1 → NocReqPacket{addr=0, size=256, ...}
 * @endcode
 *
 * ### Retrieving Requests (on receive from CacheSim):
 * @code
 * NocReqPacket* origReq = (NocReqPacket*)reqQueue->get(tid);
 * // Retrieves and REMOVES the entry from queue
 * // Returns nullptr if tid not found (error condition)
 * @endcode
 *
 * ## Queue Lifecycle Example:
 * @code{.unparsed}
 * Time  │ Event                          │ Queue State
 * ──────┼────────────────────────────────┼─────────────────────────
 * T=11  │ Req tid=0 arrives              │ {0 → NocReqPkt}
 * T=11  │ Forward to cache               │ {0 → NocReqPkt}
 * T=13  │ Req tid=1 arrives              │ {0 → NocReqPkt, 1 → NocReqPkt}
 * T=13  │ Forward to cache               │ {0 → NocReqPkt, 1 → NocReqPkt}
 * T=20  │ Resp tid=0 returns             │ {1 → NocReqPkt}
 * T=23  │ Resp tid=1 returns             │ {} (empty)
 * @endcode
 *
 * # SimChannel Port Configuration
 *
 * The NoC uses 4 SimChannel ports for bidirectional communication:
 *
 * | Port Name      | Type   | Direction | Connected To      | Packet Type      |
 * |----------------|--------|-----------|-------------------|------------------|
 * | TG2NOC-s       | Slave  | Incoming  | TrafficGenerator  | NocReqPacket     |
 * | NOC2Cache-m    | Master | Outgoing  | CacheSim          | CacheReqPacket   |
 * | Cache2NOC-s    | Slave  | Incoming  | CacheSim          | CacheRespPacket  |
 * | NOC2TG-m       | Master | Outgoing  | TrafficGenerator  | NocRespPacket    |
 *
 * ## Port Usage in Code:
 *
 * ### Sending to Cache (Request):
 * @code
 * this->pushToMasterChannelPort("NOC2Cache-m", (SimPacket*)cacheReqPkt);
 * // Immediate send (no local/remote delays specified)
 * // Equivalent to: sendPacketViaChannel("NOC2Cache-m", 0, 0, cacheReqPkt)
 * @endcode
 *
 * ### Sending to TrafficGenerator (Response):
 * @code
 * this->pushToMasterChannelPort("NOC2TG-m", (SimPacket*)nocRespPkt);
 * // Immediate send (zero latency routing)
 * @endcode
 *
 * # Timing Model and Performance Parameters
 *
 * ## Configuration Constants:
 * - **nocRespDelay**: 1 tick (response routing latency)
 * - **nocBandwidth**: 32 bytes/tick (network throughput)
 *
 * ## Current Implementation:
 * Both handleTGRequest() and handleCacheRespond() use **immediate forwarding**:
 * - Local delay: 0 ticks
 * - Remote delay: 0 ticks
 * - Total routing latency: 0 ticks
 *
 * This models an **ideal zero-latency router** for testing purposes.
 *
 * ## Realistic NoC Timing Extension:
 * To model actual network delays:
 * @code
 * void handleTGRequest(NocReqPacket* nocReqPkt) {
 *     reqQueue->add(tid, nocReqPkt);
 *
 *     // Calculate routing delay based on packet size
 *     int routingHops = 3;  // Number of router hops
 *     int hopLatency = 2;   // Cycles per hop
 *     int serialDelay = (nocReqPkt->getSize() + nocBandwidth - 1) / nocBandwidth;
 *     int totalDelay = routingHops * hopLatency + serialDelay;
 *
 *     auto cacheReqPkt = new CacheReqPacket(...);
 *     this->sendPacketViaChannel("NOC2Cache-m", 0, totalDelay, cacheReqPkt);
 * }
 * @endcode
 *
 * # Visitor Pattern Integration
 *
 * Packets route themselves to the correct handler using the visitor pattern:
 *
 * ## Request Handling:
 * @code
 * // In NocReqPacket.cc:
 * void NocReqPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto* nocSim = dynamic_cast<NocSim*>(&simulator)) {
 *         nocSim->handleTGRequest(this);  // Calls this file's method
 *     }
 * }
 * @endcode
 *
 * ## Response Handling:
 * @code
 * // In CacheRespPacket.cc:
 * void CacheRespPacket::visit(Tick when, SimBase& simulator) {
 *     if (auto* nocSim = dynamic_cast<NocSim*>(&simulator)) {
 *         nocSim->handleCacheRespond(this);  // Calls this file's method
 *     }
 * }
 * @endcode
 *
 * The framework automatically calls visit() when packets arrive at their destination port.
 *
 * # Error Handling
 *
 * ## Transaction ID Mismatch:
 * @code
 * void handleCacheRespond(CacheRespPacket* cacheRespPkt) {
 *     if (auto reqPkt = reqQueue->get(id)) {
 *         // Success: found matching request
 *     } else {
 *         CLASS_INFO << "Packet not found !";
 *         // Possible causes:
 *         // 1. Response for unknown transaction
 *         // 2. Duplicate response
 *         // 3. Transaction ID corruption
 *     }
 * }
 * @endcode
 *
 * # Usage in Custom NoC Architectures
 *
 * ## Extending to Multi-Hop Routing:
 * @code
 * class MeshNocSim : public NocSim {
 * private:
 *     struct Position { int x, y; };
 *     std::map<int, Position> nodePositions;
 *
 * public:
 *     void handleTGRequest(NocReqPacket* req) override {
 *         // Calculate Manhattan distance routing
 *         Position src = nodePositions[req->getSrcNode()];
 *         Position dst = nodePositions[req->getDstNode()];
 *         int hops = abs(dst.x - src.x) + abs(dst.y - src.y);
 *         int delay = hops * hopLatency;
 *
 *         reqQueue->add(req->getTransactionId(), req);
 *         auto cacheReq = transformToCache(req);
 *         sendPacketViaChannel("NOC2Cache-m", 0, delay, cacheReq);
 *     }
 * };
 * @endcode
 *
 * ## Adding Congestion Modeling:
 * @code
 * class CongestionAwareNoC : public NocSim {
 * private:
 *     int currentLoad = 0;
 *     const int maxBandwidth = 1024;  // bytes/tick
 *
 * public:
 *     void handleTGRequest(NocReqPacket* req) override {
 *         currentLoad += req->getSize();
 *         int congestionDelay = (currentLoad > maxBandwidth)
 *                               ? (currentLoad / maxBandwidth)
 *                               : 0;
 *
 *         reqQueue->add(req->getTransactionId(), req);
 *         auto cacheReq = new CacheReqPacket(...);
 *         sendPacketViaChannel("NOC2Cache-m", 0, congestionDelay, cacheReq);
 *
 *         // Schedule bandwidth recovery
 *         scheduleEvent(10, [this, size=req->getSize()]() {
 *             currentLoad -= size;
 *         });
 *     }
 * };
 * @endcode
 *
 * # Design Patterns and Best Practices
 *
 * ## 1. Protocol Layering:
 * - NoC packets (NocReq/NocResp): Network layer
 * - Cache packets (CacheReq/CacheResp): Memory layer
 * - Clean separation allows independent protocol evolution
 *
 * ## 2. Transaction Tracking:
 * - Always store original request before forwarding
 * - Use transaction IDs for out-of-order response handling
 * - Check for queue lookup failures (defensive programming)
 *
 * ## 3. Zero-Copy Packet Transformation:
 * - Only copy necessary fields (addr, size, tid)
 * - Data pointers are shared (no deep copy)
 * - Original packets freed after response completion
 *
 * ## 4. Immediate vs. Delayed Forwarding:
 * - pushToMasterChannelPort(): Immediate (0-latency)
 * - sendPacketViaChannel(): Delayed (specify timing)
 * - Choose based on modeling fidelity requirements
 *
 * @see NocSim.hh For class declaration and configuration constants
 * @see TrafficGenerator For upstream request source
 * @see CacheSim For downstream memory target
 * @see NocPacket For NoC packet type definitions
 * @see CachePacket For cache packet type definitions
 * @see testSimChannel.cc For complete system integration
 *
 * @author ACAL Team
 * @date 2023-2025
 * @version 1.0
 *
 * @note This is a simplified NoC for testing SimChannel patterns. Production NoCs
 *       should include routing algorithms, congestion control, flow control,
 *       virtual channels, and quality-of-service mechanisms.
 *
 * @warning Ensure reqQueue cleanup if transactions can be cancelled or timed out.
 *          Memory leaks may occur if responses never arrive for stored requests.
 */

#include "NocSim.hh"

void NocSim::handleTGRequest(NocReqPacket* _nocReqPkt) {
	CLASS_INFO << "NocSim::handleTGRequest at Tick = " << top->getGlobalTick();

	auto id = _nocReqPkt->getTransactionId();
	// Add NocReqPacket to ReqQueue
	reqQueue->add(id, (SimPacket*)_nocReqPkt);
	// Create CacheReqPacket
	auto cacheReqPkt = new CacheReqPacket(CachePktTypeEnum::TEST, _nocReqPkt->getAddr(), _nocReqPkt->getSize(), id);

	// Send CacheReqPacket to Cache
	CLASS_INFO << "NocSim sendPacketViaChannel with local latency = " << 0 << ", remote latency = " << 0;
	this->pushToMasterChannelPort("NOC2Cache-m", (SimPacket*)cacheReqPkt);
}

void NocSim::handleCacheRespond(CacheRespPacket* _cacheRespPkt) {
	auto id = _cacheRespPkt->getTransactionId();
	if (auto reqPkt = ((NocReqPacket*)reqQueue->get(id))) {
		// Create NocRespPacket
		auto nocRespPacket = new NocRespPacket(NocPktTypeEnum::TEST, _cacheRespPkt->getData(), id);

		// Send NocRespPacket to Traffic Generator
		CLASS_INFO << "NocSim sendPacketViaChannel with local latency = " << 0 << ", remote latency = " << 0;
		this->pushToMasterChannelPort("NOC2TG-m", (SimPacket*)nocRespPacket);
	} else {
		CLASS_INFO << "Packet not found !";
	}
}
