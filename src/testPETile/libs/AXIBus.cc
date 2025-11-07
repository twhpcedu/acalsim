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
 * @file AXIBus.cc
 * @brief AXI bus interconnect with transaction routing and callback-based response delivery
 *
 * This file implements the AXI bus module, which serves as the central interconnect
 * in the PE tile architecture. It demonstrates transaction ID management, request/response
 * routing, latency modeling, and callback chain orchestration for memory access operations.
 *
 * **AXI Bus Role in PE Tile Interconnect:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                          AXI Bus Module                                │
 * │                    (Interconnect Network Layer)                        │
 * │                                                                        │
 * │  Request Path (CPUTraffic → SRAM):                                     │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Receive MemReqPacket from CPUTraffic                          │ │
 * │  │    ├─ Via CPUReqEvent → visit() → memReqPktHandler()            │ │
 * │  │    └─ Extract caller's callback (CPUReqEvent::cpuReqCallback)   │ │
 * │  │                                                                  │ │
 * │  │ 2. Create BusReqEvent for SRAM                                   │ │
 * │  │    ├─ Assign new transaction ID (auto-increment)                │ │
 * │  │    ├─ Store caller's callback for response routing              │ │
 * │  │    └─ Schedule event with bus_req_delay latency                 │ │
 * │  │                                                                  │ │
 * │  │ 3. BusReqEvent::process() forwards to SRAM                       │ │
 * │  │    ├─ Set callback to BusReqEvent::busReqCallback()             │ │
 * │  │    └─ Call SRAM::accept() with updated packet                   │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Response Path (SRAM → CPUTraffic):                                    │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ 4. SRAM invokes BusReqEvent::busReqCallback()                    │ │
 * │  │    ├─ Receives MemRespPacket from SRAM                           │ │
 * │  │    ├─ Create BusRespEvent for original caller                    │ │
 * │  │    ├─ Schedule with bus_resp_delay + size-dependent delay       │ │
 * │  │    └─ Update statistics (numResp++)                              │ │
 * │  │                                                                  │ │
 * │  │ 5. BusRespEvent::process() delivers to CPUTraffic                │ │
 * │  │    ├─ Invoke caller's original callback                         │ │
 * │  │    └─ CPUReqEvent::cpuReqCallback() → CPUTraffic::MemRespHandler│ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Scalable Mesh Extension (Future):                                     │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ Multi-Tile Routing:                                              │ │
 * │  │   ├─ Add tile coordinates (x, y) to packets                      │ │
 * │  │   ├─ Implement XY/dimension-ordered routing                      │ │
 * │  │   ├─ Add virtual channel support for deadlock avoidance          │ │
 * │  │   └─ Inter-tile flow control and backpressure                    │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Transaction Flow with Callback Chain:**
 * ```
 * Complete Request-Response Cycle:
 * ═══════════════════════════════════════════════════════════════════
 *
 * Tick T0: CPUReqEvent sends MemReqPacket to AXI Bus
 *   │
 *   └─> AXIBus::memReqPktHandler(when=T0, pkt)
 *       ├─ Extract callerCallback = CPUReqEvent::cpuReqCallback
 *       ├─ Find downstream SRAM via getDownStream("PCUMem")
 *       ├─ Create BusReqEvent(tid=0, sram, callerCallback, pkt, this)
 *       └─ Schedule at T0 + bus_req_delay (e.g., T0+2)
 *
 * Tick T0+2: BusReqEvent::process() executes
 *   │
 *   ├─ Create lambda callback: λ(id, resp) { busReqCallback(tid, caller, resp); }
 *   ├─ Attach callback to memReqPkt
 *   └─> SRAM::accept(when=T0+2, memReqPkt)
 *       └─> SRAM::memReqPktHandler()
 *           ├─ Process memory request
 *           ├─ Calculate SRAM latency (sram_req_delay + size/256)
 *           └─ Schedule SRAMRespEvent at T0+2+latency
 *
 * Tick T0+2+sram_latency: SRAMRespEvent::process() executes
 *   │
 *   └─> Invoke callback (BusReqEvent::busReqCallback)
 *       ├─ INPUT: tid=0, module=AXIBus, memRespPkt
 *       ├─ Create BusRespEvent(tid, callerCallback, memRespPkt)
 *       ├─ Calculate bus response delay: bus_resp_delay + size/32
 *       ├─ Schedule BusRespEvent at current_tick + resp_delay
 *       └─ Update stats: numResp++
 *
 * Tick T0+2+sram_latency+resp_delay: BusRespEvent::process() executes
 *   │
 *   └─> Invoke callerCallback (CPUReqEvent::cpuReqCallback)
 *       └─> CPUTraffic::MemRespHandler(id, memRespPkt)
 *           ├─ Update CPUTraffic statistics
 *           └─ Free response packet
 * ```
 *
 * **BusReqEvent Class Hierarchy:**
 * ```
 * BusReqEvent : public CallbackEvent<void(int, MemRespPacket*)>
 *   │
 *   ├─ Members:
 *   │  ├─ tid (int)              : Transaction ID for downstream module
 *   │  ├─ callee (SimModule*)    : Pointer to downstream SRAM
 *   │  ├─ callerCallback (λ)     : Original caller's callback (CPUReqCallback)
 *   │  ├─ memReqPkt (MemReqPkt*) : Request packet to forward
 *   │  └─ caller (SimModule*)    : Pointer to AXI Bus (this)
 *   │
 *   ├─ Methods:
 *   │  ├─ process()              : Forward request to SRAM with new callback
 *   │  └─ busReqCallback()       : Create BusRespEvent when SRAM responds
 *   │
 *   └─ Callback Chain:
 *      Original: CPUReqEvent::cpuReqCallback
 *      Wrapped:  BusReqEvent::busReqCallback
 *      Result:   Two-level callback nesting for routing
 * ```
 *
 * **BusRespEvent Class Hierarchy:**
 * ```
 * BusRespEvent : public CallbackEvent<void(int, MemRespPacket*)>
 *   │
 *   ├─ Members:
 *   │  ├─ tid (int)              : Transaction ID from bus
 *   │  ├─ callerCallback (λ)     : Original caller's callback
 *   │  └─ memRespPkt (MemRespPkt*) : Response packet to deliver
 *   │
 *   ├─ Methods:
 *   │  └─ process()              : Invoke caller's callback with response
 *   │
 *   └─ Purpose:
 *      Deliver response back to original requester (CPUTraffic)
 *      with appropriate bus response latency
 * ```
 *
 * **AXI Bus Transaction ID Management:**
 * ```
 * Transaction ID Tracking:
 * ════════════════════════════════════════════
 *
 * AXI Bus maintains internal transaction counter:
 *   - transactionID (int): Auto-incrementing counter
 *   - Initialized to 0 in constructor
 *   - Incremented for each new request: transactionID++
 *
 * Example Sequence:
 *   Req 0: tid=0, addr=0x0000  → transactionID becomes 1
 *   Req 1: tid=1, addr=0x1000  → transactionID becomes 2
 *   Req 2: tid=2, addr=0x2000  → transactionID becomes 3
 *
 * Use Cases:
 *   - Out-of-order response tracking
 *   - Multi-outstanding request management
 *   - Debugging and trace correlation
 *   - Protocol compliance (AXI requires unique IDs)
 * ```
 *
 * **Latency Modeling:**
 * ```
 * Request Latency (bus_req_delay):
 * ════════════════════════════════════════════
 *   - Fixed delay from configuration
 *   - Models bus arbitration, routing, serialization
 *   - Configured via: top->getParameter<Tick>("PETile", "bus_req_delay")
 *   - Example: 2 ticks
 *
 * Response Latency (bus_resp_delay + size/bandwidth):
 * ════════════════════════════════════════════
 *   - Base delay: bus_resp_delay (fixed)
 *   - Transfer time: (size + 1) / 32
 *     └─ Assumes 32 bytes/tick bandwidth
 *     └─ +1 prevents zero delay for size=0
 *   - Total: getRespDelay(size) = base + transfer
 *
 * Example Calculations:
 *   size=0:   delay = 3 + (0+1)/32 = 3 + 0 = 3 ticks
 *   size=20:  delay = 3 + (20+1)/32 = 3 + 0 = 3 ticks
 *   size=64:  delay = 3 + (64+1)/32 = 3 + 2 = 5 ticks
 *   size=256: delay = 3 + (256+1)/32 = 3 + 8 = 11 ticks
 * ```
 *
 * **Callback Chain Mechanism:**
 * ```
 * Callback Wrapping for Response Routing:
 * ════════════════════════════════════════════
 *
 * Step 1: CPUReqEvent creates request with callback:
 *   callback = λ(id, resp) { cpuReqCallback(tid, resp); }
 *   memReqPkt->setCallback(callback)
 *   Send to AXI Bus
 *
 * Step 2: AXI Bus extracts and stores callback:
 *   callerCallback = memReqPkt->getCallback()
 *   // callerCallback = CPUReqEvent::cpuReqCallback
 *
 * Step 3: BusReqEvent wraps callback:
 *   newCallback = λ(id, resp) { busReqCallback(tid, caller, resp); }
 *   memReqPkt->setCallback(newCallback)
 *   Send to SRAM
 *
 * Step 4: SRAM invokes wrapped callback:
 *   callback(id, respPkt)
 *   → Calls BusReqEvent::busReqCallback()
 *
 * Step 5: busReqCallback creates BusRespEvent:
 *   BusRespEvent(tid, callerCallback, respPkt)
 *   // Uses original CPUReqEvent::cpuReqCallback
 *
 * Step 6: BusRespEvent delivers to original caller:
 *   callerCallback(tid, respPkt)
 *   → Calls CPUReqEvent::cpuReqCallback()
 *   → Eventually CPUTraffic::MemRespHandler()
 * ```
 *
 * **Statistics Collection:**
 * ```
 * AXI Bus Statistics:
 * ════════════════════════════════════════════
 *
 * struct Stats {
 *     uint32_t numResp = 0;  // Number of responses routed
 * }
 *
 * Update Points:
 *   - Incremented in BusReqEvent::busReqCallback()
 *   - Tracked per AXI Bus instance
 *   - Used for performance analysis and testing
 *
 * Google Test Integration:
 *   - When isGTestMode() == true
 *   - Check if numResp == NUM_TEST_REQ (typically 5)
 *   - Set test completion bit mask
 *   - Enables automated test verification
 * ```
 *
 * **Mesh/Torus Extension for Multi-Tile Arrays:**
 * ```
 * Current: Single Tile, Point-to-Point
 * ═══════════════════════════════════════════════
 * CPUTraffic ──> AXI Bus ──> SRAM
 *
 * Future: Multi-Tile Mesh Interconnect
 * ═══════════════════════════════════════════════
 *
 * 2D Mesh Topology (2x2 example):
 * ┌─────────────┬─────────────┐
 * │ Tile(0,0)   │ Tile(0,1)   │
 * │ ┌────┐      │ ┌────┐      │
 * │ │Bus0│──────┼─│Bus1│      │
 * │ └────┘      │ └────┘      │
 * ├─────────────┼─────────────┤
 * │ Tile(1,0)   │ Tile(1,1)   │
 * │ ┌────┐      │ ┌────┐      │
 * │ │Bus2│──────┼─│Bus3│      │
 * │ └────┘      │ └────┘      │
 * └─────────────┴─────────────┘
 *       │              │
 *    North-South Links
 *
 * Routing Extensions:
 *   1. Add destination coordinates (dst_x, dst_y) to packets
 *   2. Implement XY routing:
 *      - Route in X dimension first
 *      - Then route in Y dimension
 *   3. Add ports: North, South, East, West, Local
 *   4. Port selection logic:
 *      if (current_x < dst_x) → East
 *      else if (current_x > dst_x) → West
 *      else if (current_y < dst_y) → North
 *      else if (current_y > dst_y) → South
 *      else → Local
 *
 * Virtual Channels (VC) for Deadlock Avoidance:
 *   - Separate VCs for different message types
 *   - Request VC, Response VC, etc.
 *   - Prevents cyclic dependencies in mesh
 *
 * Flow Control:
 *   - Credit-based backpressure
 *   - Buffer occupancy tracking
 *   - Wormhole or virtual cut-through switching
 * ```
 *
 * **Module Connectivity:**
 * ```
 * AXI Bus Port Configuration:
 * ═══════════════════════════════════════════════
 *
 * Upstream Connections (Response Path):
 *   - "PCU": Connected to CPUTraffic
 *     └─ Receives requests from CPU
 *
 * Downstream Connections (Request Path):
 *   - "PCUMem": Connected to SRAM
 *     └─ Sends requests to memory
 *
 * Port Discovery:
 *   SimModule* sram = getDownStream("PCUMem")
 *   SimModule* cpu  = getUpStream("PCU")
 * ```
 *
 * **Error Handling:**
 * ```
 * Assertion Checks:
 *   CLASS_ASSERT_MSG(sram, "Cannot find the PCUMem module!\n")
 *     - Validates downstream SRAM connection exists
 *     - Prevents null pointer dereference
 *     - Fails fast with descriptive error message
 *
 * Debug Logging:
 *   CLASS_INFO << "Schedule busReqPacket with transaction id: " << tid
 *     - Logs all transaction scheduling events
 *     - Enables trace-based debugging
 *     - Correlates with CPUReq traces
 * ```
 *
 * **Design Patterns:**
 *
 * 1. **Visitor Pattern:**
 *    - MemReqPacket::visit(AXIBus&) dispatches to memReqPktHandler()
 *    - Type-safe packet handling without explicit casting
 *    - Extensible for new packet types
 *
 * 2. **Callback Chain Pattern:**
 *    - Nested callbacks for multi-hop routing
 *    - Each layer wraps previous callback
 *    - Enables response path reversal
 *
 * 3. **Event-Driven Simulation:**
 *    - BusReqEvent and BusRespEvent model timing
 *    - Scheduled events represent future actions
 *    - Automatic time advancement via event queue
 *
 * **Related Files:**
 * - Header: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/AXIBus.hh
 * - SRAM: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/SRAM.cc
 * - CPU Traffic: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUTraffic.cc
 * - CPU Req Event: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/CPUReqEvent.cc
 * - Mem Packets: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/MemReq.cc
 * - PETile: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/PETile.cc
 *
 * @author ACALSim Framework Team
 * @date 2023-2025
 */

#include "AXIBus.hh"

#include "SRAM.hh"
#include "Test.hh"

/**
 * @brief Callback invoked when SRAM completes a memory request
 *
 * This callback is the response routing mechanism in the AXI bus. When SRAM finishes
 * processing a memory request, it invokes this callback to route the response back
 * to the original requester (CPUTraffic) through the bus with appropriate latency.
 *
 * **Callback Invocation Context:**
 * ```
 * SRAM::memReqPktHandler() creates SRAMRespEvent
 *   │
 *   └─> SRAMRespEvent::process() executes
 *       └─> Invokes callback (this function)
 *           ├─ INPUT: _tid = transaction ID
 *           │         module = AXI Bus instance
 *           │         _memRespPkt = response from SRAM
 *           │
 *           ├─ Create BusRespEvent to deliver response
 *           ├─ Calculate bus response latency
 *           └─ Schedule BusRespEvent for original caller
 * ```
 *
 * **Response Routing Flow:**
 * ```
 * BusReqCallback Execution:
 *   │
 *   ├─ Verify callerCallback exists (original CPUReqEvent callback)
 *   │
 *   ├─ Log callback invocation with transaction ID and tick
 *   │
 *   ├─ Create BusRespEvent:
 *   │  ├─ tid: Transaction ID for correlation
 *   │  ├─ callerCallback: Original requester's callback
 *   │  └─ memRespPkt: Response packet from SRAM
 *   │
 *   ├─ Calculate response delay:
 *   │  └─ getRespDelay(size) = bus_resp_delay + (size+1)/32
 *   │
 *   ├─ Schedule BusRespEvent at current_tick + delay
 *   │
 *   └─ Update AXI Bus statistics (numResp++)
 * ```
 *
 * @param _tid Transaction ID assigned by AXI bus
 * @param module Pointer to AXI Bus module instance
 * @param _memRespPkt Response packet from SRAM containing result data
 *
 * @note This method is invoked as a lambda callback from SRAMRespEvent
 * @note The callerCallback is CPUReqEvent::cpuReqCallback() stored during request
 * @note Statistics are updated here for response counting and test verification
 *
 * @see BusReqEvent::process() Where this callback is registered
 * @see BusRespEvent Response event that delivers to original caller
 * @see AXIBus::getRespDelay() Response latency calculation
 */
void BusReqEvent::busReqCallback(int        _tid, /* transaction ID to the downstream module */
                                 SimModule* module, MemRespPacket* _memRespPkt) {
	if (callerCallback) {  // callerCallback = CPUReqEvent::CPUReqCallback()
		CLASS_INFO << "BusReqEvent::busReqCallback()  transaction id: " << this->tid
		           << " at Tick=" << top->getGlobalTick();
		BusRespEvent* busRespEvent = new BusRespEvent(_tid, callerCallback, _memRespPkt);
		module->scheduleEvent((SimEvent*)busRespEvent,
		                      top->getGlobalTick() + ((AXIBus*)caller)->getRespDelay(_memRespPkt->getSize()));
	}
	((AXIBus*)module)->getStats()->numResp++;
	if (top->isGTestMode() && ((AXIBus*)module)->getStats()->numResp == acalsim::gtest::NUM_TEST_REQ) {
		CLASS_INFO << "simID: " << module->getSimID() << " ID: " << module->getID();
		top->setGTestBitMask(module->getSimID(), module->getID() /*  for first critiria*/);
	}
}

/**
 * @brief Processes bus request event by forwarding to SRAM with wrapped callback
 *
 * This method implements the request forwarding logic from AXI bus to SRAM. It wraps
 * the original caller's callback with busReqCallback() to enable response routing
 * back through the bus.
 *
 * **Event Processing Flow:**
 * ```
 * BusReqEvent::process() invoked by event queue
 *   │
 *   ├─ Log processing with transaction ID and current tick
 *   │
 *   ├─ Create new callback lambda:
 *   │  └─ λ(id, resp) { busReqCallback(tid, caller, resp); }
 *   │     └─ Wraps response delivery through bus
 *   │
 *   ├─ Attach new callback to memory request packet
 *   │  └─ memReqPkt->setCallback(callback)
 *   │
 *   └─> Forward to SRAM via accept():
 *       └─ SRAM::accept(when, memReqPkt)
 *          └─> SRAM::memReqPktHandler() processes request
 * ```
 *
 * **Callback Chain Setup:**
 * ```
 * Original Callback (from CPUReqEvent):
 *   callerCallback = λ(id, resp) { cpuReqCallback(tid, resp); }
 *
 * Wrapped Callback (created here):
 *   newCallback = λ(id, resp) { busReqCallback(tid, caller, resp); }
 *     │
 *     └─> When SRAM responds:
 *         └─> busReqCallback() creates BusRespEvent
 *             └─> BusRespEvent invokes original callerCallback
 *                 └─> CPUReqEvent::cpuReqCallback() executed
 * ```
 *
 * @note This method is called when the event is dequeued from event queue
 * @note The lambda captures 'this' to access busReqCallback() and member variables
 * @note SRAM is cast from callee pointer (validated during BusReqEvent creation)
 *
 * @see BusReqEvent::busReqCallback() Response routing callback
 * @see SRAM::accept() SRAM packet reception interface
 * @see AXIBus::memReqPktHandler() Where this event is created
 */
void BusReqEvent::process() {
	CLASS_INFO << "Process BusReqEvent with transaction id: " << this->tid << " at Tick=" << top->getGlobalTick();
	auto callback = [this](int id, MemRespPacket* memRespPkt) {
		this->busReqCallback(this->tid, this->caller, this->memReqPkt->getMemRespPkt());
	};
	this->memReqPkt->setCallback(callback);  // set the callback function to BusReqEvent::busReqCallback()
	((SRAM*)callee)->accept(top->getGlobalTick(), (SimPacket&)*memReqPkt);
}

/**
 * @brief Handles incoming memory request packets from CPUTraffic
 *
 * This is the main entry point for memory requests arriving at the AXI bus. It extracts
 * the caller's callback, finds the downstream SRAM, creates a BusReqEvent with a new
 * transaction ID, and schedules it with bus request latency.
 *
 * **Request Processing Pipeline:**
 * ```
 * CPUReqEvent → MemReqPacket → visit(AXIBus) → memReqPktHandler()
 *   │
 *   ├─ Cast packet to MemReqPacket*
 *   │
 *   ├─ Extract caller's callback:
 *   │  └─ callerCallback = memReqPkt->getCallback()
 *   │     └─ This is CPUReqEvent::cpuReqCallback
 *   │
 *   ├─ Find downstream SRAM:
 *   │  ├─ sram = getDownStream("PCUMem")
 *   │  └─ Assert SRAM exists (fail fast if misconfigured)
 *   │
 *   ├─ Assign transaction ID:
 *   │  └─ tid = transactionID++ (auto-increment)
 *   │
 *   ├─ Create BusReqEvent:
 *   │  ├─ tid: Unique transaction identifier
 *   │  ├─ sram: Downstream memory module
 *   │  ├─ callerCallback: Original requester's callback
 *   │  ├─ memReqPkt: Request packet to forward
 *   │  └─ this: AXI Bus instance (for response routing)
 *   │
 *   ├─ Calculate scheduling time:
 *   │  └─ schedule_at = when + bus_req_delay
 *   │
 *   └─ Schedule BusReqEvent for future processing
 * ```
 *
 * **Transaction ID Management:**
 * ```
 * transactionID starts at 0
 *
 * Request 0: tid=0, transactionID becomes 1
 * Request 1: tid=1, transactionID becomes 2
 * Request 2: tid=2, transactionID becomes 3
 * ...
 *
 * Purpose:
 *   - Unique identifier for each bus transaction
 *   - Enables out-of-order response tracking
 *   - Required for AXI protocol compliance
 *   - Useful for debugging and trace correlation
 * ```
 *
 * **Latency Model:**
 * ```
 * Bus Request Delay:
 *   delay = top->getParameter<Tick>("PETile", "bus_req_delay")
 *   Typical value: 2 ticks
 *
 * Models:
 *   - Arbitration delay
 *   - Routing decision time
 *   - Serialization overhead
 *   - Protocol handshake
 * ```
 *
 * @param when Simulation tick when packet arrived at bus
 * @param pkt Memory request packet from CPUTraffic (via CPUReqEvent)
 *
 * @note Caller's callback is preserved for response routing
 * @note Transaction ID uniquely identifies this bus transaction
 * @note SRAM connection is validated with assertion
 *
 * @see BusReqEvent Request forwarding event to SRAM
 * @see MemReqPacket::visit() Visitor pattern dispatch
 * @see CPUReqEvent::process() Where this handler is invoked from
 */
void AXIBus::memReqPktHandler(Tick when, SimPacket* pkt) {
	auto memReqPkt = dynamic_cast<MemReqPacket*>(pkt);

	// callerCallback = CPUReqEvent::CPUReqCallback()
	std::function<void(int, MemRespPacket*)> callerCallback = memReqPkt->getCallback();

	SimModule* sram = this->getDownStream("PCUMem");
	CLASS_INFO << "Schedule busReqPacket with transaction id: " << transactionID << " at Tick=" << top->getGlobalTick();
	CLASS_ASSERT_MSG(sram, "Cannot find the PCUMem module!\n");

	BusReqEvent* busReqEvent = new BusReqEvent(this->transactionID++, sram, callerCallback, memReqPkt, this);
	scheduleEvent((SimEvent*)busReqEvent,
	              top->getGlobalTick() + (Tick)top->getParameter<Tick>("PETile", "bus_req_delay"));
}
