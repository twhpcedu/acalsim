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
 * @file SRAM.cc
 * @brief Static RAM memory model with configurable latency and bandwidth modeling
 *
 * This file implements the SRAM module, which serves as the local/private memory
 * component in the PE tile architecture. It demonstrates memory access latency modeling,
 * size-dependent transfer delays, and callback-based response generation for memory
 * read/write operations.
 *
 * **SRAM Role in PE Tile Memory Hierarchy:**
 * ```
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │                          SRAM Module                                   │
 * │                    (Local Memory Component)                            │
 * │                                                                        │
 * │  Request Reception & Processing:                                       │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ 1. Receive MemReqPacket from AXI Bus                             │ │
 * │  │    ├─ Via BusReqEvent → visit() → memReqPktHandler()            │ │
 * │  │    └─ Extract callback (BusReqEvent::busReqCallback)             │ │
 * │  │                                                                  │ │
 * │  │ 2. Process Memory Request                                        │ │
 * │  │    ├─ Extract request parameters (addr, size, type)              │ │
 * │  │    ├─ Calculate access latency:                                  │ │
 * │  │    │  └─ delay = sram_req_delay + (size+1)/256                  │ │
 * │  │    └─ Model memory access time (no actual data storage)         │ │
 * │  │                                                                  │ │
 * │  │ 3. Generate Response                                             │ │
 * │  │    ├─ Create SRAMRespEvent with response packet                 │ │
 * │  │    ├─ Schedule event with calculated latency                    │ │
 * │  │    └─ Event will invoke caller's callback when processed        │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * │                                                                        │
 * │  Latency Modeling:                                                     │
 * │  ┌──────────────────────────────────────────────────────────────────┐ │
 * │  │ Base Latency: sram_req_delay (from configuration)                │ │
 * │  │ Transfer Time: (size + 1) / 256 bytes/tick                       │ │
 * │  │ Total Delay: base + transfer                                     │ │
 * │  │                                                                  │ │
 * │  │ Examples (assuming sram_req_delay = 10):                         │ │
 * │  │   size=0:   10 + (0+1)/256   = 10 + 0 = 10 ticks                │ │
 * │  │   size=20:  10 + (20+1)/256  = 10 + 0 = 10 ticks                │ │
 * │  │   size=255: 10 + (255+1)/256 = 10 + 1 = 11 ticks                │ │
 * │  │   size=512: 10 + (512+1)/256 = 10 + 2 = 12 ticks                │ │
 * │  └──────────────────────────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────────────────────────────┘
 * ```
 *
 * **SRAM Memory Access Flow:**
 * ```
 * Complete Memory Access Cycle:
 * ═══════════════════════════════════════════════════════════════════
 *
 * Tick T0: BusReqEvent sends MemReqPacket to SRAM
 *   │
 *   └─> SRAM::accept(when=T0, memReqPkt)
 *       └─> MemReqPacket::visit(SRAM&)
 *           └─> SRAM::memReqPktHandler(when=T0, pkt)
 *               │
 *               ├─ Cast packet to MemReqPacket*
 *               │
 *               ├─ Extract caller's callback:
 *               │  └─ callerCallback = memReqPkt->getCallback()
 *               │     └─ This is BusReqEvent::busReqCallback
 *               │
 *               ├─ Calculate memory access latency:
 *               │  └─ delay = getDelay(memReqPkt)
 *               │     ├─ base = sram_req_delay (e.g., 10 ticks)
 *               │     ├─ transfer = (size+1) / 256
 *               │     └─ total = base + transfer
 *               │
 *               ├─ Create SRAMRespEvent:
 *               │  ├─ id = memReqPkt->getID() (transaction ID)
 *               │  ├─ callback = callerCallback
 *               │  └─ respPkt = memReqPkt->getMemRespPkt()
 *               │
 *               └─ Schedule SRAMRespEvent at T0 + delay
 *
 * Tick T0+delay: SRAMRespEvent::process() executes
 *   │
 *   └─> Invoke callerCallback(id, memRespPkt)
 *       └─> BusReqEvent::busReqCallback() executed
 *           └─> Creates BusRespEvent to route back to CPUTraffic
 * ```
 *
 * **SRAMRespEvent Class:**
 * ```
 * SRAMRespEvent : public CallbackEvent<void(int, MemRespPacket*)>
 *   │
 *   ├─ Members:
 *   │  ├─ tid (int)              : Transaction ID from request
 *   │  ├─ callerCallback (λ)     : Callback to AXI Bus
 *   │  └─ memRespPkt (MemRespPkt*) : Response packet to deliver
 *   │
 *   ├─ Methods:
 *   │  └─ process()              : Invoke callback with response
 *   │     └─ callerCallback(tid, memRespPkt)
 *   │
 *   └─ Purpose:
 *      Deliver memory response after access latency
 *      Models time required for SRAM read/write operation
 * ```
 *
 * **Latency Model Details:**
 * ```
 * SRAM Access Latency Components:
 * ════════════════════════════════════════════
 *
 * 1. Base Access Latency (sram_req_delay):
 *    - Fixed delay from configuration
 *    - Models:
 *      • Row activation time
 *      • Column selection time
 *      • Sense amplifier settling
 *      • Control logic overhead
 *    - Configured via: top->getParameter<Tick>("PETile", "sram_req_delay")
 *    - Typical value: 10 ticks
 *
 * 2. Transfer Time ((size+1)/bandwidth):
 *    - Size-dependent component
 *    - Bandwidth: 256 bytes/tick
 *    - Formula: (size + 1) / 256
 *    - +1 prevents zero delay for size=0
 *    - Models data transfer from SRAM to bus interface
 *
 * Total Latency:
 *   delay = base_latency + transfer_time
 *         = sram_req_delay + (size+1)/256
 *
 * Latency Examples:
 * ═══════════════════════════════════════════════
 * Request Size │ Transfer │ Base │ Total Delay
 * ─────────────┼──────────┼──────┼─────────────
 *      0 bytes │  0 ticks │ 10   │ 10 ticks
 *     20 bytes │  0 ticks │ 10   │ 10 ticks
 *    255 bytes │  1 tick  │ 10   │ 11 ticks
 *    256 bytes │  1 tick  │ 10   │ 11 ticks
 *    512 bytes │  2 ticks │ 10   │ 12 ticks
 *   1024 bytes │  4 ticks │ 10   │ 14 ticks
 * ```
 *
 * **Memory Request Packet Structure:**
 * ```
 * MemReqPacket (received from AXI Bus):
 * ┌─────────────────────────────────────────────┐
 * │ reqType: TENSOR_MEM_READ / TENSOR_MEM_WRITE │
 * │ addr:    Memory address (e.g., 0x0000)      │
 * │ size:    Transfer size in bytes (e.g., 20)  │
 * │ callback: BusReqEvent::busReqCallback       │
 * │ memRespPkt: Embedded response packet        │
 * │           ├─ reqType: Same as request       │
 * │           ├─ addr: Same as request          │
 * │           └─ size: Same as request          │
 * └─────────────────────────────────────────────┘
 *
 * Packet ID:
 *   - Extracted via memReqPkt->getID()
 *   - Used as transaction ID in SRAMRespEvent
 *   - Enables correlation with original request
 * ```
 *
 * **Callback Chain Integration:**
 * ```
 * SRAM's Role in Callback Chain:
 * ════════════════════════════════════════════
 *
 * Request arrives with callback:
 *   callerCallback = BusReqEvent::busReqCallback
 *     │
 *     └─ Set by AXI Bus in BusReqEvent::process()
 *
 * SRAM processes and schedules response:
 *   SRAMRespEvent(id, callerCallback, respPkt)
 *     │
 *     └─ When processed, invokes callback:
 *        └─> BusReqEvent::busReqCallback(id, module, respPkt)
 *            └─> Creates BusRespEvent
 *                └─> Delivers to CPUTraffic
 *
 * Chain Summary:
 *   CPUTraffic → AXI Bus → SRAM → AXI Bus → CPUTraffic
 *                ↑________________↑
 *                Callback wrapping enables return path
 * ```
 *
 * **Module Connectivity:**
 * ```
 * SRAM Port Configuration:
 * ═══════════════════════════════════════════════
 *
 * Upstream Connection (Response Path):
 *   - "Bus": Connected to AXI Bus
 *     └─ Receives memory requests from bus
 *     └─ Responses delivered via callbacks (not port)
 *
 * Port Setup (from PETile::registerModules()):
 *   pcuMem->addUpStream(bus, "Bus")
 *     └─ Establishes SRAM ← AXI Bus connection
 *
 * Note: Responses use callback mechanism, not port-based send
 * ```
 *
 * **Memory Model Abstraction:**
 * ```
 * Current Implementation:
 * ═══════════════════════════════════════════════
 *   - Timing model only (no actual data storage)
 *   - Models access latencies and bandwidth
 *   - Response packet passed through unchanged
 *   - Focus on performance simulation
 *
 * Extension Opportunities:
 * ═══════════════════════════════════════════════
 *   1. Add actual data storage:
 *      - std::vector<uint8_t> memory(capacity)
 *      - Read/write to memory array
 *
 *   2. Add memory address validation:
 *      - Check addr < capacity
 *      - Assert on out-of-bounds access
 *
 *   3. Add bank modeling:
 *      - Multiple banks for parallel access
 *      - Bank conflict detection
 *      - Bank activation tracking
 *
 *   4. Add energy modeling:
 *      - Track access counts
 *      - Calculate energy per access
 *      - Bank-specific energy states
 *
 *   5. Add error injection:
 *      - Bit flips for reliability studies
 *      - Stuck-at faults
 *      - Retention errors
 * ```
 *
 * **Performance Characteristics:**
 * ```
 * SRAM vs Other Memory Types:
 * ═══════════════════════════════════════════════
 *
 * SRAM (This Model):
 *   ✓ Fast access (10 ticks base)
 *   ✓ No refresh required
 *   ✓ Static storage (retains data)
 *   ✓ Suitable for on-chip caches/scratchpads
 *   - Higher area cost per bit
 *   - Lower density than DRAM
 *
 * Comparison to DRAM:
 *   - SRAM: ~10 ticks access
 *   - DRAM: ~50-100 ticks (with row activation)
 *   - SRAM: No refresh overhead
 *   - DRAM: Periodic refresh required
 *
 * Use Cases in PE Tile:
 *   - Local scratchpad memory
 *   - Register file
 *   - L1/L2 cache backend
 *   - Buffer storage
 * ```
 *
 * **Visitor Pattern Integration:**
 * ```
 * Packet Dispatch Mechanism:
 * ═══════════════════════════════════════════════
 *
 * BusReqEvent::process() calls:
 *   SRAM::accept(when, memReqPkt)
 *     │
 *     └─> SimModule::accept() (base class)
 *         └─> memReqPkt.visit(when, *this)
 *             │
 *             └─> MemReqPacket::visit(Tick, SimModule&)
 *                 │
 *                 └─> if (dynamic_cast<SRAM*>(&module))
 *                     └─> sram->memReqPktHandler(when, this)
 *
 * Benefits:
 *   - Type-safe packet routing
 *   - No manual dynamic_cast in accept()
 *   - Extensible for new packet types
 *   - Compile-time type checking
 * ```
 *
 * **Error Handling:**
 * ```
 * Callback Validation:
 *   if (callerCallback) { ... }
 *     - Checks if callback is valid before use
 *     - Prevents calling null function pointer
 *     - Graceful degradation if no response needed
 *
 * Typical Flow:
 *   - Callback always provided by BusReqEvent
 *   - Check is defensive programming
 *   - In normal operation, never null
 * ```
 *
 * **Configuration Parameters:**
 * ```
 * SRAM Configuration (from configs.json):
 * ═══════════════════════════════════════════════
 *
 * Direct Parameters:
 *   sram_req_delay: 10 ticks
 *     └─ Base memory access latency
 *
 * Implicit Parameters:
 *   Memory bandwidth: 256 bytes/tick
 *     └─ Hardcoded in getDelay() formula
 *     └─ Could be extracted to configuration
 *
 * Related Cache Parameters:
 *   cache_struct.mem_size: 32768 bytes
 *     └─ If SRAM models cache backend
 *   cache_struct.associativity: 4
 *     └─ For set-associative access patterns
 * ```
 *
 * **Design Patterns:**
 *
 * 1. **Visitor Pattern:**
 *    - MemReqPacket::visit(SRAM&) dispatches to memReqPktHandler()
 *    - Type-safe packet processing
 *    - No explicit type checking in handler
 *
 * 2. **Callback Event Pattern:**
 *    - SRAMRespEvent encapsulates delayed callback invocation
 *    - Automatic scheduling via event queue
 *    - Clean separation of timing and logic
 *
 * 3. **Latency Abstraction:**
 *    - getDelay() centralizes latency calculation
 *    - Easy to modify latency model
 *    - Consistent across all accesses
 *
 * **Related Files:**
 * - Header: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/SRAM.hh
 * - AXI Bus: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/AXIBus.cc
 * - Mem Packets: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/MemReq.cc
 * - Config: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/include/PETileConfig.hh
 * - PETile: /Users/weifen/work/acal/acalsim-workspace/projects/acalsim/src/testPETile/libs/PETile.cc
 *
 * @author ACALSim Framework Team
 * @date 2023-2025
 */

#include "SRAM.hh"

/**
 * @brief Handles incoming memory request packets from AXI Bus
 *
 * This is the main entry point for memory requests arriving at the SRAM. It extracts
 * the request parameters, calculates the memory access latency, and schedules a response
 * event that will invoke the caller's callback after the appropriate delay.
 *
 * **Memory Request Processing Pipeline:**
 * ```
 * BusReqEvent → MemReqPacket → visit(SRAM) → memReqPktHandler()
 *   │
 *   ├─ Cast packet to MemReqPacket*
 *   │
 *   ├─ Extract caller's callback:
 *   │  └─ callerCallback = memReqPkt->getCallback()
 *   │     └─ This is BusReqEvent::busReqCallback
 *   │
 *   ├─ Validate callback exists (defensive check)
 *   │
 *   ├─ Extract transaction ID:
 *   │  └─ id = memReqPkt->getID()
 *   │
 *   ├─ Extract embedded response packet:
 *   │  └─ respPkt = memReqPkt->getMemRespPkt()
 *   │
 *   ├─ Calculate memory access latency:
 *   │  └─ delay = getDelay(memReqPkt)
 *   │     ├─ base = sram_req_delay (e.g., 10 ticks)
 *   │     ├─ transfer = (size+1) / 256
 *   │     └─ total = base + transfer
 *   │
 *   ├─ Create SRAMRespEvent:
 *   │  ├─ id: Transaction identifier
 *   │  ├─ callback: Caller's callback function
 *   │  └─ respPkt: Response packet to deliver
 *   │
 *   └─ Schedule SRAMRespEvent at when + delay
 * ```
 *
 * **Latency Calculation:**
 * ```
 * getDelay(memReqPkt) returns:
 *   sram_req_delay + (pkt->getSize() + 1) / 256
 *
 * Example with sram_req_delay = 10:
 *   size=0:   10 + (0+1)/256   = 10 + 0 = 10 ticks
 *   size=20:  10 + (20+1)/256  = 10 + 0 = 10 ticks
 *   size=255: 10 + (255+1)/256 = 10 + 1 = 11 ticks
 *   size=512: 10 + (512+1)/256 = 10 + 2 = 12 ticks
 *
 * Components:
 *   - Base delay models SRAM access time (row/column decode, sense amp)
 *   - Transfer delay models data movement bandwidth (256 bytes/tick)
 * ```
 *
 * **Response Event Scheduling:**
 * ```
 * SRAMRespEvent Creation:
 *   SRAMRespEvent(id, callerCallback, memRespPkt)
 *     │
 *     ├─ id: Transaction ID for correlation
 *     │     └─ From memReqPkt->getID()
 *     │
 *     ├─ callerCallback: Function to invoke when ready
 *     │     └─ BusReqEvent::busReqCallback
 *     │
 *     └─ memRespPkt: Response packet to deliver
 *           └─ From memReqPkt->getMemRespPkt()
 *
 * Scheduled At:
 *   when + getDelay(memReqPkt)
 *     │
 *     ├─ 'when': Tick request arrived at SRAM
 *     └─ 'delay': Calculated access + transfer latency
 *
 * When Event Fires:
 *   SRAMRespEvent::process() executes:
 *     └─> callerCallback(id, memRespPkt)
 *         └─> BusReqEvent::busReqCallback() invoked
 * ```
 *
 * **Memory Access Timing:**
 * ```
 * Example Timeline (sram_req_delay=10, size=20):
 *
 * Tick T0: Request arrives at SRAM
 *   memReqPktHandler(when=T0, pkt)
 *     ├─ Extract parameters
 *     ├─ Calculate delay = 10 + 0 = 10 ticks
 *     └─ Schedule SRAMRespEvent at T0+10
 *
 * Tick T0+10: SRAMRespEvent::process() executes
 *   └─> callerCallback(id, respPkt)
 *       └─> BusReqEvent::busReqCallback()
 *           └─> Creates BusRespEvent
 *               └─> Eventually reaches CPUTraffic
 * ```
 *
 * @param when Simulation tick when request arrived at SRAM (from BusReqEvent)
 * @param pkt Memory request packet from AXI Bus (via BusReqEvent)
 *
 * @note Response is delivered via callback, not direct packet send
 * @note Timing model only - no actual data storage/retrieval
 * @note Callback is always BusReqEvent::busReqCallback in current architecture
 *
 * @see SRAMRespEvent Response event that invokes callback
 * @see SRAM::getDelay() Latency calculation method (in header)
 * @see BusReqEvent::busReqCallback() Callback invoked by response event
 * @see MemReqPacket::visit() Visitor pattern dispatch mechanism
 */
void SRAM::memReqPktHandler(Tick when, SimPacket* pkt) {
	auto                                     memReqPkt      = dynamic_cast<MemReqPacket*>(pkt);
	std::function<void(int, MemRespPacket*)> callerCallback = memReqPkt->getCallback();
	if (callerCallback) {  // callerCallback = BusReqEvent::busReqCallback()
		SRAMRespEvent* sramRespEvent =
		    new SRAMRespEvent(memReqPkt->getID(), callerCallback, memReqPkt->getMemRespPkt());
		scheduleEvent((SimEvent*)sramRespEvent, when + this->getDelay(memReqPkt));
	}
}
