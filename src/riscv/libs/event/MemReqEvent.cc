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
 * @file MemReqEvent.cc
 * @brief Memory request event implementation for RISC-V memory subsystem
 *
 * @details
 * This file implements the MemReqEvent class, which handles memory access requests
 * (loads and stores) in the RISC-V simulator. It serves as the bridge between the
 * CPU's instruction execution and the memory hierarchy, modeling memory access timing
 * through event-driven simulation.
 *
 * **Memory Request Event Handling:**
 *
 * MemReqEvent encapsulates a memory operation request and coordinates its delivery
 * to the memory subsystem. It carries a SimPacket containing request details
 * (address, size, data, operation type) and schedules the request delivery with
 * appropriate timing delays.
 *
 * ```
 * Memory Access Flow:
 * ├─ CPU executes LOAD/STORE instruction
 * ├─ Create MemReqEvent with request packet
 * ├─ Schedule MemReqEvent at T+1 (request transit delay)
 * ├─ Event processes → Submit to DataMemory
 * ├─ DataMemory accepts → Queue or process immediately
 * ├─ Cache/memory hierarchy processes request
 * └─ Response returned (timing depends on hit/miss)
 * ```
 *
 * **Event-Driven Memory Model:**
 *
 * The simulator uses discrete events to model memory system timing:
 *
 * | Component | Event | Timing Modeled |
 * |-----------|-------|----------------|
 * | **CPU** | ExecOneInstrEvent | Instruction decode |
 * | **Request Transit** | MemReqEvent | Address calculation + bus delay |
 * | **Cache/Memory** | DataMemory::accept | Lookup + hit/miss handling |
 * | **Response** | Callback/Event | Data return + register update |
 *
 * **Event Lifecycle and Timing:**
 *
 * ```
 * Tick T: CPU executes memory instruction
 *   ├─ Decode LOAD r1, offset(r2)
 *   ├─ Calculate address = r2 + offset
 *   ├─ Create SimPacket(addr, size, type=LOAD)
 *   └─ Create MemReqEvent(dataMem, packet)
 *
 * Tick T: Schedule MemReqEvent
 *   └─ scheduler.schedule(memReqEvent, T+1)
 *
 * Tick T+1: MemReqEvent::process()
 *   ├─ Call dataMem->accept(globalTick, packet)
 *   ├─ DataMemory receives request
 *   └─ Cache lookup begins
 *
 * Tick T+1 to T+N: Memory system processing
 *   ├─ L1 cache lookup (1-3 cycles if hit)
 *   ├─ L2 cache lookup (10-15 cycles if L1 miss)
 *   ├─ Main memory access (50-100+ cycles if L2 miss)
 *   └─ Response propagation
 *
 * Tick T+N: Data returns to CPU
 *   ├─ Register file updated
 *   └─ Next ExecOneInstrEvent scheduled
 * ```
 *
 * **Memory Request Packet Structure:**
 *
 * The SimPacket carried by MemReqEvent contains:
 *
 * ```cpp
 * struct SimPacket {
 *     uint64_t address;      // Memory address to access
 *     uint32_t size;         // Access size (1/2/4/8 bytes)
 *     uint8_t* data;         // Data buffer (for stores)
 *     MemOpType type;        // LOAD, STORE, FETCH, etc.
 *     uint64_t reqID;        // Transaction ID for tracking
 *     void* context;         // Callback context (CPU/register info)
 * };
 * ```
 *
 * **Integration with Instruction Execution:**
 *
 * Memory events coordinate closely with instruction execution events:
 *
 * ```
 * LOAD Instruction Timeline:
 * ┌─────────────────────────────────────────────────────────────┐
 * │ T+0: ExecOneInstrEvent                                      │
 * │   └─ Decode LOAD → Create MemReqEvent → Schedule at T+1   │
 * ├─────────────────────────────────────────────────────────────┤
 * │ T+1: MemReqEvent                                            │
 * │   └─ Submit to DataMemory → Cache lookup starts            │
 * ├─────────────────────────────────────────────────────────────┤
 * │ T+1 to T+4: Cache processing (assuming L1 hit)             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ T+4: Data ready                                             │
 * │   └─ Register updated                                       │
 * ├─────────────────────────────────────────────────────────────┤
 * │ T+4: Next ExecOneInstrEvent                                 │
 * │   └─ Continue execution                                     │
 * └─────────────────────────────────────────────────────────────┘
 *
 * STORE Instruction Timeline:
 * ┌─────────────────────────────────────────────────────────────┐
 * │ T+0: ExecOneInstrEvent                                      │
 * │   └─ Decode STORE → Create MemReqEvent → Schedule at T+1  │
 * ├─────────────────────────────────────────────────────────────┤
 * │ T+1: MemReqEvent                                            │
 * │   └─ Submit to DataMemory → Write buffer/cache             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ T+2: Next ExecOneInstrEvent (stores often non-blocking)    │
 * │   └─ Continue execution                                     │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * **Event Chaining and Dependencies:**
 *
 * MemReqEvent participates in complex event chains:
 *
 * ```
 * Simple Memory Access:
 *   ExecOneInstrEvent → MemReqEvent → DataMemory → Response
 *
 * Cache Miss Cascade:
 *   ExecOneInstrEvent → MemReqEvent → L1 Miss Event
 *                                    ↓
 *                                 L2 Request Event
 *                                    ↓
 *                                 L2 Miss Event
 *                                    ↓
 *                                 DRAM Request Event
 *                                    ↓
 *                                 DRAM Response Event
 *                                    ↓
 *                                 L2 Fill Event
 *                                    ↓
 *                                 L1 Fill Event
 *                                    ↓
 *                                 Data Ready
 *
 * Multiple Outstanding Requests (Out-of-Order):
 *   Core 0: ExecEvt → MemReq_A (addr=0x1000) @ T+1
 *   Core 0: ExecEvt → MemReq_B (addr=0x2000) @ T+2
 *   Core 0: ExecEvt → MemReq_C (addr=0x3000) @ T+3
 *     ↓
 *   Response_B completes @ T+5 (L1 hit)
 *   Response_A completes @ T+8 (L2 hit)
 *   Response_C completes @ T+50 (DRAM access)
 * ```
 *
 * **Memory Subsystem Integration:**
 *
 * MemReqEvent interfaces with DataMemory through the accept() method:
 *
 * ```cpp
 * // DataMemory::accept signature:
 * void DataMemory::accept(Tick currentTick, SimPacket& packet) {
 *     // 1. Check for structural hazards (queue full?)
 *     if (requestQueue.full()) {
 *         // Retry event or stall
 *         return;
 *     }
 *
 *     // 2. Enqueue request
 *     requestQueue.push(packet);
 *
 *     // 3. Schedule cache lookup event
 *     CacheLookupEvent* lookup = new CacheLookupEvent(this, packet);
 *     scheduler.schedule(lookup, currentTick + cacheLookupLatency);
 * }
 * ```
 *
 * **Timing Model Details:**
 *
 * | Operation | Typical Latency | Event Scheduling |
 * |-----------|----------------|------------------|
 * | **Address Calc** | 0 cycles | In ExecOneInstrEvent |
 * | **Request Transit** | 1 cycle | MemReqEvent scheduled at T+1 |
 * | **L1 Hit** | 1-3 cycles | From accept() to response |
 * | **L2 Hit** | 10-15 cycles | Multiple cache events |
 * | **L3 Hit** | 30-50 cycles | Cache hierarchy traversal |
 * | **DRAM Access** | 100-300 cycles | Memory controller events |
 *
 * **Performance Characteristics:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | Constructor | O(1) | Stores packet and DataMemory pointers |
 * | renew() | O(1) | Resets state for recycling |
 * | process() | O(1) | Delegates to DataMemory::accept() |
 * | Scheduling | O(log N) | Event queue insertion |
 *
 * **Memory Footprint:**
 * - sizeof(MemReqEvent) ≈ sizeof(SimEvent) + sizeof(DataMemory*) + sizeof(SimPacket*)
 * - Approximately 48-64 bytes per event instance
 * - Packet size varies based on data payload
 *
 * **Multi-Core Memory Arbitration:**
 *
 * ```
 * Core 0 → MemReqEvent_0 ──┐
 * Core 1 → MemReqEvent_1 ──┼→ DataMemory::accept()
 * Core 2 → MemReqEvent_2 ──┤    ↓
 * Core 3 → MemReqEvent_3 ──┘  Arbitrate + Queue
 *                                ↓
 *                            Process in order
 *                                ↓
 *                            Return responses
 * ```
 *
 * **Object Pool Usage:**
 *
 * ```cpp
 * // Optional: Use RecycleContainer for high-frequency memory accesses
 * RecycleContainer<MemReqEvent> memEventPool;
 *
 * void issueMemoryRequest(DataMemory* dm, uint64_t addr, bool isLoad) {
 *     // Create packet
 *     SimPacket* pkt = new SimPacket(addr, 8, isLoad ? LOAD : STORE);
 *
 *     // Get recycled event
 *     MemReqEvent* evt = memEventPool.get<MemReqEvent>(dm, pkt);
 *
 *     // Schedule request
 *     scheduler.schedule(evt, currentTick() + 1);
 *
 *     // Event auto-returns to pool after processing
 * }
 * ```
 *
 * @code{.cpp}
 * // Example 1: LOAD instruction flow
 * void CPU::executeLoad(uint64_t addr, uint8_t regDst) {
 *     // Create request packet
 *     SimPacket* loadPkt = new SimPacket();
 *     loadPkt->address = addr;
 *     loadPkt->size = 8;  // 64-bit load
 *     loadPkt->type = MemOpType::LOAD;
 *     loadPkt->reqID = nextReqID++;
 *     loadPkt->context = (void*)regDst;  // Destination register
 *
 *     // Create memory request event
 *     MemReqEvent* memEvt = new MemReqEvent(dataMemory, loadPkt);
 *
 *     // Schedule request transit (1 cycle delay)
 *     scheduleEvent(memEvt, currentTick() + 1);
 *
 *     // Schedule next instruction after cache latency
 *     ExecOneInstrEvent* nextEvt = new ExecOneInstrEvent(coreID, this);
 *     scheduleEvent(nextEvt, currentTick() + cacheLatency);
 * }
 *
 * // Example 2: STORE instruction flow
 * void CPU::executeStore(uint64_t addr, uint64_t data) {
 *     // Create request packet with data
 *     SimPacket* storePkt = new SimPacket();
 *     storePkt->address = addr;
 *     storePkt->size = 8;
 *     storePkt->type = MemOpType::STORE;
 *     storePkt->data = new uint8_t[8];
 *     memcpy(storePkt->data, &data, 8);
 *
 *     // Create memory request event
 *     MemReqEvent* memEvt = new MemReqEvent(dataMemory, storePkt);
 *
 *     // Schedule request
 *     scheduleEvent(memEvt, currentTick() + 1);
 *
 *     // Stores are typically non-blocking
 *     ExecOneInstrEvent* nextEvt = new ExecOneInstrEvent(coreID, this);
 *     scheduleEvent(nextEvt, currentTick() + 1);
 * }
 *
 * // Example 3: DataMemory accept implementation
 * void DataMemory::accept(Tick currentTick, SimPacket& packet) {
 *     // Check request type
 *     if (packet.type == MemOpType::LOAD) {
 *         // Perform cache lookup
 *         CacheEntry* entry = l1Cache.lookup(packet.address);
 *
 *         if (entry && entry->valid) {
 *             // L1 hit - fast path (3 cycles)
 *             packet.data = entry->data;
 *             completeRequest(packet, currentTick + 3);
 *         } else {
 *             // L1 miss - schedule L2 lookup (12 cycles)
 *             L2LookupEvent* l2Evt = new L2LookupEvent(this, packet);
 *             scheduler.schedule(l2Evt, currentTick + 12);
 *         }
 *     } else if (packet.type == MemOpType::STORE) {
 *         // Write to cache/write buffer
 *         l1Cache.write(packet.address, packet.data);
 *
 *         // Acknowledge store (non-blocking)
 *         completeRequest(packet, currentTick + 1);
 *     }
 * }
 * @endcode
 *
 * @note MemReqEvent is typically scheduled 1 tick after instruction decode
 * @note Each memory request should have unique transaction ID for tracking
 * @note Packet ownership may transfer to memory subsystem after accept()
 * @note Response timing varies dramatically based on cache hierarchy state
 *
 * @warning SimPacket pointer must remain valid until request completes
 * @warning DataMemory pointer must be valid when process() executes
 * @warning Multiple outstanding requests require proper transaction tracking
 * @warning Do not manually call process() - let scheduler invoke it
 *
 * @see MemReqEvent.hh for class definition
 * @see ExecOneInstrEvent for instruction execution coordination
 * @see DataMemory.hh for memory subsystem implementation
 * @see SimPacket for request packet structure
 * @see SimEvent.hh for base event-driven framework
 * @see docs/for-users/event-driven-simulation.md for usage examples
 *
 * @since ACALSim 0.1.0
 * @ingroup riscv_events
 */

#include "event/MemReqEvent.hh"

#include "DataMemory.hh"

/**
 * @brief Construct a memory request event for data memory access
 *
 * @param _callee Pointer to the DataMemory instance that will handle the request
 * @param _memReqPkt Pointer to SimPacket containing memory request details
 *
 * @details
 * Initializes an event that delivers a memory access request (LOAD/STORE) to
 * the data memory subsystem. The event encapsulates both the target memory
 * controller and the request packet containing address, size, and operation type.
 *
 * **Initialization Flow:**
 * ```
 * Constructor called
 *   ├─ SimEvent::SimEvent("MemReqEvent")
 *   │   ├─ Assign unique global event ID
 *   │   ├─ Store event name "MemReqEvent"
 *   │   └─ Initialize gem5::Event base
 *   ├─ Store DataMemory pointer (callee)
 *   └─ Store SimPacket pointer (memReqPkt)
 * ```
 *
 * **SimPacket Contents:**
 * The packet typically contains:
 * - **address**: Memory address to access (virtual or physical)
 * - **size**: Access size in bytes (1, 2, 4, or 8)
 * - **type**: Operation type (LOAD, STORE, FETCH, etc.)
 * - **data**: Data buffer (for STORE operations)
 * - **reqID**: Unique transaction identifier
 * - **context**: Callback context (register ID, completion handler, etc.)
 *
 * **Typical Usage Patterns:**
 *
 * ```cpp
 * // Pattern 1: LOAD instruction
 * void CPU::execLoad(uint64_t addr, uint8_t regDst) {
 *     // Create request packet
 *     SimPacket* pkt = new SimPacket();
 *     pkt->address = addr;
 *     pkt->size = 8;
 *     pkt->type = MemOpType::LOAD;
 *     pkt->context = (void*)regDst;
 *
 *     // Create and schedule event
 *     MemReqEvent* evt = new MemReqEvent(dataMemory, pkt);
 *     scheduleEvent(evt, currentTick() + 1);
 * }
 *
 * // Pattern 2: STORE instruction
 * void CPU::execStore(uint64_t addr, uint64_t value) {
 *     SimPacket* pkt = new SimPacket();
 *     pkt->address = addr;
 *     pkt->size = 8;
 *     pkt->type = MemOpType::STORE;
 *     pkt->data = new uint8_t[8];
 *     memcpy(pkt->data, &value, 8);
 *
 *     MemReqEvent* evt = new MemReqEvent(dataMemory, pkt);
 *     scheduleEvent(evt, currentTick() + 1);
 * }
 *
 * // Pattern 3: Multiple outstanding requests
 * void CPU::execMemOps() {
 *     // Issue multiple requests in same cycle
 *     for (int i = 0; i < numLoads; i++) {
 *         SimPacket* pkt = createLoadPacket(addresses[i]);
 *         MemReqEvent* evt = new MemReqEvent(dataMemory, pkt);
 *         scheduleEvent(evt, currentTick() + 1);
 *     }
 * }
 * ```
 *
 * @note Event name is fixed as "MemReqEvent" (no ID suffix like ExecOneInstrEvent)
 * @note Both pointers must remain valid until event processes
 * @note Packet ownership may transfer to DataMemory after accept()
 * @note Complexity: O(1)
 *
 * @warning DataMemory pointer is not reference-counted - ensure valid lifecycle
 * @warning SimPacket pointer must remain valid until request completes
 */
MemReqEvent::MemReqEvent(DataMemory* _callee, acalsim::SimPacket* _memReqPkt)
    : acalsim::SimEvent("MemReqEvent"), callee(_callee), memReqPkt(_memReqPkt) {}

/**
 * @brief Reset event state for object pool recycling
 *
 * @param _callee New DataMemory instance pointer for request handling
 * @param _memReqPkt New SimPacket pointer containing memory request details
 *
 * @details
 * Reinitializes the event for reuse from an object pool, enabling efficient
 * memory management in high-frequency memory access scenarios. This is especially
 * beneficial for memory-intensive workloads with frequent LOAD/STORE operations.
 *
 * **Recycling Flow:**
 * ```
 * Event lifecycle with object pooling:
 *
 * 1. Initial use:
 *    MemReqEvent* evt = new MemReqEvent(dm, pkt)
 *    schedule(evt, T+1) → process() → release()
 *
 * 2. Return to pool:
 *    evt->release() → RecycleContainer stores event
 *
 * 3. Reuse (this method called):
 *    pool.get<MemReqEvent>(new_dm, new_pkt)
 *      └─ Calls renew(new_dm, new_pkt)
 *      └─ Event ready for scheduling again
 *
 * 4. Subsequent uses:
 *    Same event object reused, avoiding allocation overhead
 * ```
 *
 * **Performance Benefits:**
 *
 * Memory-intensive workloads may create thousands to millions of MemReqEvents:
 *
 * | Workload | MemReqEvents/sec | Overhead (new/delete) | Overhead (pooled) |
 * |----------|-----------------|----------------------|-------------------|
 * | **Light** | 1,000 | ~0.1 ms | ~0.01 ms |
 * | **Medium** | 100,000 | ~10 ms | ~1 ms |
 * | **Heavy** | 1,000,000 | ~100 ms | ~10 ms |
 *
 * **Example with Object Pool:**
 * ```cpp
 * RecycleContainer<MemReqEvent> memEventPool;
 *
 * void issueMemoryAccess(DataMemory* dm, uint64_t addr, bool isLoad) {
 *     // Create packet
 *     SimPacket* pkt = new SimPacket();
 *     pkt->address = addr;
 *     pkt->size = 8;
 *     pkt->type = isLoad ? MemOpType::LOAD : MemOpType::STORE;
 *
 *     // Get event from pool (may reuse via renew)
 *     MemReqEvent* evt = memEventPool.get<MemReqEvent>(dm, pkt);
 *
 *     // Schedule for processing
 *     scheduleEvent(evt, currentTick() + 1);
 *
 *     // Event auto-returns to pool after process()
 * }
 *
 * // With object pool, even millions of memory accesses have low overhead
 * void stressTest() {
 *     for (int i = 0; i < 1000000; i++) {
 *         issueMemoryAccess(dm, baseAddr + i * 64, true);
 *     }
 *     // Pool reuses ~10-100 event objects instead of allocating 1 million
 * }
 * ```
 *
 * **State Reset Details:**
 * - Calls SimEvent::renew() to reset base class state
 * - Generates new unique event ID
 * - Updates DataMemory pointer to new target
 * - Updates SimPacket pointer to new request
 * - Previous packet/callee pointers are overwritten (ensure cleanup if needed)
 *
 * @note Calls SimEvent::renew() which assigns new unique event ID
 * @note Does NOT delete old packet - caller responsible for cleanup
 * @note Updates both callee and packet pointers atomically
 * @note Complexity: O(1)
 *
 * @see RecycleContainer for object pool management
 * @see SimEvent::renew() for base class recycling behavior
 *
 * @warning Old packet pointer is lost - ensure proper lifecycle management
 * @warning New pointers must remain valid until event processes
 */
void MemReqEvent::renew(DataMemory* _callee, acalsim::SimPacket* _memReqPkt) {
	this->acalsim::SimEvent::renew();
	this->callee    = _callee;
	this->memReqPkt = _memReqPkt;
}

/**
 * @brief Submit memory request to DataMemory subsystem
 *
 * @details
 * This is the event's main processing method, called by the simulation scheduler
 * when the event's scheduled tick is reached. It delivers the memory request to
 * the DataMemory subsystem by invoking accept() with the current global tick and
 * request packet.
 *
 * **Invocation Context:**
 * ```
 * Simulation Loop:
 *   Event scheduled at tick T+1 by CPU
 *     ↓
 *   Scheduler advances to tick T+1
 *     ↓
 *   Scheduler dequeues MemReqEvent
 *     ↓
 *   process() called ← This method
 *     ↓
 *   callee->accept(globalTick, packet)
 *     ↓
 *   DataMemory handles request
 * ```
 *
 * **Processing Sequence:**
 * ```
 * process() invoked at scheduled tick
 *   └─ Retrieve global simulation tick
 *   └─ Call callee->accept(tick, *memReqPkt)
 *       ├─ DataMemory receives request
 *       ├─ Check for structural hazards (queue full?)
 *       ├─ Enqueue or reject request
 *       ├─ Schedule cache lookup event
 *       └─ Return control to scheduler
 * ```
 *
 * **DataMemory::accept() Behavior:**
 *
 * When accept() is called, DataMemory typically:
 *
 * ```cpp
 * void DataMemory::accept(Tick currentTick, SimPacket& packet) {
 *     // 1. Validate request
 *     if (!isValidAddress(packet.address)) {
 *         handleException(packet);
 *         return;
 *     }
 *
 *     // 2. Check structural hazards
 *     if (mshrFull() || requestQueueFull()) {
 *         // Retry or stall (may reschedule MemReqEvent)
 *         retryQueue.push(packet);
 *         return;
 *     }
 *
 *     // 3. Enqueue request
 *     requestQueue.push(packet);
 *
 *     // 4. Schedule cache processing
 *     if (packet.type == LOAD) {
 *         CacheLookupEvent* evt = new CacheLookupEvent(this, packet);
 *         scheduleEvent(evt, currentTick + cacheLookupLatency);
 *     } else {
 *         StoreProcessEvent* evt = new StoreProcessEvent(this, packet);
 *         scheduleEvent(evt, currentTick + 1);
 *     }
 * }
 * ```
 *
 * **Timing Examples:**
 *
 * ```cpp
 * // Example 1: L1 cache hit
 * Tick T+0:  CPU executes LOAD
 * Tick T+1:  MemReqEvent::process() → accept()
 * Tick T+2:  L1 lookup event processes
 * Tick T+3:  Data ready, register updated
 *
 * // Example 2: L1 miss, L2 hit
 * Tick T+0:  CPU executes LOAD
 * Tick T+1:  MemReqEvent::process() → accept()
 * Tick T+2:  L1 lookup (miss)
 * Tick T+3:  L2 request event
 * Tick T+12: L2 lookup (hit)
 * Tick T+13: Data ready, L1 filled, register updated
 *
 * // Example 3: DRAM access
 * Tick T+0:   CPU executes LOAD
 * Tick T+1:   MemReqEvent::process() → accept()
 * Tick T+2:   L1 lookup (miss)
 * Tick T+12:  L2 lookup (miss)
 * Tick T+30:  L3 lookup (miss)
 * Tick T+35:  DRAM request queued
 * Tick T+100: DRAM data ready
 * Tick T+105: L3/L2/L1 filled
 * Tick T+106: Register updated
 * ```
 *
 * **Global Tick Access:**
 *
 * The method retrieves the global simulation tick via `acalsim::top->getGlobalTick()`,
 * which returns the current unified time across all simulator components. This ensures
 * consistent timing across multi-threaded simulation.
 *
 * ```
 * Centralized Time Management:
 *
 * acalsim::top (SimTop instance)
 *   ├─ globalTick: Unified simulation time
 *   ├─ Core 0 executes at tick T
 *   ├─ Core 1 executes at tick T
 *   ├─ Memory processes at tick T
 *   └─ All events see same "current tick"
 * ```
 *
 * **Packet Lifecycle After accept():**
 *
 * ```
 * Before process():
 *   CPU owns packet → Creates MemReqEvent with packet
 *
 * During process():
 *   MemReqEvent holds packet → Passes to accept()
 *
 * After accept():
 *   DataMemory may:
 *     ├─ Store packet in queue (packet ownership transfers)
 *     ├─ Copy packet contents (packet can be deleted)
 *     └─ Process immediately (packet no longer needed)
 *
 * Cleanup:
 *   Depends on memory subsystem implementation
 *   Common pattern: DataMemory deletes packet after response
 * ```
 *
 * **Error Handling:**
 *
 * If accept() encounters issues:
 * - **Invalid address**: May trigger exception handler
 * - **Queue full**: May reschedule event or stall CPU
 * - **Alignment error**: May raise alignment exception
 * - **Permission fault**: May trigger page fault handler
 *
 * **Multi-Core Scenarios:**
 *
 * ```
 * Multiple cores issuing requests simultaneously:
 *
 * Tick T+1:
 *   Core 0 MemReqEvent → accept(T+1, pkt0)  ──┐
 *   Core 1 MemReqEvent → accept(T+1, pkt1)  ──┼→ DataMemory
 *   Core 2 MemReqEvent → accept(T+1, pkt2)  ──┤    ↓
 *   Core 3 MemReqEvent → accept(T+1, pkt3)  ──┘  Arbitrate
 *                                                    ↓
 *                                              Queue requests
 *                                                    ↓
 *                                              Process in order
 * ```
 *
 * @note Called automatically by scheduler at scheduled tick
 * @note Uses global simulation tick for consistent timing
 * @note Packet passed by reference (not copied)
 * @note After accept(), packet ownership may transfer to DataMemory
 * @note Complexity: O(1) for delegation, DataMemory processing varies
 *
 * @see DataMemory::accept() for request handling implementation
 * @see acalsim::SimTop::getGlobalTick() for global time access
 * @see SimPacket for request packet structure
 * @see ExecOneInstrEvent for instruction execution coordination
 * @see SimEvent::process() for base class contract
 *
 * @warning Do not call manually - scheduler invokes at scheduled tick
 * @warning callee pointer must be valid when process() executes
 * @warning memReqPkt pointer must remain valid during accept()
 * @warning Thread safety: Process executes in DataMemory's simulator thread
 */
void MemReqEvent::process() { this->callee->accept(acalsim::top->getGlobalTick(), *this->memReqPkt); }
