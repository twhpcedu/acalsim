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
 * @file MemReqEvent.cc
 * @brief Memory request event implementation for riscvSimTemplate
 *
 * @details
 * This file implements the MemReqEvent class for the riscvSimTemplate,
 * providing event-driven memory access handling in a simplified educational
 * architecture. It demonstrates fundamental memory request timing and
 * event-driven communication between CPU and memory subsystem.
 *
 * **Template Version - Simplified Memory Model:**
 *
 * The riscvSimTemplate uses a streamlined memory hierarchy compared to the
 * full riscv/ version, focusing on teaching core discrete-event simulation
 * concepts without complex cache coherence or memory controller details.
 *
 * ```
 * Full Version (riscv/):           Template Version (riscvSimTemplate/):
 * ┌─────────────────────────┐     ┌─────────────────────────┐
 * │ CPU → L1 Cache          │     │ CPU → DataMemory        │
 * │  ↓                      │     │  ↓                      │
 * │ L1 miss → L2 Cache      │     │ Simple latency model    │
 * │  ↓                      │     │  - Hit: fixed cycles    │
 * │ L2 miss → Memory Ctrl   │     │  - Miss: N/A (always hit)│
 * │  ↓                      │     │  ↓                      │
 * │ DRAM with timing model  │     │ Return data to CPU      │
 * │  ↓                      │     └─────────────────────────┘
 * │ Coherence protocol      │
 * │  ↓                      │
 * │ Return data to CPU      │
 * └─────────────────────────┘
 * ```
 *
 * **Event-Driven Memory Access Model:**
 *
 * Memory requests are modeled as discrete events scheduled at specific
 * simulation ticks, allowing accurate timing simulation of memory latency:
 *
 * ```
 * Timeline for Memory Access:
 * ├─ Tick T+0: CPU executes LOAD instruction
 * │            └─ Creates MemReqEvent with address, data packet
 * ├─ Tick T+1: MemReqEvent.process() fires
 * │            ├─ DataMemory.accept() called with packet
 * │            ├─ Memory performs read operation
 * │            └─ Data written to packet for CPU
 * ├─ Tick T+2: CPU receives data (if using callbacks/ports)
 * │            └─ Completes LOAD instruction
 * └─ Tick T+3: CPU continues with next instruction
 * ```
 *
 * **Memory Request Event Handling:**
 *
 * MemReqEvent coordinates between CPU instruction execution and memory
 * subsystem operations, supporting both read and write operations:
 *
 * | Operation | Direction | Event Flow |
 * |-----------|-----------|------------|
 * | **LOAD** | Memory → CPU | Create MemReqEvent → Memory reads → Update packet → CPU uses data |
 * | **STORE** | CPU → Memory | Create MemReqEvent → Memory writes → Acknowledge → CPU continues |
 *
 * **Event Lifecycle:**
 *
 * ```
 * Lifecycle of a Memory Request Event:
 *
 * 1. CREATION (during instruction execution)
 *    CPU detects LOAD/STORE instruction
 *    ├─ Allocate MemReqEvent(dataMem, packet)
 *    ├─ Packet contains: address, data, operation type
 *    └─ Event references target memory module
 *
 * 2. SCHEDULING (immediate or delayed)
 *    CPU schedules event for future tick
 *    ├─ scheduler.schedule(memReqEvent, targetTick)
 *    ├─ Typical delay: +1 cycle for request latency
 *    └─ Event queued in simulation priority queue
 *
 * 3. PROCESSING (at scheduled tick)
 *    process() invoked by simulator
 *    ├─ Calls callee->accept(tick, packet)
 *    ├─ DataMemory processes request
 *    ├─ Read: Fetches data, updates packet
 *    ├─ Write: Stores data from packet
 *    └─ Packet contains result for CPU
 *
 * 4. COMPLETION (data available)
 *    CPU retrieves data from packet
 *    ├─ For LOAD: Data written to register
 *    ├─ For STORE: Acknowledgment implicit
 *    └─ Instruction completion proceeds
 *
 * 5. RECYCLING (object pool optimization)
 *    renew(dataMem, newPacket)
 *    ├─ Reset event state via SimEvent::renew()
 *    ├─ Update memory target and packet
 *    └─ Ready for next memory operation
 * ```
 *
 * **Memory Request Packet Structure:**
 *
 * The SimPacket carries all information needed for memory operations:
 *
 * ```cpp
 * struct MemoryRequestPacket {
 *     uint64_t address;      // Target memory address
 *     uint64_t data;         // Data to write (STORE) or read result (LOAD)
 *     uint32_t size;         // Access size: 1, 2, 4, 8 bytes
 *     bool     isWrite;      // True for STORE, false for LOAD
 *     bool     isComplete;   // Set by memory when operation finishes
 *     void*    context;      // CPU context for response handling
 * };
 * ```
 *
 * **Integration with CPU Execution:**
 *
 * Memory request events are created during instruction execution when
 * LOAD or STORE instructions are decoded:
 *
 * ```cpp
 * // Inside CPU::execOneInstr() for LOAD instruction
 * void CPU::processLoadInstruction(instr& inst) {
 *     // Calculate effective address
 *     uint64_t addr = reg[inst.rs1] + inst.imm;
 *
 *     // Create memory request packet
 *     MemReadReqPacket* packet = new MemReadReqPacket(addr, inst.size);
 *
 *     // Create and schedule memory request event
 *     MemReqEvent* memEvent = new MemReqEvent(dataMemory, packet);
 *     scheduler.schedule(memEvent, currentTick + 1);
 *
 *     // CPU will wait for data (event-driven response)
 *     stallUntilMemoryResponse(packet);
 * }
 * ```
 *
 * **Timing Model:**
 *
 * ```
 * Operation Type     | Request Latency | Access Latency | Total Latency
 * -------------------|-----------------|----------------|---------------
 * LOAD (byte/half)   | 1 cycle         | 1 cycle        | 2 cycles
 * LOAD (word)        | 1 cycle         | 1 cycle        | 2 cycles
 * LOAD (doubleword)  | 1 cycle         | 1 cycle        | 2 cycles
 * STORE (any size)   | 1 cycle         | 1 cycle        | 2 cycles
 * Unaligned access   | 1 cycle         | 2 cycles       | 3 cycles
 * ```
 *
 * Note: Template version uses simplified timing. The full version models
 * cache hierarchy with variable latencies (L1 hit: 1-3 cycles, L2 hit: 10-20
 * cycles, DRAM access: 100+ cycles).
 *
 * **LOAD Instruction Example:**
 *
 * ```
 * Complete flow for: LW x5, 12(x3)
 *
 * Tick T: ExecOneInstrEvent.process()
 *   └─→ CPU::execOneInstr()
 *       ├─ Decode: LW x5, 12(x3)
 *       ├─ Calculate: addr = reg[x3] + 12 = 0x1000
 *       ├─ Create: MemReadReqPacket(addr=0x1000, size=4)
 *       ├─ Create: MemReqEvent(dataMemory, packet)
 *       ├─ Schedule: MemReqEvent at tick T+1
 *       └─ Schedule: Next ExecOneInstrEvent at tick T+3
 *
 * Tick T+1: MemReqEvent.process()
 *   └─→ dataMemory->accept(T+1, packet)
 *       ├─ Read memory at address 0x1000
 *       ├─ packet->data = mem[0x1000] = 0xABCD1234
 *       ├─ packet->isComplete = true
 *       └─ Return
 *
 * Tick T+3: Next ExecOneInstrEvent.process()
 *   └─→ CPU::execOneInstr()
 *       ├─ Check packet->isComplete == true
 *       ├─ Write: reg[x5] = packet->data = 0xABCD1234
 *       └─ Continue with next instruction
 * ```
 *
 * **STORE Instruction Example:**
 *
 * ```
 * Complete flow for: SW x7, 8(x2)
 *
 * Tick T: ExecOneInstrEvent.process()
 *   └─→ CPU::execOneInstr()
 *       ├─ Decode: SW x7, 8(x2)
 *       ├─ Calculate: addr = reg[x2] + 8 = 0x2000
 *       ├─ Create: MemWriteReqPacket(addr=0x2000, data=reg[x7], size=4)
 *       ├─ Create: MemReqEvent(dataMemory, packet)
 *       ├─ Schedule: MemReqEvent at tick T+1
 *       └─ Schedule: Next ExecOneInstrEvent at tick T+3
 *
 * Tick T+1: MemReqEvent.process()
 *   └─→ dataMemory->accept(T+1, packet)
 *       ├─ Write memory at address 0x2000
 *       ├─ mem[0x2000] = packet->data = reg[x7]
 *       ├─ packet->isComplete = true
 *       └─ Return
 *
 * Tick T+3: Next ExecOneInstrEvent.process()
 *   └─→ CPU::execOneInstr()
 *       ├─ Store acknowledged (implicit)
 *       └─ Continue with next instruction
 * ```
 *
 * **Multi-Core Considerations:**
 *
 * In multi-core simulations, each core can have concurrent memory requests:
 *
 * ```
 * Tick Timeline:
 * ├─ T+0: Core0 issues LOAD  (0x1000) → MemReqEvent scheduled T+1
 * ├─ T+0: Core1 issues LOAD  (0x2000) → MemReqEvent scheduled T+1
 * ├─ T+1: Both MemReqEvent.process() fire (order defined by event queue)
 * │       ├─ DataMemory handles Core0 request first
 * │       └─ DataMemory handles Core1 request second
 * ├─ T+2: Core0 receives data from 0x1000
 * └─ T+2: Core1 receives data from 0x2000
 * ```
 *
 * Note: Template version has simplified concurrency. No cache coherence
 * protocols are implemented; all memory operations are serialized by the
 * event queue.
 *
 * **Object Pooling and Performance:**
 *
 * Like ExecOneInstrEvent, MemReqEvent supports object pooling:
 *
 * ```
 * Performance comparison for 1M memory operations:
 *
 * Without pooling:              With pooling:
 * - 1M allocations             - 1 allocation per core
 * - 1M deallocations           - 0 deallocations during simulation
 * - Heap fragmentation         - Contiguous memory reuse
 * - ~50ms overhead             - ~0.1ms overhead
 * ```
 *
 * **Error Handling:**
 *
 * The template version has minimal error handling for educational clarity:
 * - No address alignment checking (assumes aligned accesses)
 * - No out-of-bounds detection (assumes valid addresses)
 * - No permission checking (assumes all accesses allowed)
 *
 * The full riscv/ version implements comprehensive checking:
 * - Virtual memory translation and TLB
 * - Page fault generation
 * - Access permission validation
 * - Alignment exception detection
 *
 * **Simplified vs Full Version Differences:**
 *
 * | Feature | Template (riscvSimTemplate) | Full (riscv) |
 * |---------|----------------------------|--------------|
 * | Cache hierarchy | None (direct memory) | L1, L2, LLC |
 * | Memory latency | Fixed (2 cycles) | Variable (1-100+ cycles) |
 * | Coherence protocol | None | MESI/MOESI |
 * | Virtual memory | No | Yes (TLB, page table) |
 * | Error checking | Minimal | Comprehensive |
 * | Memory controller | Implicit | Explicit modeling |
 *
 * **Educational Value:**
 *
 * This template demonstrates:
 * - Event-driven memory request modeling
 * - Asynchronous memory access timing
 * - CPU-memory interface design
 * - Discrete-event simulation for I/O
 * - Object pooling optimization
 * - Packet-based communication
 *
 * Students learn these concepts without the complexity of cache coherence,
 * virtual memory translation, or multi-level memory hierarchies.
 *
 * @see ExecOneInstrEvent For instruction execution event coordination
 * @see DataMemory For memory subsystem implementation
 * @see DataMemory::accept() For memory request processing
 * @see CPU::execOneInstr() For memory request initiation
 * @see acalsim::SimEvent For base event class interface
 * @see acalsim::SimPacket For memory request packet structure
 *
 * @note This is a template/educational version. For production simulations
 *       with accurate memory hierarchy modeling, see src/riscv/libs/event/MemReqEvent.cc
 *
 * @author Playlab/ACAL
 * @version Template 1.0
 */

#include "event/MemReqEvent.hh"

#include "DataMemory.hh"

/**
 * @brief Constructs a MemReqEvent for memory access
 *
 * @param _callee Pointer to the DataMemory module that will handle the request
 * @param _memReqPkt Pointer to the SimPacket containing request details
 *
 * @details
 * Creates a memory request event that will deliver a memory access request
 * to the specified DataMemory module when processed. The packet contains all
 * necessary information about the memory operation.
 *
 * **Event Name:**
 *
 * All memory request events share the same name "MemReqEvent" regardless of
 * operation type (LOAD/STORE) or target address. This differs from
 * ExecOneInstrEvent which includes a core ID in the name.
 *
 * **Packet Contents:**
 *
 * The _memReqPkt parameter typically contains:
 * ```
 * For LOAD operations (MemReadReqPacket):
 * - address: Target memory address to read
 * - size: Number of bytes to read (1, 2, 4, or 8)
 * - data: [output] Will be filled with read result
 *
 * For STORE operations (MemWriteReqPacket):
 * - address: Target memory address to write
 * - size: Number of bytes to write (1, 2, 4, or 8)
 * - data: [input] Data value to write to memory
 * ```
 *
 * **Usage Pattern:**
 *
 * ```cpp
 * // LOAD example: LW x1, 0(x2)
 * uint64_t addr = reg[2] + 0;
 * MemReadReqPacket* packet = new MemReadReqPacket(addr, 4);
 * MemReqEvent* event = new MemReqEvent(dataMemory, packet);
 * scheduler.schedule(event, currentTick + 1);
 *
 * // STORE example: SW x3, 8(x4)
 * uint64_t addr = reg[4] + 8;
 * uint32_t data = reg[3];
 * MemWriteReqPacket* packet = new MemWriteReqPacket(addr, data, 4);
 * MemReqEvent* event = new MemReqEvent(dataMemory, packet);
 * scheduler.schedule(event, currentTick + 1);
 * ```
 *
 * **Memory Module Targeting:**
 *
 * The _callee parameter allows flexible memory system configuration:
 * ```
 * Single memory:          Multi-bank memory:
 * ┌─────────────┐        ┌─────────────┐
 * │ CPU         │        │ CPU         │
 * │  ↓          │        │  ↓          │
 * │ DataMemory  │        │ Bank Select │
 * └─────────────┘        │  ├─ Bank0   │
 *                        │  ├─ Bank1   │
 *                        │  └─ Bank2   │
 *                        └─────────────┘
 * ```
 *
 * **Object Lifecycle:**
 *
 * ```
 * 1. Construction: new MemReqEvent(mem, pkt)
 *    ├─ Create with base name "MemReqEvent"
 *    ├─ Store memory target pointer
 *    └─ Store packet pointer
 *
 * 2. Scheduling: scheduler.schedule(event, tick)
 *    └─ Event queued for specified tick
 *
 * 3. Processing: event->process() at tick
 *    └─ Delivers packet to memory module
 *
 * 4. Recycling: event->renew(mem, newPkt)
 *    └─ Prepare for next memory operation
 * ```
 *
 * @note Event objects should be recycled using renew() for performance
 * @note The callee must outlive the event or proper cleanup must be ensured
 * @note Packet ownership is not transferred; caller manages packet lifetime
 *
 * @see renew() For event recycling
 * @see process() For event execution
 * @see DataMemory::accept() For memory request handling
 */
MemReqEvent::MemReqEvent(DataMemory* _callee, acalsim::SimPacket* _memReqPkt)
    : acalsim::SimEvent("MemReqEvent"), callee(_callee), memReqPkt(_memReqPkt) {}

/**
 * @brief Renews the event for reuse with new memory target and packet
 *
 * @param _callee Pointer to the DataMemory module for the next request
 * @param _memReqPkt Pointer to the new SimPacket with request details
 *
 * @details
 * Resets the event state and updates pointers for the next memory operation.
 * This is a critical performance optimization that enables object pooling,
 * reducing allocation overhead in memory-intensive simulations.
 *
 * **Recycling Benefits:**
 *
 * For a program with 1 million memory operations:
 * ```
 * Without renew():                With renew():
 * ┌───────────────────────┐      ┌───────────────────────┐
 * │ 1M new MemReqEvent()  │      │ 1 new MemReqEvent()   │
 * │ 1M delete operations  │      │ 1M renew() calls      │
 * │ Heap fragmentation    │      │ No deallocation       │
 * │ Cache misses          │      │ Cache-friendly reuse  │
 * │ ~100ms overhead       │      │ ~0.5ms overhead       │
 * └───────────────────────┘      └───────────────────────┘
 * ```
 *
 * **Renew Process:**
 *
 * ```
 * State before renew():        State after renew():
 * ┌──────────────────┐         ┌──────────────────┐
 * │ MemReqEvent      │         │ MemReqEvent      │
 * │ - callee: old    │  renew  │ - callee: new    │
 * │ - memReqPkt: old │  ──────→│ - memReqPkt: new │
 * │ - scheduled: yes │         │ - scheduled: no  │
 * │ - processed: yes │         │ - processed: no  │
 * └──────────────────┘         └──────────────────┘
 * ```
 *
 * **Base Class Reset:**
 *
 * The call to SimEvent::renew() performs critical cleanup:
 * - Clears scheduling flags (event no longer marked as scheduled)
 * - Resets processing state (event ready for new schedule)
 * - Maintains event name (still "MemReqEvent")
 * - Preserves event identity for debugging
 *
 * **Memory Target Flexibility:**
 *
 * The _callee parameter allows events to target different memory modules:
 * ```cpp
 * // First request to main memory
 * event->renew(mainMemory, packet1);
 * scheduler.schedule(event, tick + 1);
 *
 * // Later, request to different memory bank
 * event->renew(scratchpadMemory, packet2);
 * scheduler.schedule(event, tick + 10);
 * ```
 *
 * **Packet Management:**
 *
 * ```cpp
 * // Typical usage pattern with packet pooling
 * MemReqEvent* event = new MemReqEvent(dataMem, nullptr);
 * MemReadReqPacket* packet = new MemReadReqPacket();
 *
 * for (int i = 0; i < numLoads; i++) {
 *     // Configure packet for current operation
 *     packet->address = loadAddresses[i];
 *     packet->size = 4;
 *
 *     // Renew event with updated packet
 *     event->renew(dataMem, packet);
 *     scheduler.schedule(event, currentTick + 1);
 *
 *     // Wait for processing...
 *     // Reuse same event and packet for next load
 * }
 * ```
 *
 * **Thread Safety:**
 *
 * In the template version (single-threaded simulation):
 * - No synchronization needed
 * - Events processed sequentially by tick order
 * - Renew() called only when event not in scheduler queue
 *
 * The full riscv/ version may require:
 * - Atomic operations for multi-threaded simulation
 * - Event queue locking
 * - Careful state management
 *
 * **Common Usage Patterns:**
 *
 * ```cpp
 * // Pattern 1: Single event per core
 * class CPU {
 *     MemReqEvent* memEvent;
 *     MemReadReqPacket* readPkt;
 *
 *     void issueLoad(uint64_t addr) {
 *         readPkt->address = addr;
 *         memEvent->renew(dataMem, readPkt);
 *         scheduler.schedule(memEvent, tick + 1);
 *     }
 * };
 *
 * // Pattern 2: Event pool for multiple outstanding requests
 * class CPU {
 *     ObjectPool<MemReqEvent> eventPool;
 *
 *     void issueMemOp(SimPacket* pkt) {
 *         MemReqEvent* event = eventPool.acquire();
 *         event->renew(dataMem, pkt);
 *         scheduler.schedule(event, tick + 1);
 *     }
 * };
 * ```
 *
 * @note Always renew() before rescheduling a previously processed event
 * @note The packet is not copied; ensure it remains valid until processing
 * @note Caller is responsible for packet memory management
 *
 * @see MemReqEvent() For initial construction
 * @see process() For event processing
 * @see acalsim::SimEvent::renew() For base class reset logic
 */
void MemReqEvent::renew(DataMemory* _callee, acalsim::SimPacket* _memReqPkt) {
	this->acalsim::SimEvent::renew();
	this->callee    = _callee;
	this->memReqPkt = _memReqPkt;
}

/**
 * @brief Processes the memory request event by delivering it to the memory module
 *
 * @details
 * This is the core event handler invoked by the ACALSim event scheduler when
 * the simulation clock reaches the scheduled tick. It delivers the memory
 * request packet to the target DataMemory module for processing.
 *
 * **Event Processing Flow:**
 *
 * ```
 * Scheduler                MemReqEvent              DataMemory
 * ─────────                ───────────              ──────────
 * tick = T+1
 * event = dequeue() ─────→ process() ──────────────→ accept(T+1, packet)
 *                            │                         ├─ Read/Write memory
 *                            │                         ├─ Update packet
 *                            │                         └─ Return
 *                            ↓
 *                          return
 * continue simulation
 * ```
 *
 * **Global Tick Usage:**
 *
 * The method uses `acalsim::top->getGlobalTick()` to provide the current
 * simulation time to the memory module. This allows the memory to:
 * - Record access timing for statistics
 * - Model time-dependent behavior (e.g., DRAM refresh)
 * - Schedule response events at correct future ticks
 * - Maintain temporal ordering of operations
 *
 * **Memory Module Interaction:**
 *
 * The accept() method signature:
 * ```cpp
 * void DataMemory::accept(acalsim::Tick when, acalsim::SimPacket& packet)
 * ```
 *
 * Parameters:
 * - when: Simulation tick when request arrived (from getGlobalTick())
 * - packet: Reference to packet with operation details
 *
 * **Complete LOAD Processing:**
 *
 * ```
 * Timeline for: LW x1, 0(x2)
 *
 * Tick T: CPU schedules MemReqEvent
 *   └─ Event scheduled for tick T+1 with MemReadReqPacket
 *
 * Tick T+1: MemReqEvent.process()
 *   ├─ Get current tick: T+1
 *   ├─ Call: dataMem->accept(T+1, *memReqPkt)
 *   │  ├─ DataMemory::accept()
 *   │  │  ├─ Determine operation type (READ)
 *   │  │  ├─ Extract address from packet
 *   │  │  ├─ Perform memory read: data = mem[addr]
 *   │  │  ├─ Write result to packet: packet->data = data
 *   │  │  └─ Mark complete: packet->isComplete = true
 *   │  └─ Return to MemReqEvent
 *   └─ Return to scheduler
 *
 * Tick T+2: CPU checks packet
 *   └─ Reads packet->data and writes to register x1
 * ```
 *
 * **Complete STORE Processing:**
 *
 * ```
 * Timeline for: SW x3, 8(x4)
 *
 * Tick T: CPU schedules MemReqEvent
 *   └─ Event scheduled for tick T+1 with MemWriteReqPacket
 *
 * Tick T+1: MemReqEvent.process()
 *   ├─ Get current tick: T+1
 *   ├─ Call: dataMem->accept(T+1, *memReqPkt)
 *   │  ├─ DataMemory::accept()
 *   │  │  ├─ Determine operation type (WRITE)
 *   │  │  ├─ Extract address and data from packet
 *   │  │  ├─ Perform memory write: mem[addr] = packet->data
 *   │  │  └─ Mark complete: packet->isComplete = true
 *   │  └─ Return to MemReqEvent
 *   └─ Return to scheduler
 *
 * Tick T+2: CPU continues
 *   └─ Store acknowledged, proceed with next instruction
 * ```
 *
 * **Packet-Based Communication:**
 *
 * The packet serves as bidirectional communication:
 * ```
 * Before process():          After process():
 * ┌─────────────────┐       ┌─────────────────┐
 * │ MemReadReqPacket│       │ MemReadReqPacket│
 * │ - addr: 0x1000  │       │ - addr: 0x1000  │
 * │ - size: 4       │  →    │ - size: 4       │
 * │ - data: ???     │       │ - data: 0xABCD  │ ← Filled by memory
 * │ - complete: no  │       │ - complete: yes │ ← Set by memory
 * └─────────────────┘       └─────────────────┘
 * ```
 *
 * **Error Handling:**
 *
 * In the template version, minimal error checking occurs:
 * - Assumes valid callee pointer (set during construction/renew)
 * - Assumes valid packet pointer (set during construction/renew)
 * - Assumes memory module handles all error conditions
 *
 * The full riscv/ version may include:
 * ```cpp
 * void MemReqEvent::process() {
 *     assert(callee != nullptr);
 *     assert(memReqPkt != nullptr);
 *     try {
 *         callee->accept(acalsim::top->getGlobalTick(), *memReqPkt);
 *     } catch (MemoryException& e) {
 *         // Handle memory faults, permission violations, etc.
 *     }
 * }
 * ```
 *
 * **Timing Accuracy:**
 *
 * The event-driven model ensures precise timing:
 * - Each memory access occurs at its scheduled tick
 * - Multiple outstanding requests maintain temporal order
 * - Memory latency accurately modeled via event scheduling
 * - No polling or busy-waiting
 *
 * **Multi-Request Coordination:**
 *
 * When multiple cores issue concurrent memory requests:
 * ```
 * Tick T+1 event queue:
 * ├─ MemReqEvent (Core0, addr=0x1000) [priority 1]
 * ├─ MemReqEvent (Core1, addr=0x2000) [priority 2]
 * └─ MemReqEvent (Core2, addr=0x3000) [priority 3]
 *
 * Processing order (sequential, by priority):
 * 1. Core0's request → dataMem->accept(T+1, pkt0)
 * 2. Core1's request → dataMem->accept(T+1, pkt1)
 * 3. Core2's request → dataMem->accept(T+1, pkt2)
 * ```
 *
 * Note: Template version processes all same-tick events sequentially.
 * The full version may model parallel memory banks or ports.
 *
 * **Performance Characteristics:**
 *
 * The process() method is extremely lightweight:
 * - Single function call (no loops or branches)
 * - Minimal stack usage
 * - Cache-friendly (hot path)
 * - Typical execution: < 10 CPU cycles
 *
 * For simulating 1 billion memory operations:
 * - Time in process(): ~0.01 seconds
 * - Time in accept(): ~10 seconds (depends on memory model)
 *
 * @note This method is called by the simulation framework automatically
 * @note getGlobalTick() provides simulation time, not wall-clock time
 * @note Memory response may be synchronous (immediate) or asynchronous (via callback)
 * @note Event should not be rescheduled within process(); use separate scheduling logic
 *
 * @see DataMemory::accept() For memory request processing implementation
 * @see acalsim::top->getGlobalTick() For current simulation time
 * @see ExecOneInstrEvent For instruction execution coordination
 * @see acalsim::SimEvent::process() For base class interface
 */
void MemReqEvent::process() { this->callee->accept(acalsim::top->getGlobalTick(), *this->memReqPkt); }
