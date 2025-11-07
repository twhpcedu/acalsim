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
 * @file ExecOneInstrEvent.cc
 * @brief RISC-V instruction execution event implementation
 *
 * @details
 * This file implements the ExecOneInstrEvent class, which triggers the execution
 * of a single RISC-V instruction in the simulated CPU. It is a core component of
 * the event-driven instruction scheduling and execution model.
 *
 * **Event-Driven Execution Model:**
 *
 * The RISC-V simulator uses an event-driven approach to model instruction execution
 * timing. Each instruction's execution is triggered by an ExecOneInstrEvent scheduled
 * at a specific simulation tick, allowing precise modeling of pipeline delays,
 * multi-cycle instructions, and instruction-level parallelism.
 *
 * ```
 * Execution Timeline:
 * ├─ Tick 0:  Schedule ExecOneInstrEvent for CPU core 0 at tick 1
 * ├─ Tick 1:  Process event → Execute ADD instruction → Schedule next at tick 2
 * ├─ Tick 2:  Process event → Execute LOAD (miss) → Schedule next at tick 12
 * ├─ Tick 12: Process event → LOAD completes → Execute next instruction
 * └─ Continue until program completion...
 * ```
 *
 * **Instruction Scheduling Mechanisms:**
 *
 * ExecOneInstrEvent supports multiple scheduling patterns:
 *
 * | Pattern | Timing | Use Case |
 * |---------|--------|----------|
 * | **Sequential** | 1 cycle | Simple in-order execution |
 * | **Multi-cycle** | N cycles | Complex operations (DIV, FPU) |
 * | **Stalled** | Variable | Cache miss, data hazard |
 * | **Pipelined** | Overlapped | Multiple instructions in flight |
 *
 * **Event Lifecycle:**
 *
 * ```
 * 1. Construction: ExecOneInstrEvent(id, cpu_ptr)
 *    ├─ Assigns unique event name with CPU ID
 *    ├─ Stores CPU reference for execution
 *    └─ Inherits SimEvent scheduling capabilities
 *
 * 2. Scheduling: scheduler.schedule(event, target_tick)
 *    ├─ Event placed in CPU's event queue
 *    ├─ Sorted by target tick for ordered processing
 *    └─ Awaits simulation time advancement
 *
 * 3. Processing: process() called at target_tick
 *    ├─ Invokes cpu->execOneInstr()
 *    ├─ Fetches next instruction from PC
 *    ├─ Decodes and executes instruction
 *    ├─ Updates architectural state
 *    └─ Schedules next ExecOneInstrEvent
 *
 * 4. Recycling: renew(id, cpu_ptr)
 *    ├─ Resets event for object pool reuse
 *    ├─ Updates CPU reference
 *    └─ Ready for next scheduling
 * ```
 *
 * **Memory Integration:**
 *
 * When an instruction requires memory access (LOAD/STORE), the execution flow
 * involves coordination with MemReqEvent:
 *
 * ```
 * ExecOneInstrEvent (tick T)
 *   ├─ Decode LOAD instruction
 *   ├─ Calculate effective address
 *   ├─ Create MemReqEvent with address
 *   ├─ Schedule MemReqEvent at tick T+1
 *   └─ Schedule next ExecOneInstrEvent at tick T+cache_latency
 *
 * MemReqEvent (tick T+1)
 *   ├─ Submit request to DataMemory
 *   ├─ DataMemory processes request
 *   └─ Returns data via callback
 *
 * ExecOneInstrEvent (tick T+cache_latency)
 *   ├─ Data now available in register
 *   └─ Continue with next instruction
 * ```
 *
 * **Timing Model:**
 *
 * The event implements a cycle-accurate timing model:
 *
 * ```cpp
 * // Example timing scenarios:
 *
 * // Fast path: Register-to-register operation (1 cycle)
 * tick_current:     Process ADD r3, r1, r2
 * tick_current + 1: Process next instruction
 *
 * // Memory load with L1 hit (3 cycles)
 * tick_current:     Process LOAD r1, 0(r2)
 * tick_current + 1: Submit memory request
 * tick_current + 3: Data available, process next instruction
 *
 * // Memory load with L2 hit (12 cycles)
 * tick_current:      Process LOAD r1, 0(r2)
 * tick_current + 1:  Submit memory request (L1 miss)
 * tick_current + 12: Data available, process next instruction
 *
 * // Division operation (variable cycles)
 * tick_current:     Process DIV r3, r1, r2
 * tick_current + N: Division completes (N depends on operands)
 * tick_current + N: Process next instruction
 * ```
 *
 * **Multi-Core Support:**
 *
 * Each CPU core has independent ExecOneInstrEvent instances:
 *
 * ```
 * Core 0 Event Stream:  [Exec@T=0] → [Exec@T=1] → [Exec@T=2] → ...
 * Core 1 Event Stream:  [Exec@T=0] → [Exec@T=1] → [Exec@T=2] → ...
 * Core 2 Event Stream:  [Exec@T=0] → [Exec@T=1] → [Exec@T=2] → ...
 * Core 3 Event Stream:  [Exec@T=0] → [Exec@T=1] → [Exec@T=2] → ...
 * ```
 *
 * **Event Chaining and Dependencies:**
 *
 * Instructions often create chains of dependent events:
 *
 * ```
 * LOAD r1, 0(r2)
 *   └─ ExecOneInstrEvent (decode)
 *       └─ MemReqEvent (fetch data)
 *           └─ ExecOneInstrEvent (register write + next instr)
 *
 * STORE r3, 0(r4)
 *   └─ ExecOneInstrEvent (decode)
 *       └─ MemReqEvent (write data)
 *           └─ ExecOneInstrEvent (next instr)
 *
 * BRANCH target
 *   └─ ExecOneInstrEvent (evaluate condition)
 *       ├─ Taken: Schedule ExecOneInstrEvent at target PC
 *       └─ Not taken: Schedule ExecOneInstrEvent at PC+4
 * ```
 *
 * **Performance Characteristics:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | Constructor | O(1) | String concatenation for event name |
 * | renew() | O(1) | Resets state for recycling |
 * | process() | O(1) | Delegates to CPU::execOneInstr() |
 * | Scheduling | O(log N) | Insertion into priority queue |
 *
 * **Memory Footprint:**
 * - sizeof(ExecOneInstrEvent) ≈ sizeof(SimEvent) + sizeof(CPU*) + event_name
 * - Approximately 64-96 bytes per event instance
 *
 * **Object Pool Usage:**
 *
 * While not mandatory, ExecOneInstrEvent can be used with RecycleContainer
 * for improved memory performance in high-frequency execution scenarios:
 *
 * ```cpp
 * // Object pool pattern (optional optimization)
 * RecycleContainer<ExecOneInstrEvent> execEventPool;
 *
 * // Get recycled event
 * ExecOneInstrEvent* evt = execEventPool.get<ExecOneInstrEvent>(core_id, cpu_ptr);
 * scheduler.schedule(evt, target_tick);
 *
 * // After process(), event auto-returns to pool
 * // Next get() reuses the same memory
 * ```
 *
 * @code{.cpp}
 * // Example: Simple sequential execution
 * CPU* cpu = new CPU(0);  // Core ID 0
 * ExecOneInstrEvent* execEvt = new ExecOneInstrEvent(0, cpu);
 *
 * // Schedule first instruction
 * scheduler.schedule(execEvt, currentTick() + 1);
 *
 * // Inside CPU::execOneInstr():
 * void CPU::execOneInstr() {
 *     // Fetch instruction from memory
 *     uint32_t instr = fetchInstr(pc);
 *
 *     // Decode and execute
 *     decodeAndExecute(instr);
 *
 *     // Schedule next instruction (with appropriate delay)
 *     ExecOneInstrEvent* nextEvt = new ExecOneInstrEvent(coreID, this);
 *     scheduler.schedule(nextEvt, currentTick() + getInstrLatency(instr));
 * }
 *
 * // Example: Memory instruction with MemReqEvent coordination
 * void CPU::execLoadInstr(uint64_t addr) {
 *     // Create memory request event
 *     MemReqEvent* memEvt = new MemReqEvent(dataMem, createLoadPacket(addr));
 *     scheduler.schedule(memEvt, currentTick() + 1);
 *
 *     // Schedule next instruction after memory latency
 *     ExecOneInstrEvent* nextEvt = new ExecOneInstrEvent(coreID, this);
 *     scheduler.schedule(nextEvt, currentTick() + memoryLatency);
 * }
 *
 * // Example: Multi-core initialization
 * void initCores(int numCores) {
 *     for (int i = 0; i < numCores; i++) {
 *         CPU* cpu = new CPU(i);
 *         ExecOneInstrEvent* evt = new ExecOneInstrEvent(i, cpu);
 *         scheduler.schedule(evt, 0);  // All start at tick 0
 *     }
 * }
 * @endcode
 *
 * @note Each CPU core typically has one active ExecOneInstrEvent at any time
 * @note The event name includes CPU ID for debugging: "ExecOneInstrEvent0"
 * @note Instruction execution is cycle-accurate based on scheduling
 * @note Memory operations coordinate with MemReqEvent for timing accuracy
 *
 * @warning Do not manually call process() - let the scheduler invoke it
 * @warning Ensure CPU pointer remains valid during event lifetime
 * @warning Scheduling delays must match architectural timing specifications
 *
 * @see ExecOneInstrEvent.hh for class definition
 * @see CPU.hh for instruction execution implementation
 * @see MemReqEvent for memory operation coordination
 * @see SimEvent.hh for base event-driven framework
 * @see docs/for-users/event-driven-simulation.md for usage examples
 *
 * @since ACALSim 0.1.0
 * @ingroup riscv_events
 */

#include "event/ExecOneInstrEvent.hh"

#include "CPU.hh"

/**
 * @brief Construct an instruction execution event for a specific CPU core
 *
 * @param _id CPU core identifier used for event naming and debugging
 * @param _cpu Pointer to the CPU instance that will execute instructions
 *
 * @details
 * Initializes an event that triggers single instruction execution for the
 * specified CPU core. The event name is constructed as "ExecOneInstrEvent{id}"
 * for easy identification in debug logs and traces.
 *
 * **Initialization Flow:**
 * ```
 * Constructor called
 *   ├─ SimEvent::SimEvent("ExecOneInstrEvent{id}")
 *   │   ├─ Assign unique global event ID
 *   │   ├─ Store event name
 *   │   └─ Initialize gem5::Event base
 *   └─ Store CPU pointer for execution
 * ```
 *
 * **Typical Usage Pattern:**
 * ```cpp
 * // Single-core setup
 * CPU* cpu0 = new CPU(0);
 * ExecOneInstrEvent* evt = new ExecOneInstrEvent(0, cpu0);
 * scheduler.schedule(evt, startTick);
 *
 * // Multi-core setup
 * for (int i = 0; i < numCores; i++) {
 *     CPU* cpu = new CPU(i);
 *     ExecOneInstrEvent* evt = new ExecOneInstrEvent(i, cpu);
 *     scheduler.schedule(evt, 0);  // All cores start at tick 0
 * }
 * ```
 *
 * @note The _id parameter is purely for naming - it does not affect execution
 * @note The CPU pointer must remain valid throughout the event's lifetime
 * @note Complexity: O(1) with string concatenation overhead
 *
 * @warning CPU pointer is not reference-counted - ensure valid lifecycle
 */
ExecOneInstrEvent::ExecOneInstrEvent(int _id, CPU* _cpu)
    : acalsim::SimEvent("ExecOneInstrEvent" + std::to_string(_id)), cpu(_cpu) {}

/**
 * @brief Reset event state for object pool recycling
 *
 * @param _id New CPU core identifier for event naming
 * @param _cpu New CPU instance pointer for instruction execution
 *
 * @details
 * Reinitializes the event for reuse from an object pool, avoiding expensive
 * allocation/deallocation cycles. This is the preferred method when using
 * RecycleContainer for memory management.
 *
 * **Recycling Flow:**
 * ```
 * Event completes → Release to pool → Later reuse:
 *
 * 1. After process():
 *    event->release()  // Return to RecycleContainer
 *
 * 2. Pool storage:
 *    Event stays in memory, marked as available
 *
 * 3. Next allocation:
 *    pool.get<ExecOneInstrEvent>(new_id, new_cpu)
 *      └─ Calls renew(new_id, new_cpu)
 *      └─ Event ready for scheduling
 * ```
 *
 * **Performance Benefit:**
 * | Approach | Allocation | Performance |
 * |----------|-----------|-------------|
 * | **New/Delete** | malloc/free every time | ~100-500 cycles |
 * | **Object Pool** | Reuse from pool | ~10-50 cycles |
 *
 * **Example with Object Pool:**
 * ```cpp
 * RecycleContainer<ExecOneInstrEvent> eventPool;
 *
 * void scheduleInstruction(int coreID, CPU* cpu, Tick when) {
 *     // Get event from pool (may call renew if reusing)
 *     ExecOneInstrEvent* evt = eventPool.get<ExecOneInstrEvent>(coreID, cpu);
 *
 *     // Schedule for execution
 *     scheduler.schedule(evt, when);
 *
 *     // Event auto-returns to pool after process()
 * }
 * ```
 *
 * @note Calls SimEvent::renew() to reset base class state
 * @note Updates CPU pointer to new target core
 * @note Generates new unique event ID via SimEvent::renew()
 * @note Complexity: O(1)
 *
 * @see RecycleContainer for object pool management
 * @see SimEvent::renew() for base class recycling behavior
 */
void ExecOneInstrEvent::renew(int _id, CPU* _cpu) {
	this->SimEvent::renew();
	this->cpu = _cpu;
}

/**
 * @brief Execute one RISC-V instruction on the associated CPU core
 *
 * @details
 * This is the event's main processing method, called by the simulation scheduler
 * when the event's scheduled tick is reached. It delegates to the CPU's
 * execOneInstr() method, which performs the actual instruction fetch, decode,
 * and execution.
 *
 * **Invocation Context:**
 * ```
 * Simulation Loop:
 *   while (eventsRemaining()) {
 *     Tick currentTick = scheduler.nextEventTick();
 *     advanceTimeTo(currentTick);
 *     Event* evt = scheduler.dequeueNext();
 *     evt->process();  // ← This method called here
 *   }
 * ```
 *
 * **Execution Sequence:**
 * ```
 * process() called at scheduled tick
 *   └─ cpu->execOneInstr()
 *       ├─ Fetch instruction from PC
 *       ├─ Decode instruction opcode/operands
 *       ├─ Execute operation:
 *       │   ├─ ALU ops: Update registers, PC += 4
 *       │   ├─ LOAD/STORE: Create MemReqEvent
 *       │   ├─ BRANCH: Evaluate condition, update PC
 *       │   └─ System: Handle CSR/exceptions
 *       └─ Schedule next ExecOneInstrEvent
 *           └─ Delay based on instruction type
 * ```
 *
 * **Scheduling Patterns After Execution:**
 *
 * ```cpp
 * // Pattern 1: Simple ALU instruction (1 cycle)
 * void CPU::execOneInstr() {
 *     uint32_t instr = fetchInstr(pc);
 *     if (isALU(instr)) {
 *         executeALU(instr);
 *         pc += 4;
 *         scheduleEvent(new ExecOneInstrEvent(id, this), curTick + 1);
 *     }
 * }
 *
 * // Pattern 2: Load instruction (variable latency)
 * void CPU::execOneInstr() {
 *     uint32_t instr = fetchInstr(pc);
 *     if (isLoad(instr)) {
 *         uint64_t addr = calculateAddress(instr);
 *         MemReqEvent* memEvt = new MemReqEvent(dataMem, loadPacket(addr));
 *         scheduleEvent(memEvt, curTick + 1);
 *         // Next instruction after memory returns (cache latency)
 *         scheduleEvent(new ExecOneInstrEvent(id, this), curTick + cacheLatency);
 *     }
 * }
 *
 * // Pattern 3: Branch instruction (conditional execution)
 * void CPU::execOneInstr() {
 *     uint32_t instr = fetchInstr(pc);
 *     if (isBranch(instr)) {
 *         bool taken = evaluateBranch(instr);
 *         pc = taken ? branchTarget(instr) : pc + 4;
 *         scheduleEvent(new ExecOneInstrEvent(id, this), curTick + 1);
 *     }
 * }
 * ```
 *
 * **Interaction with Memory System:**
 * ```
 * ExecOneInstrEvent processes LOAD:
 *   ├─ Creates MemReqEvent with load address
 *   ├─ Schedules MemReqEvent for next tick
 *   └─ Schedules next ExecOneInstrEvent after cache latency
 *
 * MemReqEvent processes at next tick:
 *   ├─ Submits request to DataMemory
 *   ├─ DataMemory queues or processes immediately
 *   └─ Callback when data ready
 *
 * ExecOneInstrEvent resumes after latency:
 *   ├─ Data now in register file
 *   └─ Continue with next instruction
 * ```
 *
 * **Error Handling:**
 * If CPU::execOneInstr() encounters an exception (illegal instruction,
 * page fault, etc.), it typically:
 * - Sets exception state in CPU
 * - Transitions to exception handler PC
 * - Schedules next ExecOneInstrEvent at handler address
 *
 * @note Called automatically by scheduler - never call manually
 * @note After process(), event may be recycled (if using object pools)
 * @note CPU pointer must be valid when process() is invoked
 * @note Thread safety: Process executes in CPU's simulator thread
 * @note Complexity: O(1) for delegation, actual execution varies by instruction
 *
 * @see CPU::execOneInstr() for instruction execution implementation
 * @see MemReqEvent for memory operation handling
 * @see SimEvent::process() for base class contract
 */
void ExecOneInstrEvent::process() { this->cpu->execOneInstr(); }
