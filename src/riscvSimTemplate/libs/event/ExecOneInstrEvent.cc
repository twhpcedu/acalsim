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
 * @brief Simplified instruction execution event for riscvSimTemplate
 *
 * @details
 * This file implements the ExecOneInstrEvent class for the riscvSimTemplate,
 * a simplified educational version of the full RISC-V simulator. It demonstrates
 * core event-driven execution concepts in a streamlined architecture.
 *
 * **Template Version - Simplified Architecture:**
 *
 * Unlike the full riscv/ implementation with separate pipeline stages (IF, DE, EXE, MEM, WB),
 * riscvSimTemplate uses a unified execution model where CPU.execOneInstr() handles the
 * complete instruction flow. This makes it ideal for learning event-driven simulation
 * fundamentals without pipeline complexity.
 *
 * ```
 * Full Version (riscv/):        Template Version (riscvSimTemplate/):
 * ┌─────────────────────┐      ┌─────────────────────┐
 * │ ExecOneInstrEvent   │      │ ExecOneInstrEvent   │
 * │  ↓                  │      │  ↓                  │
 * │ IFStage (fetch)     │      │ CPU.execOneInstr()  │
 * │  ↓                  │      │  - Fetch            │
 * │ DEStage (decode)    │      │  - Decode           │
 * │  ↓                  │      │  - Execute          │
 * │ EXEStage (execute)  │      │  - Memory           │
 * │  ↓                  │      │  - Writeback        │
 * │ MEMStage (memory)   │      │  ↓                  │
 * │  ↓                  │      │ Schedule next       │
 * │ WBStage (writeback) │      └─────────────────────┘
 * │  ↓                  │
 * │ Schedule next       │
 * └─────────────────────┘
 * ```
 *
 * **Event-Driven Execution Model:**
 *
 * The template demonstrates discrete-event simulation where each instruction's
 * execution is modeled as a timed event. This approach allows accurate timing
 * simulation while maintaining simplicity.
 *
 * ```
 * Execution Timeline Example:
 * ├─ Tick 0:  Initialize, schedule first ExecOneInstrEvent at tick 1
 * ├─ Tick 1:  Execute ADD x1, x2, x3 → Schedule next at tick 2
 * ├─ Tick 2:  Execute LW x4, 0(x5) [cache hit] → Schedule next at tick 4
 * ├─ Tick 4:  Execute SUB x6, x7, x8 → Schedule next at tick 5
 * ├─ Tick 5:  Execute SW x4, 8(x5) → Schedule next at tick 7
 * └─ Tick 7:  Continue execution...
 * ```
 *
 * **Instruction Scheduling Mechanisms:**
 *
 * The template supports fundamental scheduling patterns:
 *
 * | Pattern | Timing | Implementation |
 * |---------|--------|----------------|
 * | **Sequential** | Fixed cycles | Default path, predictable timing |
 * | **Memory Access** | Variable | Depends on cache hit/miss |
 * | **Control Flow** | Immediate | Branch/jump updates PC directly |
 *
 * **Event Lifecycle and Object Pooling:**
 *
 * ExecOneInstrEvent uses object pooling for efficiency, demonstrating
 * important simulation performance optimization:
 *
 * ```
 * Lifecycle Stages:
 *
 * 1. CONSTRUCTION (one-time per pool object)
 *    ExecOneInstrEvent(id, cpu)
 *    ├─ Create event name: "ExecOneInstrEvent" + id
 *    ├─ Call SimEvent base constructor
 *    └─ Store CPU pointer for execution
 *
 * 2. FIRST USE (scheduled for specific tick)
 *    scheduler.schedule(event, tick)
 *    ├─ Event added to simulation queue
 *    ├─ Sorted by target tick
 *    └─ Wait for simulation to advance
 *
 * 3. PROCESSING (when tick arrives)
 *    process()
 *    ├─ Called automatically by simulator
 *    ├─ Invokes cpu->execOneInstr()
 *    ├─ Executes one complete instruction
 *    └─ May schedule next event or self-reschedule
 *
 * 4. RECYCLING (object pool optimization)
 *    renew(id, cpu)
 *    ├─ Reset base SimEvent state
 *    ├─ Update CPU pointer (if changed)
 *    └─ Ready for reuse without allocation
 *
 * 5. REUSE (back to step 2)
 *    Same object used for next instruction
 * ```
 *
 * **Integration with CPU Execution:**
 *
 * The event acts as a trigger for CPU instruction processing:
 *
 * ```cpp
 * // Example usage in CPU initialization
 * void CPU::startExecution() {
 *     // Create or reuse event from pool
 *     ExecOneInstrEvent* event = new ExecOneInstrEvent(coreId, this);
 *
 *     // Schedule first instruction at tick 1
 *     scheduler.schedule(event, 1);
 * }
 *
 * // When event fires, it calls:
 * void ExecOneInstrEvent::process() {
 *     this->cpu->execOneInstr();  // Execute one instruction
 * }
 *
 * // Inside CPU::execOneInstr()
 * void CPU::execOneInstr() {
 *     // Fetch instruction from PC
 *     instr inst = fetchInstruction(pc);
 *
 *     // Decode and execute
 *     processInstr(inst);
 *
 *     // Schedule next instruction
 *     scheduleNextExecution();
 * }
 * ```
 *
 * **Memory Interaction Example:**
 *
 * When an instruction needs memory access, coordination with MemReqEvent occurs:
 *
 * ```
 * Timeline for LOAD instruction:
 *
 * Tick T: ExecOneInstrEvent.process()
 *   ├─ cpu->execOneInstr() called
 *   ├─ Decode: LW x1, 0(x2)
 *   ├─ Calculate address: addr = x2 + 0
 *   ├─ Create MemReqEvent(dataMem, readPacket)
 *   ├─ Schedule MemReqEvent at tick T+1
 *   └─ Schedule next ExecOneInstrEvent at tick T+latency
 *
 * Tick T+1: MemReqEvent.process()
 *   ├─ dataMem->accept() called
 *   ├─ Memory processes read request
 *   ├─ Returns data to CPU
 *   └─ Updates instruction packet with result
 *
 * Tick T+latency: Next ExecOneInstrEvent.process()
 *   ├─ Previous LOAD has completed
 *   ├─ Data written to register x1
 *   └─ Continue with next instruction
 * ```
 *
 * **Simplified vs Full Version Differences:**
 *
 * Key simplifications in riscvSimTemplate:
 *
 * - **Single execution method**: CPU.execOneInstr() vs separate pipeline stages
 * - **Direct instruction flow**: No pipeline registers between stages
 * - **Simpler hazard handling**: Basic structural hazards only
 * - **Educational focus**: Clear, linear execution path
 * - **Reduced state tracking**: Minimal inter-stage communication
 *
 * **Timing Model:**
 *
 * ```
 * Instruction Type    | Cycles | Notes
 * --------------------|--------|----------------------------------
 * ALU operations      | 1      | ADD, SUB, AND, OR, XOR, shifts
 * Branches/Jumps      | 1      | No branch prediction in template
 * LOAD (cache hit)    | 2      | Request + response
 * LOAD (cache miss)   | 10+    | Depends on memory hierarchy
 * STORE               | 2      | Write-through model
 * MUL/DIV             | 1      | Simplified, not pipelined
 * ```
 *
 * **Code Example - Complete Flow:**
 *
 * ```cpp
 * // 1. System initialization
 * CPU* cpu = new CPU("CPU0", soc);
 * ExecOneInstrEvent* execEvent = new ExecOneInstrEvent(0, cpu);
 *
 * // 2. Start simulation
 * scheduler.schedule(execEvent, 1);  // First instruction at tick 1
 *
 * // 3. Event fires at tick 1
 * void ExecOneInstrEvent::process() {
 *     this->cpu->execOneInstr();
 *     // CPU executes: ADD x1, x2, x3
 *     // Then schedules: execEvent at tick 2
 * }
 *
 * // 4. Event fires at tick 2
 * void ExecOneInstrEvent::process() {
 *     this->cpu->execOneInstr();
 *     // CPU executes: LW x4, 0(x5)
 *     // Creates MemReqEvent for tick 3
 *     // Schedules next execEvent at tick 4
 * }
 *
 * // 5. Continue until program end
 * ```
 *
 * **Educational Value:**
 *
 * This template is designed to teach:
 * - Event-driven simulation fundamentals
 * - Discrete-event scheduling concepts
 * - Object pooling for performance
 * - CPU-memory interaction timing
 * - Simulation tick-based modeling
 *
 * Students can understand these concepts without the complexity of
 * full pipeline modeling, making it an ideal learning platform.
 *
 * @see MemReqEvent For memory request event handling
 * @see CPU::execOneInstr() For complete instruction execution logic
 * @see acalsim::SimEvent For base event class interface
 * @see DataMemory For memory subsystem integration
 *
 * @note This is a template/educational version. For production simulations
 *       with accurate pipeline modeling, see src/riscv/libs/event/ExecOneInstrEvent.cc
 *
 * @author Playlab/ACAL
 * @version Template 1.0
 */

#include "event/ExecOneInstrEvent.hh"

#include "CPU.hh"

/**
 * @brief Constructs an ExecOneInstrEvent with CPU core identification
 *
 * @param _id Unique identifier for the CPU core (used for event naming)
 * @param _cpu Pointer to the CPU that will execute instructions
 *
 * @details
 * Creates a named event for instruction execution. The event name includes
 * the CPU ID to support multi-core simulations where each core has its own
 * execution event stream.
 *
 * Example event names:
 * - Core 0: "ExecOneInstrEvent0"
 * - Core 1: "ExecOneInstrEvent1"
 *
 * The constructor initializes the base SimEvent class with the unique name
 * and stores the CPU pointer for later use during event processing.
 *
 * **Object Pooling Support:**
 *
 * This constructor is typically called once per CPU core during initialization.
 * The same event object is then recycled using renew() for subsequent instructions,
 * avoiding repeated heap allocations.
 *
 * @note In riscvSimTemplate, typically one event per CPU core is sufficient
 *       since instructions execute sequentially (no instruction-level parallelism)
 *
 * @see renew() For event recycling and reuse
 * @see process() For event execution logic
 */
ExecOneInstrEvent::ExecOneInstrEvent(int _id, CPU* _cpu)
    : acalsim::SimEvent("ExecOneInstrEvent" + std::to_string(_id)), cpu(_cpu) {}

/**
 * @brief Renews the event for reuse with potentially different CPU
 *
 * @param _id CPU core identifier (for consistency checking)
 * @param _cpu Pointer to the CPU that will execute the next instruction
 *
 * @details
 * Implements object pooling by resetting the event state for reuse. This is
 * a critical performance optimization in discrete-event simulation, where
 * millions of events may be processed during a long-running simulation.
 *
 * **Recycling Process:**
 *
 * ```
 * Before renew():           After renew():
 * ┌─────────────────┐      ┌─────────────────┐
 * │ Event (used)    │      │ Event (fresh)   │
 * │ - scheduled: yes│      │ - scheduled: no │
 * │ - processed: yes│      │ - processed: no │
 * │ - cpu: old_ptr  │      │ - cpu: new_ptr  │
 * └─────────────────┘      └─────────────────┘
 * ```
 *
 * **Performance Impact:**
 *
 * ```
 * Without Object Pooling:        With Object Pooling (renew):
 * ┌───────────────────────┐     ┌───────────────────────┐
 * │ For each instruction: │     │ One-time allocation:  │
 * │  - new ExecEvent()    │     │  - new ExecEvent()    │
 * │  - schedule()         │     │ For each instruction: │
 * │  - process()          │     │  - event->renew()     │
 * │  - delete ExecEvent() │     │  - schedule()         │
 * │                       │     │  - process()          │
 * │ 10M instr = 10M alloc │     │ 10M instr = 1 alloc   │
 * └───────────────────────┘     └───────────────────────┘
 * ```
 *
 * **Usage Pattern:**
 *
 * ```cpp
 * // Initialize once
 * ExecOneInstrEvent* event = new ExecOneInstrEvent(0, cpu);
 * scheduler.schedule(event, 1);
 *
 * // After processing, reuse for next instruction
 * void CPU::scheduleNextInstruction() {
 *     event->renew(coreId, this);  // Reset for reuse
 *     scheduler.schedule(event, currentTick + cycles);
 * }
 * ```
 *
 * @note The base SimEvent::renew() clears scheduling state and flags
 * @note CPU pointer update allows event migration between cores (rarely used)
 *
 * @see ExecOneInstrEvent() For initial event construction
 * @see acalsim::SimEvent::renew() For base class reset logic
 */
void ExecOneInstrEvent::renew(int _id, CPU* _cpu) {
	this->SimEvent::renew();
	this->cpu = _cpu;
}

/**
 * @brief Processes the event by executing one instruction
 *
 * @details
 * This is the core event handler called automatically by the ACALSim event
 * scheduler when the simulation clock reaches the scheduled tick. It delegates
 * to CPU::execOneInstr() which performs the complete instruction execution.
 *
 * **Execution Flow:**
 *
 * ```
 * Simulation Loop                Event Processing              CPU Execution
 * ───────────────                ────────────────              ─────────────
 * while (!done) {
 *   tick = getNext()            ExecOneInstrEvent             CPU::execOneInstr()
 *   event = queue[tick] ─────→  .process() ─────────────────→ {
 *                                  │                             fetch();
 *                                  │                             decode();
 *                                  │                             execute();
 *                                  │                             memory();
 *                                  │                             writeback();
 *                                  │                             scheduleNext();
 *                                  ↓                           }
 *                               return
 * }
 * ```
 *
 * **Complete Instruction Processing:**
 *
 * Unlike the full riscv/ version with separate pipeline stages, this template
 * version executes the entire instruction within a single method call:
 *
 * ```
 * process() called at tick T
 *   └─→ cpu->execOneInstr()
 *       ├─ FETCH: Read instruction from imem[pc]
 *       ├─ DECODE: Determine operation and operands
 *       ├─ EXECUTE: Perform ALU operation or address calculation
 *       ├─ MEMORY: Issue load/store if needed (creates MemReqEvent)
 *       ├─ WRITEBACK: Update register file
 *       └─ SCHEDULE: Plan next ExecOneInstrEvent
 * ```
 *
 * **Self-Scheduling Pattern:**
 *
 * After executing an instruction, the CPU typically reschedules this same event
 * for the next instruction:
 *
 * ```cpp
 * void CPU::execOneInstr() {
 *     // ... execute current instruction ...
 *
 *     if (!halted) {
 *         // Schedule next instruction
 *         int latency = getInstructionLatency(inst);
 *         execEvent->renew(coreId, this);
 *         scheduler.schedule(execEvent, currentTick + latency);
 *     }
 * }
 * ```
 *
 * **Timing Accuracy:**
 *
 * The event-driven model ensures accurate timing simulation:
 * - Each instruction executes at its scheduled tick
 * - Memory latency properly delays dependent instructions
 * - Branch mispredictions affect timing (if modeled)
 * - Multi-cycle operations span multiple ticks
 *
 * **Error Handling:**
 *
 * The process() method itself performs no error checking; all validation
 * occurs within CPU::execOneInstr():
 * - Invalid instruction exceptions
 * - Memory access violations
 * - Privilege violations
 * - Breakpoint/watchpoint triggers
 *
 * @note This method is called by the simulation framework, not user code
 * @note The method has no return value; state changes are implicit
 * @note Re-scheduling for next instruction happens inside CPU::execOneInstr()
 *
 * @see CPU::execOneInstr() For complete instruction execution implementation
 * @see MemReqEvent For memory access event coordination
 * @see acalsim::SimEvent::process() For base class interface
 */
void ExecOneInstrEvent::process() { this->cpu->execOneInstr(); }
