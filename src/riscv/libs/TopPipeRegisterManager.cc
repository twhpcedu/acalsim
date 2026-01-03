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
 * @file TopPipeRegisterManager.cc
 * @brief Implementation of pipeline register manager for RISC-V processor simulation
 *
 * @details
 * This file implements the TopPipeRegisterManager class, which manages the synchronization
 * and state updates of pipeline registers in the RISC-V processor simulator. Pipeline
 * registers are the latches between pipeline stages that hold intermediate instruction
 * state as instructions flow through the pipeline.
 *
 * <b>Pipeline Register Concept:</b>
 *
 * In a pipelined processor, pipeline registers separate adjacent pipeline stages:
 *
 * @code
 * [IF Stage] --> [IF/ID Reg] --> [ID Stage] --> [ID/EX Reg] --> [EX Stage] -->
 *                                                  [EX/MEM Reg] --> [MEM Stage] -->
 *                                                  [MEM/WB Reg] --> [WB Stage]
 * @endcode
 *
 * Each pipeline register:
 * - Captures outputs from the previous stage (input side)
 * - Holds values stable for the next stage to read (output side)
 * - Updates on clock edge (synchronization point)
 * - Can be stalled to freeze pipeline advancement
 *
 * <b>Register Manager Responsibilities:</b>
 *
 * The TopPipeRegisterManager coordinates all pipeline registers to:
 * 1. Synchronize register updates at clock boundaries
 * 2. Propagate stall signals across pipeline stages
 * 3. Handle pipeline flushes (e.g., for branch misprediction)
 * 4. Maintain pipeline state consistency
 * 5. Support debug and introspection
 *
 * <b>Synchronization Model:</b>
 *
 * The runSyncPipeRegister() method implements a two-phase update protocol:
 *
 * Phase 1: All stages compute next values (combinational logic)
 * - Stages read from register output sides
 * - Stages write to register input sides
 * - No state changes occur yet
 *
 * Phase 2: All registers synchronize simultaneously
 * - Input values are transferred to outputs atomically
 * - This simulates the rising clock edge
 * - All pipeline registers update "at the same time"
 *
 * @code
 * // Simplified synchronization sequence
 * void runSyncPipeRegister() {
 *     // Phase 1: Stages have already computed new values
 *
 *     // Phase 2: All registers update simultaneously
 *     for (each register in pipeline) {
 *         register.sync();  // input -> output
 *     }
 * }
 * @endcode
 *
 * This ensures correct pipeline semantics where all stages advance together
 * on each clock cycle.
 *
 * <b>Stall Propagation:</b>
 *
 * Pipeline stalls occur when a stage cannot proceed, requiring earlier stages
 * to freeze. Common stall causes:
 * - Data hazards (waiting for previous instruction result)
 * - Structural hazards (resource conflicts)
 * - Memory access delays
 * - Cache misses
 *
 * The manager handles stall propagation:
 * @code
 * // Stall example: MEM stage stalled on cache miss
 * // Effect on pipeline:
 * // IF: Stalled (cannot advance new instructions)
 * // ID: Stalled (cannot accept instruction from IF)
 * // EX: Stalled (cannot send instruction to MEM)
 * // MEM: Stalled (waiting for cache)
 * // WB: May proceed if independent instruction present
 * @endcode
 *
 * After synchronization, stall flags are cleared to prepare for the next cycle:
 * @code
 * for (each register) {
 *     register.clearStallFlag();  // Reset for next cycle
 * }
 * @endcode
 *
 * <b>Register Storage:</b>
 *
 * Pipeline registers are stored in an unordered_map:
 * @code
 * std::unordered_map<key_type, PipeRegisterBase*> registers;
 * @endcode
 *
 * This allows:
 * - Dynamic registration of pipeline registers
 * - Name-based lookup for debugging
 * - Efficient iteration during synchronization
 * - Flexible pipeline configuration
 *
 * <b>Typical Pipeline Register Contents:</b>
 *
 * - <b>IF/ID Register:</b>
 *   - Fetched instruction
 *   - Program counter (PC)
 *   - Exception information
 *   - Valid bit
 *
 * - <b>ID/EX Register:</b>
 *   - Decoded instruction
 *   - Source operand values
 *   - Immediate values
 *   - Control signals
 *   - Destination register ID
 *
 * - <b>EX/MEM Register:</b>
 *   - ALU result
 *   - Memory address (for load/store)
 *   - Store data
 *   - Destination register ID
 *   - Control signals
 *
 * - <b>MEM/WB Register:</b>
 *   - Memory read data or ALU result
 *   - Destination register ID
 *   - Writeback control signal
 *
 * <b>Clock Cycle Timing:</b>
 *
 * In discrete-event simulation, the clock is modeled as periodic events:
 * @code
 * // Each cycle:
 * 1. Execute combinational logic in all stages
 * 2. Call runSyncPipeRegister() to update all pipeline registers
 * 3. Schedule next clock event
 * 4. Continue simulation
 * @endcode
 *
 * The manager ensures atomic updates so that:
 * - No stage sees partial updates
 * - All stages observe consistent state
 * - Pipeline invariants are maintained
 *
 * <b>Customization Points:</b>
 *
 * The implementation includes a comment indicating where custom logic can be added:
 * @code
 * // you may do customized logic here. For example stall propogation
 * @endcode
 *
 * Possible customizations:
 * - Complex stall propagation rules
 * - Pipeline flushing logic
 * - Debug instrumentation
 * - Performance counter updates
 * - State snapshots for checkpointing
 *
 * <b>Flush Operations:</b>
 *
 * Pipeline flushes invalidate instructions in the pipeline (e.g., after branch
 * misprediction). While not shown in this basic implementation, flushes would:
 * 1. Clear valid bits in pipeline registers
 * 2. Prevent stalled instructions from advancing
 * 3. Insert bubbles (NOPs) into the pipeline
 * 4. Restart fetch from correct PC
 *
 * <b>Usage Example:</b>
 *
 * @code
 * // Create pipeline register manager
 * TopPipeRegisterManager* regMgr = new TopPipeRegisterManager("PipeMgr");
 *
 * // Register pipeline stages would add their registers:
 * // ifStage->registerPipeRegister(regMgr, "IF/ID");
 * // idStage->registerPipeRegister(regMgr, "ID/EX");
 * // etc.
 *
 * // In simulation loop (each clock cycle):
 * while (simulating) {
 *     // 1. Execute all stage logic
 *     ifStage->execute(currentTick);
 *     idStage->execute(currentTick);
 *     exStage->execute(currentTick);
 *     memStage->execute(currentTick);
 *     wbStage->execute(currentTick);
 *
 *     // 2. Synchronize all pipeline registers
 *     regMgr->runSyncPipeRegister();
 *
 *     // 3. Advance simulation time
 *     currentTick++;
 * }
 * @endcode
 *
 * <b>Performance Considerations:</b>
 *
 * - Iteration over all registers is O(n) where n is number of registers
 * - Typically only 4-5 registers in a basic 5-stage pipeline
 * - Unordered_map provides O(1) average lookup for debugging
 * - Synchronization overhead is minimal compared to stage logic
 *
 * <b>Debugging Support:</b>
 *
 * The manager can support debugging features:
 * - Inspecting register contents at any cycle
 * - Breakpoints on register value changes
 * - Trace generation for pipeline visualization
 * - State dumps for failure analysis
 *
 * <b>Correctness Properties:</b>
 *
 * The synchronization protocol maintains these invariants:
 * - All registers update atomically (appear simultaneous)
 * - No combinational loops through registers
 * - Pipeline state is consistent at sync points
 * - Stall propagation is race-free
 *
 * <b>Extension Points:</b>
 *
 * Future enhancements could include:
 * - Multi-cycle register updates (for complex stages)
 * - Register file port conflict detection
 * - Pipeline stage power modeling
 * - Cycle-accurate timing validation
 * - Support for superscalar pipelines
 * - Dynamic pipeline reconfiguration
 *
 * @see TopPipeRegisterManager.hh for class declaration
 * @see PipeRegisterManagerBase for base class interface
 * @see ACALSim.hh for simulation framework
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 */

#include "TopPipeRegisterManager.hh"

/**
 * @brief Synchronizes all pipeline registers on clock edge
 *
 * @details
 * This method implements the pipeline register synchronization that occurs on
 * each clock cycle. It performs two critical operations for each register:
 *
 * 1. <b>sync():</b> Transfers input values to output values, simulating the
 *    clock edge where pipeline registers latch new values. This is the
 *    fundamental operation that advances the pipeline.
 *
 * 2. <b>clearStallFlag():</b> Resets the stall flag for the next cycle.
 *    Stall conditions must be re-evaluated each cycle, so flags are cleared
 *    after synchronization.
 *
 * <b>Synchronization Semantics:</b>
 *
 * The loop iterates over all registered pipeline registers and synchronizes
 * them "simultaneously" (from the architectural perspective). This ensures:
 * - All stages see consistent input values
 * - Updates appear atomic to pipeline logic
 * - No race conditions between stages
 *
 * <b>Execution Flow:</b>
 *
 * @code
 * // Before sync: registers hold previous cycle's values
 * // Pipeline stages have computed new values on input side
 *
 * runSyncPipeRegister();
 *
 * // After sync: registers hold new values
 * // Input sides are ready for next cycle's computation
 * // Stall flags are cleared
 * @endcode
 *
 * <b>Stall Handling:</b>
 *
 * If a register was stalled during this cycle:
 * - sync() will NOT update the output (values remain unchanged)
 * - The pipeline stage effectively "pauses"
 * - clearStallFlag() still executes (stall must be reasserted if still needed)
 *
 * <b>Customization Point:</b>
 *
 * The implementation includes a comment suggesting where custom logic can be
 * added. Potential customizations:
 *
 * - <b>Stall Propagation:</b>
 *   @code
 *   // Propagate stalls backward through pipeline
 *   if (exMemReg->isStalled()) {
 *       idExReg->stall();
 *       ifIdReg->stall();
 *   }
 *   @endcode
 *
 * - <b>Flush Logic:</b>
 *   @code
 *   // Clear pipeline on branch misprediction
 *   if (branchMispredicted) {
 *       for (auto& [_, reg] : registers) {
 *           reg->invalidate();
 *       }
 *   }
 *   @endcode
 *
 * - <b>Performance Monitoring:</b>
 *   @code
 *   // Count stall cycles
 *   for (auto& [_, reg] : registers) {
 *       if (reg->isStalled()) {
 *           stallCycleCounter++;
 *       }
 *   }
 *   @endcode
 *
 * <b>Iteration Order:</b>
 *
 * The structured binding `auto& [_, reg]` iterates through the unordered_map:
 * - `_` is the key (register name), unused in this implementation
 * - `reg` is the PipeRegisterBase* pointer to the pipeline register
 *
 * The iteration order is unspecified (unordered_map), but this doesn't matter
 * because synchronization should be order-independent. All registers should
 * appear to update simultaneously.
 *
 * <b>Example Pipeline State Transition:</b>
 *
 * @code
 * Cycle N:
 *   IF/ID: [Instruction A]
 *   ID/EX: [Instruction B]
 *   EX/MEM: [Instruction C]
 *   MEM/WB: [Instruction D]
 *
 * runSyncPipeRegister() executes...
 *
 * Cycle N+1:
 *   IF/ID: [Instruction E] (newly fetched)
 *   ID/EX: [Instruction A] (advanced from IF/ID)
 *   EX/MEM: [Instruction B] (advanced from ID/EX)
 *   MEM/WB: [Instruction C] (advanced from EX/MEM)
 *   (Instruction D has been retired)
 * @endcode
 *
 * <b>Thread Safety:</b>
 *
 * This method is not thread-safe. The simulator is single-threaded and
 * event-driven, so no synchronization primitives are needed.
 *
 * @note This method is called once per clock cycle by the simulation kernel
 * @note All pipeline stages must complete their computation before this is called
 *
 * @post All pipeline registers have synchronized their state
 * @post All stall flags are cleared
 *
 * @see PipeRegisterBase::sync() for individual register synchronization
 * @see PipeRegisterBase::clearStallFlag() for stall flag management
 */
void TopPipeRegisterManager::runSyncPipeRegister() {
	// you may do customized logic here. For example stall propogation
	for (auto& [_, reg] : this->registers) {
		reg->sync();
		reg->clearStallFlag();
	}
}
