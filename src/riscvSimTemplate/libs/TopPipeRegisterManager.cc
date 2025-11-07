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
 * @file TopPipeRegisterManager.cc
 * @brief Pipeline register manager implementation for riscvSimTemplate
 *
 * @details This file implements the TopPipeRegisterManager class, which manages all pipeline
 * registers in the RISC-V simulator. Pipeline registers are critical components that hold
 * intermediate values between pipeline stages, enabling pipelined execution.
 *
 * Pipeline Register Concept:
 * In a pipelined processor, each stage needs to pass data to the next stage. Pipeline registers
 * (also called interstage buffers or pipeline latches) serve this purpose:
 * - They separate different pipeline stages
 * - They store instruction state as it moves through the pipeline
 * - They synchronize on clock edges to advance the pipeline
 *
 * Typical Pipeline Register Contents:
 * - Instruction being processed
 * - Program counter value
 * - Computed values (ALU results, memory addresses)
 * - Control signals for subsequent stages
 * - Valid/bubble bits (indicating if stage has real work)
 *
 * Pipeline Register Flow in riscvSimTemplate:
 * ```
 * [IF Stage] ---> [IF/EXE Register] ---> [EXE Stage] ---> [EXE/WB Register] ---> [WB Stage]
 * ```
 *
 * The manager coordinates these registers:
 * - Maintains a map of all pipeline registers by name
 * - Synchronizes all registers on clock edges via runSyncPipeRegister()
 * - Supports stalling individual registers for hazard handling
 * - Manages register lifecycle (allocation and cleanup)
 *
 * Synchronization Process:
 * Pipeline registers use a two-phase update mechanism:
 * 1. Write Phase: Stages write new values to register "next" state
 * 2. Sync Phase: Manager calls sync() on all registers to commit changes
 *
 * This two-phase approach prevents race conditions where:
 * - Stage N reads from register R
 * - Stage N-1 writes to register R
 * If both happened simultaneously, Stage N might see partial updates.
 *
 * Pipeline Stalls:
 * The manager supports pipeline stalls through setPipeStallControl():
 * - Stalled registers maintain their current value (don't advance)
 * - Used for data hazards (RAW, WAR, WAW)
 * - Used for control hazards (branch mispredictions)
 * - Used for structural hazards (resource conflicts)
 *
 * Example Stall Scenario:
 * ```
 * Cycle 1: LW  x1, 0(x2)     ; Load from memory
 * Cycle 2: ADD x3, x1, x4    ; Needs x1 (data hazard!)
 * ```
 * The IF/EXE register must stall to wait for the load to complete.
 *
 * Template Design Philosophy:
 * This is the template version for educational purposes. It demonstrates the fundamental
 * pipeline register management without advanced features like:
 * - Multi-cycle operations with variable latency
 * - Out-of-order execution with reorder buffers
 * - Speculative execution with rollback support
 * - Performance counters for pipeline efficiency
 * These features may be present in the full src/riscv/ implementation.
 *
 * Integration with ACALSim:
 * - Inherits from PipeRegisterManagerBase for core functionality
 * - Uses HashableType for efficient lookup in simulation framework
 * - Registers are created by pipeline stages during initialization
 * - Manager is invoked by the main simulation loop each cycle
 *
 * Design Patterns:
 * - Manager Pattern: Centralized control of pipeline registers
 * - Template Method: Provides concrete implementation of abstract sync method
 * - Iterator Pattern: Iterates over all registers for synchronization
 *
 * @note Part of the riscvSimTemplate educational framework
 * @note This template uses a simplified pipeline with fewer stages than typical processors
 * @see TopPipeRegisterManager.hh for class interface documentation
 * @see PipeRegisterManagerBase for base class functionality
 * @see PipeRegister for individual register implementation
 *
 * @author Playlab/ACAL
 * @version 1.0
 * @date 2023-2025
 */

#include "TopPipeRegisterManager.hh"

/**
 * @brief Synchronizes all pipeline registers to advance the pipeline
 *
 * @details This method is the heart of the pipeline register manager. It is called once
 * per simulation cycle (typically at the rising edge of the clock) to synchronize all
 * pipeline registers and advance the pipeline state.
 *
 * Synchronization Process:
 * For each pipeline register in the manager:
 * 1. The sync() method is called on the register
 * 2. The register commits its "next" value to become its "current" value
 * 3. Unless the register has a stall flag set, in which case it maintains current value
 * 4. The register clears its stall flag for the next cycle
 *
 * Two-Phase Update Protocol:
 * This synchronization implements the second phase of a two-phase update:
 * - Phase 1 (During cycle): Stages compute and write to register "next" state
 * - Phase 2 (At cycle end): This method commits all "next" values atomically
 *
 * This ensures deterministic behavior where all stages see a consistent snapshot
 * of pipeline state at the start of each cycle.
 *
 * Iteration Details:
 * - Iterates over the registers map (name -> register* mapping)
 * - Uses structured binding [_, reg] to extract register pointer
 * - The underscore indicates we don't use the name in this loop
 * - Calls sync() on each register regardless of type or content
 *
 * Timing Considerations:
 * In the event-driven simulation:
 * - This method is scheduled at the end of each simulated clock cycle
 * - All stage processing for the cycle must complete before this is called
 * - After sync, registers hold the state for the next cycle
 * - Next cycle's stage processing can then begin
 *
 * Performance Implications:
 * - The number of registers is typically small (2-5 for simple pipelines)
 * - Iteration overhead is negligible
 * - Each sync() call is O(1) - just a value copy
 * - Overall complexity: O(n) where n is number of pipeline stages
 *
 * Stall Behavior:
 * If a register's stall flag is set (via setPipeStallControl):
 * - sync() will not advance the register
 * - The "current" value remains unchanged
 * - The "next" value is discarded
 * - This creates a pipeline bubble in subsequent stages
 *
 * Example Pipeline Advancement:
 * ```
 * Before sync():
 *   IF/EXE register: current = Inst1, next = Inst2
 *   EXE/WB register: current = Inst0, next = Inst1
 *
 * After sync():
 *   IF/EXE register: current = Inst2, next = (empty)
 *   EXE/WB register: current = Inst1, next = (empty)
 * ```
 *
 * Error Handling:
 * - No error checking needed - registers handle their own state
 * - Null pointers cannot occur (registers are created during initialization)
 * - Invalid register states are detected by the stages, not the manager
 *
 * @note This method must be called exactly once per simulated clock cycle
 * @note Call order relative to stage processing is critical for correctness
 * @note Registers automatically clear their stall flags after sync
 *
 * @see PipeRegister::sync() for individual register synchronization logic
 * @see setPipeStallControl() for stall control mechanism
 * @see PipeRegisterManagerBase for base class implementation
 */
void TopPipeRegisterManager::runSyncPipeRegister() {
	for (auto& [_, reg] : this->registers) { reg->sync(); }
}
