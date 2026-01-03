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
 * @file InstPacket.cc
 * @brief Instruction packet implementation for pipeline communication in riscvSimTemplate
 *
 * @details This file implements the InstPacket class, which represents instructions as they
 * flow through the RISC-V pipeline. InstPacket is a fundamental data structure that carries
 * instruction information between pipeline stages using the visitor pattern for routing.
 *
 * Packet Contents:
 * An InstPacket encapsulates:
 * - inst: The decoded instruction structure containing opcode and operands
 * - pc: Program counter value for this instruction
 * - isTakenBranch: Branch prediction/resolution state
 *
 * Pipeline Flow:
 * The InstPacket travels through the simplified pipeline stages:
 * 1. IFStage (Instruction Fetch): Creates packet with fetched instruction
 * 2. EXEStage (Execute): Processes instruction, updates branch information
 * 3. WBStage (Write Back): Commits results to register file
 *
 * Visitor Pattern Implementation:
 * InstPacket uses the visitor pattern for type-safe routing to different pipeline stages.
 * The visit() method performs dynamic type checking to dispatch the packet to the
 * appropriate handler in each pipeline stage. This design allows:
 * - Type-safe packet handling without explicit downcasting in user code
 * - Centralized routing logic
 * - Easy extension to new pipeline stages
 *
 * Packet Lifecycle:
 * 1. Creation: Allocated from packet pool when instruction is fetched
 * 2. Transfer: Passed between pipeline stages via SimBase communication
 * 3. Modification: Updated as it flows through stages (e.g., branch resolution)
 * 4. Recycling: Returned to packet pool after writeback for reuse
 *
 * Template Design Philosophy:
 * This is the template version for educational purposes. It demonstrates the packet-based
 * communication model used in event-driven simulation. The full src/riscv/ implementation
 * may include additional fields for:
 * - Out-of-order execution metadata
 * - Speculative execution tracking
 * - Performance monitoring counters
 * - Exception/interrupt information
 *
 * Design Rationale:
 * The packet-based approach (vs. direct method calls) enables:
 * - Event-driven simulation with timing accuracy
 * - Pipeline stalls and bubbles
 * - Asynchronous communication between stages
 * - Memory efficiency through packet pooling
 *
 * @note Part of the riscvSimTemplate educational framework
 * @note This template uses a simplified 3-stage pipeline (IF, EXE, WB)
 * @see InstPacket.hh for class interface and member documentation
 * @see SimPacket base class for packet infrastructure
 * @see IFStage, EXEStage, WBStage for pipeline stage implementations
 *
 * @author Playlab/ACAL
 * @version 1.0
 * @date 2023-2025
 */

#include "InstPacket.hh"

#include "EXEStage.hh"
#include "IFStage.hh"
#include "WBStage.hh"

/**
 * @brief Visitor method for SimModule routing (not implemented in template)
 *
 * @param _when Simulation tick when the visit occurs
 * @param _module The SimModule to visit
 *
 * @details This visitor overload handles routing to SimModule-derived classes.
 * In the riscvSimTemplate, instruction packets are routed to SimBase-derived
 * pipeline stages, not SimModules. This method is provided for interface
 * compatibility but is not used in the template implementation.
 *
 * SimModule vs SimBase:
 * - SimModule: Typically used for passive components (e.g., memory, caches)
 * - SimBase: Used for active simulation components (e.g., pipeline stages)
 *
 * In this template, InstPackets are handled by SimBase-derived stages,
 * while memory packets (MemReadReqPacket, MemWriteReqPacket) use SimModule
 * for routing to DataMemory.
 *
 * @throws Triggers CLASS_ERROR if called, as this code path is not implemented
 *
 * @note The full implementation may support routing to additional module types
 * @see visit(Tick, SimBase&) for the active visitor implementation
 */
void InstPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	CLASS_ERROR << "void InstPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visitor method for SimBase routing to pipeline stages
 *
 * @param _when Simulation tick when the visit occurs (used for timing-accurate simulation)
 * @param _simulator Reference to the SimBase-derived pipeline stage
 *
 * @details This is the primary visitor implementation that routes InstPackets to the
 * appropriate pipeline stage handler. It uses dynamic_cast to determine the runtime
 * type of the simulator and dispatches to the corresponding handler.
 *
 * Supported Pipeline Stages:
 * 1. IFStage (Instruction Fetch):
 *    - Receives packets for branch target updates
 *    - May handle instruction fetch completions
 *
 * 2. EXEStage (Execute):
 *    - Receives packets from IF stage
 *    - Executes instruction logic
 *    - Updates branch prediction state
 *    - Forwards results to WB stage
 *
 * 3. WBStage (Write Back):
 *    - Receives packets from EXE stage
 *    - Commits results to architectural state
 *    - Recycles packet back to pool
 *
 * Routing Mechanism:
 * The method performs sequential dynamic_cast checks:
 * - First checks if simulator is an IFStage
 * - Then checks for EXEStage
 * - Then checks for WBStage
 * - Triggers error if none match
 *
 * This design allows the same packet type to be handled differently by different stages,
 * implementing stage-specific behavior while maintaining a uniform packet interface.
 *
 * Timing Semantics:
 * The _when parameter represents the simulation time when this packet arrives at the
 * stage. In an event-driven simulator:
 * - Packets may arrive at different times due to pipeline latencies
 * - Each stage can model its processing delay
 * - The timing affects when results propagate to the next stage
 *
 * Error Handling:
 * If the simulator type doesn't match any known pipeline stage, CLASS_ERROR is
 * triggered. This catches programming errors where packets are sent to incorrect
 * destinations.
 *
 * @note The order of dynamic_cast checks may affect performance for hot paths
 * @note Each stage's handler (instPacketHandler) is responsible for further processing
 * @warning Adding new pipeline stages requires updating this method
 *
 * @see IFStage::instPacketHandler() for IF-specific packet handling
 * @see EXEStage::instPacketHandler() for EXE-specific packet handling
 * @see WBStage::instPacketHandler() for WB-specific packet handling
 */
void InstPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	if (auto sim = dynamic_cast<IFStage*>(&_simulator)) {
		sim->instPacketHandler(_when, this);
	} else if (auto sim = dynamic_cast<EXEStage*>(&_simulator)) {
		sim->instPacketHandler(_when, this);
	} else if (auto sim = dynamic_cast<WBStage*>(&_simulator)) {
		sim->instPacketHandler(_when, this);
	} else {
		CLASS_ERROR << "Invalid simulator type!";
	}
}
