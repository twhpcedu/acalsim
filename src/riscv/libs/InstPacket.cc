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
 * @brief Implementation of instruction packet for RISC-V pipeline communication
 *
 * @details
 * This file implements the InstPacket class, which encapsulates a RISC-V instruction
 * and its associated metadata as it flows through the processor pipeline. InstPacket
 * serves as the primary communication mechanism between pipeline stages, carrying
 * both static instruction information (encoding, PC) and dynamic execution state
 * (branch decisions).
 *
 * <b>Instruction Packet Architecture:</b>
 *
 * InstPacket is a specialized SimPacket that travels through the processor pipeline,
 * visiting different pipeline stages in sequence. Each stage can:
 * - Read instruction information
 * - Update dynamic state (e.g., branch taken flag)
 * - Pass packet to next stage
 * - Recycle packet when complete
 *
 * <b>Packet Contents:</b>
 *
 * An InstPacket contains:
 * - <b>inst</b> (instr): The raw RISC-V instruction encoding
 * - <b>str</b> (std::string): Human-readable disassembly of the instruction
 * - <b>pc</b> (uint32_t): Program counter (address) of this instruction
 * - <b>isTakenBranch</b> (bool): Whether a branch was taken (updated during execution)
 *
 * The instruction encoding (inst) contains the decoded fields following RISC-V
 * instruction formats (R-type, I-type, S-type, B-type, U-type, J-type).
 *
 * <b>Pipeline Flow:</b>
 *
 * A typical instruction packet lifecycle:
 *
 * 1. <b>Instruction Fetch (IF):</b>
 *    - InstPacket created with instruction encoding and PC
 *    - Disassembly string generated for debugging
 *    - Packet sent to next stage via visit()
 *
 * 2. <b>Instruction Decode (ID):</b>
 *    - Instruction fields extracted
 *    - Operands identified
 *    - Control signals generated
 *    - Packet forwarded to execution stage
 *
 * 3. <b>Execute (EX):</b>
 *    - ALU operations performed
 *    - Branch conditions evaluated
 *    - isTakenBranch flag updated for control instructions
 *    - Memory addresses calculated
 *    - Packet sent to memory or writeback stage
 *
 * 4. <b>Memory Access (MEM):</b>
 *    - Load/store operations initiated (separate MemPackets)
 *    - Packet forwarded to writeback
 *
 * 5. <b>Writeback (WB):</b>
 *    - Results written to register file
 *    - Packet recycled or deallocated
 *
 * <b>Visitor Pattern Implementation:</b>
 *
 * InstPacket implements the visitor pattern through two visit() methods:
 *
 * - <b>visit(SimModule&):</b> For visiting hardware modules (currently not implemented)
 * - <b>visit(SimBase&):</b> For visiting pipeline stages (IF, EX, WB)
 *
 * The visit(SimBase&) method uses dynamic_cast to identify the target stage type
 * and dispatches to the appropriate handler:
 *
 * @code
 * void InstPacket::visit(Tick _when, SimBase& _simulator) {
 *     if (auto ifStage = dynamic_cast<IFStage*>(&_simulator)) {
 *         ifStage->instPacketHandler(_when, this);
 *     } else if (auto exStage = dynamic_cast<EXEStage*>(&_simulator)) {
 *         exStage->instPacketHandler(_when, this);
 *     } else if (auto wbStage = dynamic_cast<WBStage*>(&_simulator)) {
 *         wbStage->instPacketHandler(_when, this);
 *     } else {
 *         // Error: unknown stage type
 *     }
 * }
 * @endcode
 *
 * This pattern allows pipeline stages to define custom handling logic for
 * instruction packets without modifying the InstPacket class itself.
 *
 * <b>Branch Handling:</b>
 *
 * The isTakenBranch field tracks whether a conditional branch was taken:
 * - Initialized to false when packet is created
 * - Updated in the Execute stage based on branch condition evaluation
 * - Used by Fetch stage to update PC correctly
 *
 * Example branch decision flow:
 * @code
 * // In Execute stage
 * if (inst.type == BEQ || inst.type == BNE || ...) {
 *     bool condition = evaluateBranchCondition(inst);
 *     packet->isTakenBranch = condition;
 *     if (condition) {
 *         // Calculate and update branch target
 *     }
 * }
 * @endcode
 *
 * <b>Packet Recycling:</b>
 *
 * InstPacket objects are typically managed through an object pool (RecycleContainer)
 * to minimize allocation overhead. The renew() method reinitializes a recycled
 * packet with new instruction data:
 *
 * @code
 * // Get recycled packet from pool
 * InstPacket* pkt = recycleContainer->getPacket<InstPacket>();
 *
 * // Reinitialize with new instruction
 * instr new_inst = fetchedInstruction();
 * pkt->renew(new_inst);
 *
 * // Use packet...
 *
 * // Return to pool when done
 * recycleContainer->recycle(pkt);
 * @endcode
 *
 * This approach significantly reduces malloc/free overhead in the simulator's
 * critical path.
 *
 * <b>Instruction Disassembly:</b>
 *
 * The str field contains a human-readable representation of the instruction,
 * useful for:
 * - Debug output and logging
 * - Trace generation
 * - Instruction retirement tracking
 * - Performance analysis
 *
 * Example disassembly strings:
 * - "add x1, x2, x3"
 * - "lw x4, 100(x5)"
 * - "beq x6, x7, label"
 *
 * <b>Multi-Stage Processing:</b>
 *
 * Different pipeline stages access different packet fields:
 *
 * | Stage | Reads | Writes |
 * |-------|-------|--------|
 * | IF    | -     | inst, pc, str |
 * | ID    | inst, pc | - |
 * | EX    | inst  | isTakenBranch |
 * | MEM   | inst  | - |
 * | WB    | inst  | - |
 *
 * <b>Timing Simulation:</b>
 *
 * The _when parameter in visit() methods represents the simulation tick
 * when the packet arrives at a stage. This enables:
 * - Modeling pipeline latency
 * - Tracking instruction timing
 * - Identifying performance bottlenecks
 * - Simulating out-of-order execution
 *
 * <b>Usage Example:</b>
 *
 * @code
 * // Create instruction packet in Fetch stage
 * instr fetched_inst = decodeInstruction(memory[pc]);
 * InstPacket* pkt = new InstPacket(fetched_inst);
 * pkt->pc = pc;
 * pkt->str = disassemble(fetched_inst);
 *
 * // Send to next stage
 * sendToNextStage(pkt, currentTick + 1);
 *
 * // In Execute stage handler
 * void EXEStage::instPacketHandler(Tick when, InstPacket* pkt) {
 *     // Process instruction
 *     if (pkt->inst.type == ADD) {
 *         uint32_t result = regFile[pkt->inst.rs1] + regFile[pkt->inst.rs2];
 *         // Store result...
 *     }
 *
 *     // Forward to next stage
 *     sendToMemoryStage(pkt, when + 1);
 * }
 * @endcode
 *
 * <b>Extension Points:</b>
 *
 * InstPacket can be extended to support:
 * - Speculative execution tracking (speculation depth, recovery info)
 * - Performance counters (cycle counts, stall reasons)
 * - Dependency tracking (producer instructions)
 * - Exception information (fault address, cause)
 * - Cache information (hit/miss status)
 *
 * <b>Error Handling:</b>
 *
 * Invalid visitor types trigger CLASS_ERROR logging. In a production
 * simulator, this could:
 * - Throw exceptions
 * - Halt simulation
 * - Generate debug traces
 *
 * @see InstPacket.hh for class declaration
 * @see SimPacket for base packet class
 * @see IFStage, EXEStage, WBStage for pipeline stage implementations
 * @see DataStruct.hh for instruction encoding definitions
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 */

#include "InstPacket.hh"

#include "EXEStage.hh"
#include "IFStage.hh"
#include "WBStage.hh"

/**
 * @brief Visitor method for hardware modules (not implemented)
 *
 * @param _when Simulation tick when the visit occurs
 * @param _module Reference to the SimModule to visit
 *
 * @details
 * This visit() overload is intended for instruction packets visiting hardware
 * modules (as opposed to pipeline stages). Currently, this functionality is
 * not implemented, as InstPacket primarily communicates with pipeline stages
 * (SimBase-derived classes) rather than standalone modules.
 *
 * <b>Current Behavior:</b>
 * Logs an error message indicating the function is not implemented.
 *
 * <b>Future Implementation:</b>
 * Could be used to support:
 * - Memory controllers receiving instruction fetch requests
 * - Cache modules handling instruction cache operations
 * - Debug modules intercepting instructions for breakpoints
 * - Performance monitoring units tracking instruction execution
 *
 * @note This method currently triggers a CLASS_ERROR. Do not call unless
 *       implementing module-based instruction handling.
 *
 * @see visit(Tick, SimBase&) for the active visitor implementation
 */
void InstPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	CLASS_ERROR << "void InstPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visitor method for pipeline stages
 *
 * @param _when Simulation tick when the packet arrives at the stage
 * @param _simulator Reference to the pipeline stage (SimBase-derived)
 *
 * @details
 * This is the primary visitor method for InstPacket. It implements the visitor
 * pattern by using dynamic_cast to identify the target pipeline stage type and
 * dispatching to the appropriate handler method.
 *
 * <b>Supported Pipeline Stages:</b>
 *
 * - <b>IFStage (Instruction Fetch):</b>
 *   Handles instruction fetch completion, updating the packet with fetched
 *   instruction data and forwarding to the next stage.
 *
 * - <b>EXEStage (Execute):</b>
 *   Processes the instruction, performing ALU operations, branch evaluation,
 *   and updating the isTakenBranch flag for control-flow instructions.
 *
 * - <b>WBStage (Writeback):</b>
 *   Completes instruction execution by writing results to the register file
 *   and handling instruction retirement.
 *
 * <b>Dispatch Mechanism:</b>
 *
 * The method uses dynamic_cast to safely identify the stage type:
 * @code
 * if (auto ifStage = dynamic_cast<IFStage*>(&_simulator)) {
 *     // _simulator is an IFStage
 *     ifStage->instPacketHandler(_when, this);
 * }
 * @endcode
 *
 * dynamic_cast returns nullptr if the cast fails, allowing safe type checking
 * without throwing exceptions.
 *
 * <b>Handler Invocation:</b>
 *
 * Each pipeline stage implements an instPacketHandler() method:
 * @code
 * void IFStage::instPacketHandler(Tick when, InstPacket* pkt) {
 *     // Stage-specific processing
 * }
 * @endcode
 *
 * The packet (this) and arrival time (_when) are passed to the handler,
 * enabling the stage to process the instruction with timing information.
 *
 * <b>Error Handling:</b>
 *
 * If the simulator type doesn't match any known pipeline stage, a CLASS_ERROR
 * is logged. This indicates either:
 * - A programming error (wrong stage type)
 * - Missing stage implementation
 * - Incorrect packet routing
 *
 * In production code, this could trigger:
 * - Exception throwing
 * - Simulation halt
 * - Debug breakpoint
 *
 * <b>Pipeline Stage Sequencing:</b>
 *
 * Typical packet flow through stages:
 * 1. Created in IFStage after instruction fetch
 * 2. Routed to EXEStage via visit()
 * 3. Processed in EXEStage, potentially generating MemPackets
 * 4. Routed to WBStage via visit()
 * 5. Completed in WBStage and recycled
 *
 * @note The order of dynamic_cast checks affects performance minimally but
 *       could be optimized based on expected packet distribution (e.g.,
 *       check most frequent stage first).
 *
 * @warning If an unknown simulator type is passed, an error is logged but
 *          execution continues. This may lead to packets being dropped or
 *          simulation inconsistency.
 *
 * @pre _simulator must reference a valid SimBase-derived pipeline stage
 * @post The appropriate stage handler is invoked with the packet
 *
 * @see IFStage::instPacketHandler() for fetch stage handling
 * @see EXEStage::instPacketHandler() for execute stage handling
 * @see WBStage::instPacketHandler() for writeback stage handling
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
