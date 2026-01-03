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
 * @file MemPacket.cc
 * @brief Memory packet implementation for data memory transactions in riscvSimTemplate
 *
 * @details This file implements four packet types that handle communication between
 * pipeline stages and the data memory subsystem. These packets implement the request-response
 * pattern for memory operations, enabling asynchronous memory transactions in the simulator.
 *
 * Packet Types:
 * 1. MemReadReqPacket: Request to read data from memory (load instructions)
 * 2. MemReadRespPacket: Response containing data read from memory
 * 3. MemWriteReqPacket: Request to write data to memory (store instructions)
 * 4. MemWriteRespPacket: Response confirming write completion
 *
 * Request-Response Pattern:
 * The memory system uses an asynchronous request-response model:
 * ```
 * Pipeline Stage                    DataMemory
 *      |                                 |
 *      |---MemReadReqPacket------------->|
 *      |                                 | (performs read)
 *      |<--MemReadRespPacket-------------|
 *      |                                 |
 *      |---MemWriteReqPacket------------>|
 *      |                                 | (performs write)
 *      |<--MemWriteRespPacket------------|
 * ```
 *
 * Callback Mechanism:
 * Request packets include callback functions that are invoked when the response is ready.
 * This allows the requesting stage to handle the response asynchronously:
 * - Callback receives the response packet as a parameter
 * - Stage can update its state based on the response
 * - Enables modeling of variable-latency memory operations
 *
 * Packet Recycling:
 * All packets support the renew() method for efficient reuse:
 * - Packets are allocated from a pool, not via new/delete
 * - renew() resets the packet to a clean state with new parameters
 * - After use, packets are returned to the pool via RecycleContainer
 * - This reduces allocation overhead in simulation
 *
 * Visitor Pattern:
 * Memory packets use the visitor pattern for routing:
 * - Request packets visit DataMemory (SimModule) for processing
 * - Response packets visit pipeline stages (SimBase) for handling
 * - Type-safe dispatch without manual downcasting
 *
 * Template Design Philosophy:
 * This is the template version for educational purposes. It demonstrates the packet-based
 * memory interface without advanced features like:
 * - Cache coherence protocols
 * - Memory ordering constraints
 * - Variable latency modeling
 * - Outstanding request tracking
 * These features may be present in the full src/riscv/ implementation.
 *
 * Supported RISC-V Operations:
 * - Load: LB, LBU, LH, LHU, LW (via MemReadReqPacket)
 * - Store: SB, SH, SW (via MemWriteReqPacket)
 *
 * Integration Points:
 * - MEMStage: Creates request packets, handles response packets
 * - DataMemory: Receives request packets, generates response packets
 * - RecycleContainer: Manages packet lifecycle and memory efficiency
 *
 * @note Part of the riscvSimTemplate educational framework
 * @note In this template, memory operations complete synchronously (single cycle)
 * @see MemPacket.hh for packet class interfaces and member documentation
 * @see DataMemory for memory request handling implementation
 * @see InstPacket for instruction packet implementation
 *
 * @author Playlab/ACAL
 * @version 1.0
 * @date 2023-2025
 */

#include "MemPacket.hh"

#include "DataMemory.hh"

/**
 * @brief Renews a MemReadRespPacket with new response data
 *
 * @param _i The instruction that requested this read
 * @param _op The operation type (LB, LBU, LH, LHU, LW)
 * @param _data The data read from memory (already sign/zero extended)
 * @param _a1 The destination operand (register to write to)
 *
 * @details This method resets the packet to a clean state and initializes it with
 * new response data. It's called when reusing a packet from the pool rather than
 * allocating a new one.
 *
 * The renew process:
 * 1. Calls parent SimPacket::renew() to reset base packet state
 * 2. Sets instruction information for tracking
 * 3. Sets operation type for proper handling
 * 4. Sets the data value (already processed by DataMemory)
 * 5. Sets destination operand for writeback
 *
 * This packet flows back to the pipeline stage that requested the read, where
 * the data will be written to the register file or forwarded to dependent instructions.
 *
 * @note The data is already sign/zero extended by DataMemory based on operation type
 * @see MemReadReqPacket for the corresponding request packet
 */
void MemReadRespPacket::renew(const instr& _i, instr_type _op, uint32_t _data, operand _a1) {
	this->acalsim::SimPacket::renew();
	this->i    = _i;
	this->op   = _op;
	this->data = _data;
	this->a1   = _a1;
}

/**
 * @brief Renews a MemReadReqPacket with new request parameters
 *
 * @param _callback Function to call when the read completes (receives MemReadRespPacket*)
 * @param _i The instruction requesting the read
 * @param _op The operation type (LB, LBU, LH, LHU, LW)
 * @param _addr The memory address to read from
 * @param _a1 The destination operand (register to write to)
 *
 * @details This method resets the packet and initializes it with parameters for a new
 * memory read request. The packet is typically allocated from a pool and reused across
 * multiple memory operations.
 *
 * The renew process:
 * 1. Calls parent SimPacket::renew() to reset base state
 * 2. Sets callback for asynchronous response handling
 * 3. Sets instruction information for tracking
 * 4. Sets operation type (determines read size and extension)
 * 5. Sets memory address to read from
 * 6. Sets destination operand for writeback
 *
 * Callback Pattern:
 * The callback allows the requesting stage to handle the response when ready:
 * ```cpp
 * auto callback = [this](MemReadRespPacket* resp) {
 *     uint32_t data = resp->getData();
 *     // Update register file or forward data
 * };
 * packet->renew(callback, instr, LW, addr, dest_reg);
 * ```
 *
 * After renewal, this packet is sent to DataMemory via the visitor pattern.
 *
 * @note In the template, memory operations are synchronous, but the callback structure
 *       supports future extensions with realistic latency modeling
 * @see DataMemory::memReadReqHandler() for request processing
 */
void MemReadReqPacket::renew(std::function<void(MemReadRespPacket*)> _callback, const instr& _i, instr_type _op,
                             uint32_t _addr, operand _a1) {
	this->acalsim::SimPacket::renew();
	this->callback = _callback;
	this->i        = _i;
	this->op       = _op;
	this->addr     = _addr;
	this->a1       = _a1;
}

/**
 * @brief Visitor method for routing read requests to DataMemory
 *
 * @param _when Simulation tick when the request arrives
 * @param _module The SimModule to visit (expected to be DataMemory)
 *
 * @details This is the primary visitor implementation for read requests. It performs
 * dynamic type checking to ensure the packet is routed to DataMemory, then invokes
 * the appropriate handler.
 *
 * Routing Logic:
 * - Attempts to cast _module to DataMemory*
 * - If successful, calls memReadReqHandler() to process the request
 * - If cast fails, triggers CLASS_ERROR (programming error)
 *
 * Memory Request Processing:
 * Once routed to DataMemory:
 * 1. DataMemory extracts address and operation type
 * 2. Determines read size (1, 2, or 4 bytes)
 * 3. Reads data from BaseMemory
 * 4. Applies sign/zero extension based on operation
 * 5. Returns data synchronously (in template version)
 * 6. Recycles this request packet
 *
 * @note This visitor targets SimModule (DataMemory) not SimBase (pipeline stages)
 * @see DataMemory::memReadReqHandler() for the actual request processing logic
 */
void MemReadReqPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	if (auto dm = dynamic_cast<DataMemory*>(&_module)) {
		dm->memReadReqHandler(_when, this);
	} else {
		CLASS_ERROR << "Invalid module type!";
	}
}

/**
 * @brief Visitor method for SimBase routing (not implemented for read requests)
 *
 * @param _when Simulation tick when the visit occurs
 * @param _simulator The SimBase to visit
 *
 * @details Read request packets are designed to visit SimModule (DataMemory) not
 * SimBase (pipeline stages). This overload exists for interface completeness but
 * should never be called in normal operation.
 *
 * If this method is called, it indicates a routing error where a read request
 * was sent to a pipeline stage instead of to memory.
 *
 * @throws Triggers CLASS_ERROR if called
 * @note Response packets (MemReadRespPacket) may use SimBase routing in future versions
 */
void MemReadReqPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	CLASS_ERROR << "void MemReadReqPacket::visit (SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Renews a MemWriteRespPacket with confirmation data
 *
 * @param _i The instruction that requested this write
 *
 * @details This method resets the packet and initializes it with confirmation that
 * a write operation has completed. Write responses are simpler than read responses
 * because they don't carry data - only confirmation of completion.
 *
 * The renew process:
 * 1. Calls parent SimPacket::renew() to reset base state
 * 2. Sets instruction information for tracking
 *
 * This packet flows back to the pipeline stage that requested the write, typically
 * to allow the next operation to proceed or to update performance counters.
 *
 * @note In the template version, writes are synchronous so this response is immediate
 * @see MemWriteReqPacket for the corresponding request packet
 */
void MemWriteRespPacket::renew(const instr& _i) {
	this->acalsim::SimPacket::renew();
	this->i = _i;
}

/**
 * @brief Renews a MemWriteReqPacket with new write parameters
 *
 * @param _callback Function to call when the write completes (receives MemWriteRespPacket*)
 * @param _i The instruction requesting the write
 * @param _op The operation type (SB, SH, SW)
 * @param _addr The memory address to write to
 * @param _data The data to write (will be truncated based on operation type)
 *
 * @details This method resets the packet and initializes it with parameters for a new
 * memory write request. The packet is typically allocated from a pool and reused.
 *
 * The renew process:
 * 1. Calls parent SimPacket::renew() to reset base state
 * 2. Sets callback for asynchronous completion handling
 * 3. Sets instruction information for tracking
 * 4. Sets operation type (determines write size: SB=1, SH=2, SW=4 bytes)
 * 5. Sets memory address to write to
 * 6. Sets data value (DataMemory will truncate as needed)
 *
 * Data Truncation:
 * The full 32-bit data value is stored in the packet. DataMemory will extract
 * the appropriate number of bytes:
 * - SB: least significant byte (bits 7:0)
 * - SH: least significant halfword (bits 15:0)
 * - SW: full word (bits 31:0)
 *
 * Callback Pattern:
 * The callback allows the requesting stage to handle write completion:
 * ```cpp
 * auto callback = [this](MemWriteRespPacket* resp) {
 *     // Write completed, can proceed with next operation
 * };
 * packet->renew(callback, instr, SW, addr, data);
 * ```
 *
 * @note In the template, writes are synchronous, but the callback structure supports
 *       future extensions with realistic latency modeling
 * @see DataMemory::memWriteReqHandler() for request processing
 */
void MemWriteReqPacket::renew(std::function<void(MemWriteRespPacket*)> _callback, const instr& _i, instr_type _op,
                              uint32_t _addr, uint32_t _data) {
	this->acalsim::SimPacket::renew();
	this->callback = _callback;
	this->i        = _i;
	this->op       = _op;
	this->addr     = _addr;
	this->data     = _data;
}

/**
 * @brief Visitor method for routing write requests to DataMemory
 *
 * @param _when Simulation tick when the request arrives
 * @param _module The SimModule to visit (expected to be DataMemory)
 *
 * @details This is the primary visitor implementation for write requests. It performs
 * dynamic type checking to ensure the packet is routed to DataMemory, then invokes
 * the appropriate handler.
 *
 * Routing Logic:
 * - Attempts to cast _module to DataMemory*
 * - If successful, calls memWriteReqHandler() to process the request
 * - If cast fails, triggers CLASS_ERROR (programming error)
 *
 * Memory Write Processing:
 * Once routed to DataMemory:
 * 1. DataMemory extracts address, data, and operation type
 * 2. Determines write size (1, 2, or 4 bytes)
 * 3. Truncates data to appropriate size
 * 4. Writes to BaseMemory
 * 5. Recycles this request packet
 * 6. (In async version) Would invoke callback with response
 *
 * @note This visitor targets SimModule (DataMemory) not SimBase (pipeline stages)
 * @see DataMemory::memWriteReqHandler() for the actual request processing logic
 */
void MemWriteReqPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	if (auto dm = dynamic_cast<DataMemory*>(&_module)) {
		dm->memWriteReqHandler(_when, this);
	} else {
		CLASS_ERROR << "Invalid module type!";
	}
}

/**
 * @brief Visitor method for SimBase routing (not implemented for write requests)
 *
 * @param _when Simulation tick when the visit occurs
 * @param _simulator The SimBase to visit
 *
 * @details Write request packets are designed to visit SimModule (DataMemory) not
 * SimBase (pipeline stages). This overload exists for interface completeness but
 * should never be called in normal operation.
 *
 * If this method is called, it indicates a routing error where a write request
 * was sent to a pipeline stage instead of to memory.
 *
 * @throws Triggers CLASS_ERROR if called
 * @note Response packets (MemWriteRespPacket) may use SimBase routing in future versions
 */
void MemWriteReqPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	CLASS_ERROR << "void MemWriteReqPacket::visit (SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Visitor method for SimModule routing (not implemented for read responses)
 *
 * @param _when Simulation tick when the visit occurs
 * @param _module The SimModule to visit
 *
 * @details Read response packets flow back to pipeline stages (SimBase), not to
 * SimModule components. This overload exists for interface completeness but is
 * not used in the current implementation.
 *
 * In the template version, read responses are handled synchronously through
 * direct return values rather than through packet routing.
 *
 * @throws Triggers CLASS_ERROR if called
 * @note The full implementation may use response packet routing for async memory
 */
void MemReadRespPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	CLASS_ERROR << "void MemReadRespPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visitor method for SimBase routing (not implemented for read responses)
 *
 * @param _when Simulation tick when the visit occurs
 * @param _simulator The SimBase to visit
 *
 * @details In a fully asynchronous memory system, read responses would be routed back
 * to the requesting pipeline stage via this visitor. However, in the template version,
 * memory operations are synchronous and handled through callbacks rather than packet routing.
 *
 * Future extensions with multi-cycle memory operations may implement this method to
 * route responses to stages like MEMStage or EXEStage.
 *
 * @throws Triggers CLASS_ERROR if called
 * @note Template version uses synchronous return values instead of async routing
 */
void MemReadRespPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	CLASS_ERROR << "void MemReadRespPacket::visit (SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Visitor method for SimModule routing (not implemented for write responses)
 *
 * @param _when Simulation tick when the visit occurs
 * @param _module The SimModule to visit
 *
 * @details Write response packets flow back to pipeline stages (SimBase), not to
 * SimModule components. This overload exists for interface completeness but is
 * not used in the current implementation.
 *
 * In the template version, write confirmations are handled synchronously through
 * direct return rather than through packet routing.
 *
 * @throws Triggers CLASS_ERROR if called
 * @note The full implementation may use response packet routing for async memory
 */
void MemWriteRespPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	CLASS_ERROR << "void MemWriteRespPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visitor method for SimBase routing (not implemented for write responses)
 *
 * @param _when Simulation tick when the visit occurs
 * @param _simulator The SimBase to visit
 *
 * @details In a fully asynchronous memory system, write responses would be routed back
 * to the requesting pipeline stage via this visitor. However, in the template version,
 * memory operations are synchronous and handled through direct completion rather than
 * packet routing.
 *
 * Future extensions with multi-cycle memory operations may implement this method to
 * route write confirmations to stages like MEMStage.
 *
 * @throws Triggers CLASS_ERROR if called
 * @note Template version uses synchronous completion instead of async routing
 */
void MemWriteRespPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	CLASS_ERROR << "void MemWriteRespPacket::visit (SimBase& simulator) is not implemented yet!";
}
