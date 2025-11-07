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
 * @file MemPacket.cc
 * @brief Implementation of memory request/response packets for RISC-V load/store operations
 *
 * @details
 * This file implements four packet classes that form the memory transaction protocol
 * between the processor pipeline and the data memory subsystem. These packets enable
 * asynchronous, non-blocking memory operations with callback-based completion handling.
 *
 * <b>Memory Transaction Protocol:</b>
 *
 * The memory subsystem uses a request-response protocol with four packet types:
 *
 * 1. <b>MemReadReqPacket:</b> Request to read data from memory (LB, LBU, LH, LHU, LW)
 * 2. <b>MemReadRespPacket:</b> Response containing data read from memory
 * 3. <b>MemWriteReqPacket:</b> Request to write data to memory (SB, SH, SW)
 * 4. <b>MemWriteRespPacket:</b> Acknowledgment that write completed
 *
 * <b>Transaction Flow - Memory Read:</b>
 *
 * @code
 * // 1. Execute stage creates read request
 * auto read_req = new MemReadReqPacket(
 *     [this](MemReadRespPacket* resp) {
 *         // Callback: handle loaded data
 *         uint32_t value = resp->getData();
 *         writeRegister(resp->getA1(), value);
 *     },
 *     instruction,  // RISC-V instruction
 *     LW,          // operation type
 *     0x1000,      // memory address
 *     dest_reg     // destination register
 * );
 *
 * // 2. Send to data memory
 * sendToMemory(read_req);
 *
 * // 3. Memory processes request, creates response
 * uint32_t data = memory->read(0x1000, 4);
 * auto read_resp = new MemReadRespPacket(instruction, LW, data, dest_reg);
 *
 * // 4. Invoke callback with response
 * read_req->getCallback()(read_resp);
 * @endcode
 *
 * <b>Transaction Flow - Memory Write:</b>
 *
 * @code
 * // 1. Execute stage creates write request
 * auto write_req = new MemWriteReqPacket(
 *     [this](MemWriteRespPacket* resp) {
 *         // Callback: write complete
 *         markStoreComplete(resp->getInstr());
 *     },
 *     instruction,  // RISC-V instruction
 *     SW,          // operation type
 *     0x2000,      // memory address
 *     0x12345678   // data to write
 * );
 *
 * // 2. Send to data memory
 * sendToMemory(write_req);
 *
 * // 3. Memory processes write, creates response
 * memory->write(0x2000, 0x12345678, 4);
 * auto write_resp = new MemWriteRespPacket(instruction);
 *
 * // 4. Invoke callback with response
 * write_req->getCallback()(write_resp);
 * @endcode
 *
 * <b>Packet Recycling and Renewal:</b>
 *
 * All packet classes support the renew() method for object pool reuse:
 * - Reinitializes packet fields with new values
 * - Calls base class SimPacket::renew() to reset SimPacket state
 * - Avoids repeated allocation/deallocation overhead
 * - Improves cache locality and reduces memory fragmentation
 *
 * Example:
 * @code
 * // Get recycled packet from pool
 * auto pkt = pool->get<MemReadReqPacket>();
 *
 * // Reinitialize for new request
 * pkt->renew(callback, inst, op, addr, operand);
 *
 * // Use packet...
 *
 * // Return to pool when done
 * pool->recycle(pkt);
 * @endcode
 *
 * <b>Visitor Pattern for Packet Routing:</b>
 *
 * Each packet class implements visit() methods to route packets to appropriate
 * handlers using the visitor pattern:
 *
 * - Request packets (MemReadReqPacket, MemWriteReqPacket) visit DataMemory module
 * - Response packets (MemReadRespPacket, MemWriteRespPacket) typically handled via callbacks
 *
 * The visit() method uses dynamic_cast to identify the target type:
 * @code
 * void MemReadReqPacket::visit(Tick when, SimModule& module) {
 *     if (auto dm = dynamic_cast<DataMemory*>(&module)) {
 *         dm->memReadReqHandler(when, this);
 *     } else {
 *         // Error: wrong module type
 *     }
 * }
 * @endcode
 *
 * <b>Memory Operations Supported:</b>
 *
 * Read Operations (Load Instructions):
 * - LB  (Load Byte): Read 1 byte, sign-extend to 32 bits
 * - LBU (Load Byte Unsigned): Read 1 byte, zero-extend to 32 bits
 * - LH  (Load Halfword): Read 2 bytes, sign-extend to 32 bits
 * - LHU (Load Halfword Unsigned): Read 2 bytes, zero-extend to 32 bits
 * - LW  (Load Word): Read 4 bytes (32 bits)
 *
 * Write Operations (Store Instructions):
 * - SB (Store Byte): Write lower 8 bits
 * - SH (Store Halfword): Write lower 16 bits
 * - SW (Store Word): Write 32 bits
 *
 * <b>Callback Mechanism:</b>
 *
 * Request packets include std::function callbacks that are invoked when the
 * memory operation completes. This enables:
 * - Asynchronous memory operations
 * - Non-blocking pipeline execution
 * - Flexible response handling
 * - Support for out-of-order completion
 *
 * Callback signatures:
 * - Read requests: std::function<void(MemReadRespPacket*)>
 * - Write requests: std::function<void(MemWriteRespPacket*)>
 *
 * <b>Instruction Association:</b>
 *
 * Each packet contains the original RISC-V instruction (instr) that generated
 * the memory request. This enables:
 * - Tracking instruction-to-memory-operation mapping
 * - Exception handling (which instruction caused fault)
 * - Performance analysis (instruction-level memory latency)
 * - Debugging and trace generation
 *
 * <b>Operand Tracking:</b>
 *
 * Read packets include operand information (a1) representing the destination
 * register. This allows the response handler to:
 * - Write loaded data to correct register
 * - Track register dependencies
 * - Implement register renaming
 * - Handle load-use hazards
 *
 * <b>Timing and Latency Modeling:</b>
 *
 * The _when parameter (acalsim::Tick) in visit() methods represents simulation
 * time, enabling:
 * - Memory access latency modeling
 * - Queuing delays
 * - Contention simulation
 * - Bandwidth modeling
 *
 * <b>Error Handling:</b>
 *
 * Currently, unimplemented visit() methods log CLASS_ERROR. Future enhancements
 * could include:
 * - Memory access exceptions (page faults, protection violations)
 * - Alignment exceptions
 * - Bus errors
 * - Timeout handling
 *
 * <b>Extension Points:</b>
 *
 * The packet-based design supports future extensions:
 * - Cache coherency protocols (add cache state to packets)
 * - Memory consistency models (add ordering constraints)
 * - Virtual memory (add page table walk requests)
 * - DMA operations (add burst transfer support)
 * - Atomic operations (LR/SC from RV32A extension)
 * - Memory ordering (FENCE instruction support)
 *
 * <b>Performance Considerations:</b>
 *
 * - Packets are lightweight (small fixed size)
 * - Callbacks avoid polling overhead
 * - Object pooling reduces allocation cost
 * - Direct function calls (not virtual) for handlers
 *
 * <b>Thread Safety:</b>
 *
 * Packets are not thread-safe. The ACALSim simulator is event-driven and
 * single-threaded, so synchronization is not required. For parallel simulation,
 * external synchronization would be needed.
 *
 * @see MemPacket.hh for packet class declarations
 * @see DataMemory for memory request handlers
 * @see BaseMemory for underlying memory storage
 * @see DataStruct.hh for RISC-V instruction types
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 */

#include "MemPacket.hh"

#include "DataMemory.hh"

/**
 * @brief Reinitializes a MemReadRespPacket for reuse from object pool
 *
 * @param _i The RISC-V instruction that generated the load request
 * @param _op The load operation type (LB, LBU, LH, LHU, LW)
 * @param _data The data value loaded from memory (32-bit, sign/zero-extended)
 * @param _a1 The destination register operand for the loaded data
 *
 * @details
 * This method reinitializes a recycled MemReadRespPacket with new response data.
 * It is called when reusing packet objects from the object pool to avoid repeated
 * allocation/deallocation overhead.
 *
 * <b>Reinitialization Process:</b>
 * 1. Calls SimPacket::renew() to reset base class state
 * 2. Updates instruction field with new instruction
 * 3. Sets operation type (load instruction variant)
 * 4. Stores loaded data value
 * 5. Sets destination operand information
 *
 * <b>Usage in Object Pool Pattern:</b>
 * @code
 * // Get recycled packet
 * auto resp = pool->get<MemReadRespPacket>();
 *
 * // Reinitialize with load response data
 * resp->renew(instruction, LW, 0x12345678, dest_reg);
 *
 * // Send response to callback
 * callback(resp);
 *
 * // Recycle when done
 * pool->recycle(resp);
 * @endcode
 *
 * @note The data should already be sign/zero-extended according to the
 *       operation type before calling renew().
 *
 * @see MemReadReqPacket for the corresponding request packet
 * @see SimPacket::renew() for base class reinitialization
 */
void MemReadRespPacket::renew(const instr& _i, instr_type _op, uint32_t _data, operand _a1) {
	this->acalsim::SimPacket::renew();
	this->i    = _i;
	this->op   = _op;
	this->data = _data;
	this->a1   = _a1;
}

/**
 * @brief Reinitializes a MemReadReqPacket for reuse from object pool
 *
 * @param _callback Callback function to invoke when read completes
 * @param _i The RISC-V instruction requesting the load
 * @param _op The load operation type (LB, LBU, LH, LHU, LW)
 * @param _addr The memory address to read from
 * @param _a1 The destination register operand
 *
 * @details
 * This method reinitializes a recycled MemReadReqPacket with new request parameters.
 * It prepares a packet for a new memory read operation, avoiding the overhead of
 * allocating a new packet object.
 *
 * <b>Reinitialization Process:</b>
 * 1. Calls SimPacket::renew() to reset base class state
 * 2. Stores completion callback function
 * 3. Updates instruction and operation type
 * 4. Sets target memory address
 * 5. Stores destination operand information
 *
 * <b>Callback Pattern:</b>
 * The callback function is invoked when the memory read completes:
 * @code
 * auto callback = [this](MemReadRespPacket* resp) {
 *     // Handle loaded data
 *     uint32_t value = resp->getData();
 *     regFile[resp->getA1()] = value;
 * };
 * @endcode
 *
 * <b>Usage Example:</b>
 * @code
 * // Get recycled request packet
 * auto req = pool->get<MemReadReqPacket>();
 *
 * // Reinitialize for new load operation
 * req->renew(
 *     [](MemReadRespPacket* r) { handleLoad(r); },
 *     inst,      // LW instruction
 *     LW,        // operation type
 *     0x1000,    // address
 *     x5         // destination register
 * );
 *
 * // Send to memory
 * memory->visit(currentTick, *req);
 * @endcode
 *
 * @note The callback must remain valid until the memory operation completes.
 *       Avoid capturing temporary references in the lambda.
 *
 * @see MemReadRespPacket for the response packet type
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
 * @brief Routes memory read request to DataMemory module
 *
 * @param _when Simulation tick when packet arrives
 * @param _module Target module (must be DataMemory)
 *
 * @details
 * Implements visitor pattern to route read requests to DataMemory.
 * Uses dynamic_cast to verify module type, then invokes the memory
 * read handler. Logs error if module is not DataMemory.
 *
 * @see DataMemory::memReadReqHandler()
 */
void MemReadReqPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	if (auto dm = dynamic_cast<DataMemory*>(&_module)) {
		dm->memReadReqHandler(_when, this);
	} else {
		CLASS_ERROR << "Invalid module type!";
	}
}

/**
 * @brief Visitor for SimBase (not implemented for read requests)
 *
 * @param _when Simulation tick
 * @param _simulator Target simulator
 *
 * @details
 * Read requests visit modules (DataMemory), not simulators (pipeline stages).
 * This method is not implemented and logs an error if called.
 */
void MemReadReqPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	CLASS_ERROR << "void MemReadReqPacket::visit (SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Reinitializes a MemWriteRespPacket for reuse
 *
 * @param _i The RISC-V instruction that generated the store
 *
 * @details
 * Reinitializes a write response packet from the object pool.
 * Write responses only need to carry the originating instruction
 * for tracking purposes, as no data is returned.
 *
 * @see MemWriteReqPacket for the corresponding request
 */
void MemWriteRespPacket::renew(const instr& _i) {
	this->acalsim::SimPacket::renew();
	this->i = _i;
}

/**
 * @brief Reinitializes a MemWriteReqPacket for reuse
 *
 * @param _callback Callback to invoke when write completes
 * @param _i The RISC-V instruction requesting the store
 * @param _op The store operation type (SB, SH, SW)
 * @param _addr The memory address to write to
 * @param _data The data to write (32-bit, lower bits used based on op)
 *
 * @details
 * Reinitializes a write request packet from the object pool with new
 * store parameters. The data field contains the full register value;
 * the memory handler extracts the appropriate bytes based on operation type.
 *
 * <b>Data Truncation by Operation:</b>
 * - SB: Uses lower 8 bits of _data
 * - SH: Uses lower 16 bits of _data
 * - SW: Uses all 32 bits of _data
 *
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
 * @brief Routes memory write request to DataMemory module
 *
 * @param _when Simulation tick when packet arrives
 * @param _module Target module (must be DataMemory)
 *
 * @details
 * Implements visitor pattern to route write requests to DataMemory.
 * Uses dynamic_cast to verify module type, then invokes the memory
 * write handler. Logs error if module is not DataMemory.
 *
 * @see DataMemory::memWriteReqHandler()
 */
void MemWriteReqPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	if (auto dm = dynamic_cast<DataMemory*>(&_module)) {
		dm->memWriteReqHandler(_when, this);
	} else {
		CLASS_ERROR << "Invalid module type!";
	}
}

/**
 * @brief Visitor for SimBase (not implemented for write requests)
 *
 * @param _when Simulation tick
 * @param _simulator Target simulator
 *
 * @details
 * Write requests visit modules (DataMemory), not simulators (pipeline stages).
 * This method is not implemented and logs an error if called.
 */
void MemWriteReqPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	CLASS_ERROR << "void MemWriteReqPacket::visit (SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Visitor for modules (not implemented for read responses)
 *
 * @param _when Simulation tick
 * @param _module Target module
 *
 * @details
 * Read responses are typically handled via callbacks rather than
 * the visitor pattern. This method is not implemented.
 */
void MemReadRespPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	CLASS_ERROR << "void MemReadRespPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visitor for simulators (not implemented for read responses)
 *
 * @param _when Simulation tick
 * @param _simulator Target simulator
 *
 * @details
 * Read responses are typically handled via callbacks rather than
 * the visitor pattern. This method is not implemented.
 */
void MemReadRespPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	CLASS_ERROR << "void MemReadRespPacket::visit (SimBase& simulator) is not implemented yet!";
}

/**
 * @brief Visitor for modules (not implemented for write responses)
 *
 * @param _when Simulation tick
 * @param _module Target module
 *
 * @details
 * Write responses are typically handled via callbacks rather than
 * the visitor pattern. This method is not implemented.
 */
void MemWriteRespPacket::visit(acalsim::Tick _when, acalsim::SimModule& _module) {
	CLASS_ERROR << "void MemWriteRespPacket::visit (SimModule& module) is not implemented yet!";
}

/**
 * @brief Visitor for simulators (not implemented for write responses)
 *
 * @param _when Simulation tick
 * @param _simulator Target simulator
 *
 * @details
 * Write responses are typically handled via callbacks rather than
 * the visitor pattern. This method is not implemented.
 */
void MemWriteRespPacket::visit(acalsim::Tick _when, acalsim::SimBase& _simulator) {
	CLASS_ERROR << "void MemWriteRespPacket::visit (SimBase& simulator) is not implemented yet!";
}
