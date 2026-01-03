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
 * @file DataMemory.cc
 * @brief Data memory implementation for RISC-V load/store operations in riscvSimTemplate
 *
 * @details This file implements the DataMemory module, which handles all data memory operations
 * for the RISC-V simulator template. It processes memory read and write requests from the
 * pipeline, implementing the semantics of RISC-V load and store instructions.
 *
 * Supported RISC-V Load Instructions:
 * - LB  (Load Byte): Loads 1 byte, sign-extends to 32 bits
 * - LBU (Load Byte Unsigned): Loads 1 byte, zero-extends to 32 bits
 * - LH  (Load Halfword): Loads 2 bytes, sign-extends to 32 bits
 * - LHU (Load Halfword Unsigned): Loads 2 bytes, zero-extends to 32 bits
 * - LW  (Load Word): Loads 4 bytes (full 32-bit word)
 *
 * Supported RISC-V Store Instructions:
 * - SB (Store Byte): Stores least significant byte
 * - SH (Store Halfword): Stores least significant 2 bytes
 * - SW (Store Word): Stores full 32-bit word
 *
 * Memory Access Pattern:
 * The DataMemory class acts as a handler for memory request packets. When a memory operation
 * is needed (from MEM stage), a packet is created and routed to this module via the visitor
 * pattern. The module then:
 * 1. Extracts operation type and parameters from the packet
 * 2. Determines the access size (1, 2, or 4 bytes)
 * 3. Performs the read or write operation with appropriate type conversion
 * 4. Returns data (for reads) or confirms completion (for writes)
 * 5. Recycles the packet to avoid memory allocation overhead
 *
 * Template Design Philosophy:
 * This is the template version for educational purposes. It demonstrates the core memory
 * subsystem behavior without advanced features like:
 * - Cache hierarchies (L1, L2, L3)
 * - Memory latency modeling
 * - Memory bandwidth constraints
 * - Virtual memory / address translation
 * These features may be present in the full src/riscv/ implementation.
 *
 * Integration with Pipeline:
 * - Receives MemReadReqPacket from MEM stage for load instructions
 * - Receives MemWriteReqPacket from MEM stage for store instructions
 * - Returns data synchronously (no latency modeling in template version)
 * - Uses ACALSim's packet recycling system for efficient memory management
 *
 * @note Part of the riscvSimTemplate educational framework
 * @note This implementation assumes little-endian byte ordering
 * @see BaseMemory for underlying memory storage implementation
 * @see MemPacket.hh for memory request/response packet definitions
 * @see DataMemory.hh for class interface documentation
 *
 * @author Playlab/ACAL
 * @version 1.0
 * @date 2023-2025
 */

#include "DataMemory.hh"

/**
 * @brief Handles memory read request packets from the pipeline
 *
 * @param _when The simulation tick when this request was received (used for timing)
 * @param _memReqPkt Pointer to the memory read request packet containing:
 *                   - Instruction information (instr)
 *                   - Operation type (LB, LBU, LH, LHU, or LW)
 *                   - Memory address to read from
 *                   - Destination operand information
 *
 * @return uint32_t The data read from memory, properly sign-extended or zero-extended
 *                  based on the instruction type
 *
 * @details This method implements the core logic for RISC-V load instructions:
 *
 * Step 1: Extract packet information
 * - Retrieves the instruction, operation type, address, and operand from the packet
 *
 * Step 2: Determine access size
 * - LB/LBU: 1 byte (8 bits)
 * - LH/LHU: 2 bytes (16 bits)
 * - LW: 4 bytes (32 bits)
 *
 * Step 3: Read from BaseMemory
 * - Calls readData() with shallow copy (efficient pointer return)
 * - No deep copy needed since data is immediately processed
 *
 * Step 4: Apply sign/zero extension
 * - Signed loads (LB, LH): Cast to signed type first, then to uint32_t for sign extension
 * - Unsigned loads (LBU, LHU): Direct cast to unsigned type for zero extension
 * - LW: No extension needed (already 32 bits)
 *
 * Step 5: Cleanup
 * - Recycles the request packet to the packet pool for reuse
 * - Reduces allocation overhead in the simulation
 *
 * Example Memory Read Sequence:
 * ```
 * Memory contains: [0xFF, 0x80, ...]
 * Address: 0x1000
 *
 * LB  0x1000 -> 0xFFFFFF80 (sign extended from 0x80)
 * LBU 0x1000 -> 0x00000080 (zero extended from 0x80)
 * ```
 *
 * @note In this template version, memory access is synchronous (single cycle)
 * @note The packet is automatically recycled after processing
 * @note No alignment checking is performed - assumes aligned accesses
 *
 * @warning Undefined instruction types will leave 'bytes' as 0, potentially causing issues
 */
uint32_t DataMemory::memReadReqHandler(acalsim::Tick _when, MemReadReqPacket* _memReqPkt) {
	instr      i    = _memReqPkt->getInstr();
	instr_type op   = _memReqPkt->getOP();
	uint32_t   addr = _memReqPkt->getAddr();
	operand    a1   = _memReqPkt->getA1();

	size_t   bytes = 0;
	uint32_t ret   = 0;

	// Determine the number of bytes to read based on instruction type
	switch (op) {
		case LB:
		case LBU: bytes = 1; break;
		case LH:
		case LHU: bytes = 2; break;
		case LW: bytes = 4; break;
	}

	// Read data from memory (shallow copy for efficiency)
	void* data = this->readData(addr, bytes, false);

	// Apply appropriate sign or zero extension based on instruction
	switch (op) {
		case LB: ret = static_cast<uint32_t>(*(int8_t*)data); break;   // Sign extend from 8 bits
		case LBU: ret = *(uint8_t*)data; break;                        // Zero extend from 8 bits
		case LH: ret = static_cast<uint32_t>(*(int16_t*)data); break;  // Sign extend from 16 bits
		case LHU: ret = *(uint16_t*)data; break;                       // Zero extend from 16 bits
		case LW: ret = *(uint32_t*)data; break;                        // No extension needed
	}

	// Recycle the packet to avoid repeated allocations
	auto rc = acalsim::top->getRecycleContainer();
	rc->recycle(_memReqPkt);
	return ret;
}

/**
 * @brief Handles memory write request packets from the pipeline
 *
 * @param _when The simulation tick when this request was received (used for timing)
 * @param _memReqPkt Pointer to the memory write request packet containing:
 *                   - Instruction information (instr)
 *                   - Operation type (SB, SH, or SW)
 *                   - Memory address to write to
 *                   - Data value to write
 *                   - Callback function for write completion (if needed)
 *
 * @details This method implements the core logic for RISC-V store instructions:
 *
 * Step 1: Extract packet information
 * - Retrieves instruction, operation type, address, data, and callback
 *
 * Step 2: Determine access size
 * - SB: 1 byte (8 bits)
 * - SH: 2 bytes (16 bits)
 * - SW: 4 bytes (32 bits)
 *
 * Step 3: Truncate and write data
 * - For SB: Extract least significant byte (bits 7:0)
 * - For SH: Extract least significant halfword (bits 15:0)
 * - For SW: Write full word (bits 31:0)
 * - Creates temporary variable of appropriate size for writeData call
 *
 * Step 4: Cleanup
 * - Recycles the request packet back to the packet pool
 * - The callback is extracted but not used in this synchronous implementation
 *
 * RISC-V Store Behavior:
 * Store instructions always take the least significant bits of the source register:
 * ```
 * Source register value: 0x12345678
 *
 * SB -> Writes 0x78 (least significant byte)
 * SH -> Writes 0x5678 (least significant halfword)
 * SW -> Writes 0x12345678 (full word)
 * ```
 *
 * Memory Layout (Little-Endian):
 * After SW 0x12345678 to address 0x1000:
 * ```
 * Address: 0x1000 0x1001 0x1002 0x1003
 * Data:    0x78   0x56   0x34   0x12
 * ```
 *
 * @note In this template version, writes are synchronous (single cycle)
 * @note The callback mechanism exists for compatibility but is not invoked
 * @note Packet is recycled after the write completes
 * @note No write-through or write-back cache behavior (direct memory write)
 *
 * @warning Undefined instruction types will not write any data
 * @warning No alignment checking is performed - assumes aligned accesses
 */
void DataMemory::memWriteReqHandler(acalsim::Tick _when, MemWriteReqPacket* _memReqPkt) {
	instr      i        = _memReqPkt->getInstr();
	instr_type op       = _memReqPkt->getOP();
	uint32_t   addr     = _memReqPkt->getAddr();
	uint32_t   data     = _memReqPkt->getData();
	auto       callback = _memReqPkt->getCallback();

	size_t bytes = 0;

	// Determine the number of bytes to write based on instruction type
	switch (op) {
		case SB: bytes = 1; break;
		case SH: bytes = 2; break;
		case SW: bytes = 4; break;
	}

	// Truncate data to appropriate size and write to memory
	switch (op) {
		case SB: {
			uint8_t val8 = static_cast<uint8_t>(data);  // Extract bits 7:0
			this->writeData(&val8, addr, 1);
			break;
		}
		case SH: {
			uint16_t val16 = static_cast<uint16_t>(data);  // Extract bits 15:0
			this->writeData(&val16, addr, 2);
			break;
		}
		case SW: {
			uint32_t val32 = static_cast<uint32_t>(data);  // Full 32 bits
			this->writeData(&val32, addr, 4);
			break;
		}
	}

	// Recycle the packet to avoid repeated allocations
	auto rc = acalsim::top->getRecycleContainer();
	rc->recycle(_memReqPkt);
}
