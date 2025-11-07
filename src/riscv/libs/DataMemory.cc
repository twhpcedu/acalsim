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
 * @file DataMemory.cc
 * @brief Implementation of RISC-V data memory with load/store instruction handling
 *
 * @details
 * This file implements the DataMemory class, which extends BaseMemory to provide
 * RISC-V-specific data memory functionality. DataMemory handles memory request
 * packets from the processor pipeline and executes the corresponding load/store
 * operations according to RISC-V instruction semantics.
 *
 * <b>RISC-V Memory Model:</b>
 *
 * The RISC-V architecture defines a byte-addressable memory system with support
 * for aligned and unaligned memory accesses. DataMemory implements the following
 * RISC-V load/store instructions:
 *
 * <b>Load Instructions (Memory to Register):</b>
 * - LB  (Load Byte): Sign-extends 8-bit value to 32 bits
 * - LBU (Load Byte Unsigned): Zero-extends 8-bit value to 32 bits
 * - LH  (Load Halfword): Sign-extends 16-bit value to 32 bits
 * - LHU (Load Halfword Unsigned): Zero-extends 16-bit value to 32 bits
 * - LW  (Load Word): Loads 32-bit value (no extension needed)
 *
 * <b>Store Instructions (Register to Memory):</b>
 * - SB (Store Byte): Stores lower 8 bits of register
 * - SH (Store Halfword): Stores lower 16 bits of register
 * - SW (Store Word): Stores full 32-bit register value
 *
 * <b>Sign Extension vs Zero Extension:</b>
 *
 * Sign extension preserves the sign of signed integers when promoting from
 * smaller to larger types:
 * - LB with value 0xFF (-1 as signed byte) → 0xFFFFFFFF (-1 as signed word)
 * - LB with value 0x7F (127 as signed byte) → 0x0000007F (127 as signed word)
 *
 * Zero extension pads with zeros, treating values as unsigned:
 * - LBU with value 0xFF → 0x000000FF (255 as unsigned word)
 * - LBU with value 0x7F → 0x0000007F (127 as unsigned word)
 *
 * <b>Memory Access Sizes:</b>
 *
 * RISC-V defines three fundamental memory access sizes:
 * - Byte: 8 bits (1 byte) - LB, LBU, SB
 * - Halfword: 16 bits (2 bytes) - LH, LHU, SH
 * - Word: 32 bits (4 bytes) - LW, SW
 *
 * Note: RISC-V RV32I does not include doubleword (64-bit) memory operations.
 * Those are part of RV64I (64-bit RISC-V).
 *
 * <b>Memory Request/Response Protocol:</b>
 *
 * DataMemory uses a packet-based communication protocol:
 *
 * 1. Read Request Flow:
 *    - Pipeline stage creates MemReadReqPacket with instruction info and address
 *    - Packet is sent to DataMemory module via visit() pattern
 *    - memReadReqHandler() processes request:
 *      * Extracts instruction type, address, and operand info
 *      * Determines access size based on instruction type
 *      * Reads data from BaseMemory
 *      * Applies sign/zero extension as appropriate
 *      * Returns the loaded value
 *    - Packet is recycled to object pool
 *
 * 2. Write Request Flow:
 *    - Pipeline stage creates MemWriteReqPacket with instruction, address, and data
 *    - Packet is sent to DataMemory module via visit() pattern
 *    - memWriteReqHandler() processes request:
 *      * Extracts instruction type, address, and data value
 *      * Determines access size based on instruction type
 *      * Writes appropriate number of bytes to BaseMemory
 *      * Invokes callback if registered
 *    - Packet is recycled to object pool
 *
 * <b>Timing Model:</b>
 *
 * The current implementation provides functional simulation without detailed
 * timing modeling. Memory operations complete instantaneously (zero latency).
 * For timing-accurate simulation, derived classes could:
 * - Add latency modeling (e.g., fixed delay, queue-based)
 * - Implement memory hierarchy (caches, TLB)
 * - Model bandwidth constraints
 * - Simulate contention and arbitration
 *
 * <b>Packet Recycling:</b>
 *
 * To minimize dynamic memory allocation overhead, DataMemory uses the ACALSim
 * recycle container (object pool pattern). After processing each packet, it
 * is returned to the pool for reuse rather than being deleted. This improves
 * performance by:
 * - Reducing malloc/free overhead
 * - Improving cache locality
 * - Minimizing memory fragmentation
 *
 * <b>Byte Ordering (Endianness):</b>
 *
 * RISC-V defines a little-endian memory model for RV32I. Multi-byte values
 * are stored with the least significant byte at the lowest address:
 * - Word 0x12345678 at address 0x1000:
 *   - Byte at 0x1000: 0x78 (LSB)
 *   - Byte at 0x1001: 0x56
 *   - Byte at 0x1002: 0x34
 *   - Byte at 0x1003: 0x12 (MSB)
 *
 * The implementation relies on the host system's byte ordering. For accurate
 * cross-platform simulation, explicit byte swapping may be needed on big-endian
 * hosts.
 *
 * <b>Usage Example:</b>
 *
 * @code
 * // Create 1MB data memory
 * DataMemory* dmem = new DataMemory("DataMem", 1024*1024);
 *
 * // Example: Load Word (LW) instruction
 * instr lw_inst;  // Assume this is populated with LW encoding
 * auto read_pkt = new MemReadReqPacket(
 *     [](MemReadRespPacket* resp) {
 *         uint32_t loaded_value = resp->getData();
 *         // Process loaded value...
 *     },
 *     lw_inst,    // instruction
 *     LW,         // operation type
 *     0x1000,     // address
 *     operand()   // destination register info
 * );
 *
 * // Send packet to memory (normally done by simulator)
 * uint32_t value = dmem->memReadReqHandler(current_tick, read_pkt);
 *
 * // Example: Store Byte (SB) instruction
 * auto write_pkt = new MemWriteReqPacket(
 *     [](MemWriteRespPacket* resp) {
 *         // Store complete callback
 *     },
 *     sb_inst,    // instruction
 *     SB,         // operation type
 *     0x2000,     // address
 *     0xAB        // data (only lower 8 bits written)
 * );
 *
 * dmem->memWriteReqHandler(current_tick, write_pkt);
 * @endcode
 *
 * <b>Error Handling:</b>
 *
 * Memory access errors (out of bounds) are caught by assertions in BaseMemory.
 * In a production simulator, these could be replaced with:
 * - Exception generation (memory access fault)
 * - Error logging and recovery
 * - Debugger breakpoints
 *
 * <b>Extension Points:</b>
 *
 * DataMemory can be extended to add:
 * - Memory-mapped I/O (intercept specific address ranges)
 * - Cache simulation (add cache hierarchy above DataMemory)
 * - Memory protection (read-only regions, access permissions)
 * - Atomic operations (LR/SC for RV32A extension)
 * - Performance counters (track access patterns, hit rates)
 *
 * @see BaseMemory for underlying memory storage implementation
 * @see MemPacket.hh for memory request/response packet definitions
 * @see DataStruct.hh for RISC-V instruction type definitions
 * @see BaseMemory::readData() for low-level memory read
 * @see BaseMemory::writeData() for low-level memory write
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 */

#include "DataMemory.hh"

/**
 * @brief Handles memory read request packets for RISC-V load instructions
 *
 * @param _when Simulation tick when the request is being processed
 * @param _memReqPkt Pointer to the memory read request packet
 * @return uint32_t The loaded value, properly sign/zero-extended to 32 bits
 *
 * @details
 * This handler processes memory read requests generated by RISC-V load instructions.
 * It performs the following operations:
 *
 * <b>Processing Steps:</b>
 * 1. Extracts instruction metadata from packet (instruction, operation type, address)
 * 2. Determines access size based on instruction type (1, 2, or 4 bytes)
 * 3. Reads raw bytes from memory using BaseMemory::readData()
 * 4. Performs appropriate sign/zero extension to 32-bit value
 * 5. Recycles the request packet to object pool
 * 6. Returns the loaded value to caller
 *
 * <b>Supported Load Operations:</b>
 *
 * - <b>LB (Load Byte):</b>
 *   Reads 1 byte and sign-extends to 32 bits. Used for signed char.
 *   Example: Memory[addr]=0xFF → Return 0xFFFFFFFF (-1 in two's complement)
 *
 * - <b>LBU (Load Byte Unsigned):</b>
 *   Reads 1 byte and zero-extends to 32 bits. Used for unsigned char.
 *   Example: Memory[addr]=0xFF → Return 0x000000FF (255 unsigned)
 *
 * - <b>LH (Load Halfword):</b>
 *   Reads 2 bytes and sign-extends to 32 bits. Used for signed short.
 *   Example: Memory[addr..addr+1]=0xFFFF → Return 0xFFFFFFFF (-1)
 *
 * - <b>LHU (Load Halfword Unsigned):</b>
 *   Reads 2 bytes and zero-extends to 32 bits. Used for unsigned short.
 *   Example: Memory[addr..addr+1]=0xFFFF → Return 0x0000FFFF (65535)
 *
 * - <b>LW (Load Word):</b>
 *   Reads 4 bytes (full 32-bit word). No extension needed.
 *   Example: Memory[addr..addr+3]=0x12345678 → Return 0x12345678
 *
 * <b>Sign Extension Mechanism:</b>
 *
 * Sign extension is performed by casting to the appropriate signed type,
 * then casting back to uint32_t:
 * @code
 * int8_t signed_byte = *(int8_t*)data;   // Interpret as signed
 * uint32_t result = (uint32_t)signed_byte; // Extend with sign bit
 * @endcode
 *
 * This leverages C++'s implicit sign extension during integer promotion.
 *
 * <b>Memory Access Pattern:</b>
 *
 * The handler uses shallow read (no deep copy) for efficiency:
 * @code
 * void* data = this->readData(addr, bytes, false);
 * @endcode
 *
 * The returned pointer points directly into the memory array. Since the
 * data is immediately read and not modified, shallow read is safe and
 * avoids allocation overhead.
 *
 * <b>Packet Lifecycle:</b>
 *
 * After processing, the request packet is returned to the recycle container
 * (object pool) rather than being deleted:
 * @code
 * auto rc = acalsim::top->getRecycleContainer();
 * rc->recycle(_memReqPkt);
 * @endcode
 *
 * This improves performance by reusing packet objects instead of repeatedly
 * allocating and freeing them.
 *
 * <b>Example Scenarios:</b>
 *
 * Scenario 1: Loading a signed byte
 * @code
 * // Memory at 0x1000 contains: 0x80 (binary: 10000000)
 * // LB instruction reads this byte
 * // Result: 0xFFFFFF80 (-128 in 32-bit two's complement)
 * @endcode
 *
 * Scenario 2: Loading an unsigned byte
 * @code
 * // Memory at 0x1000 contains: 0x80
 * // LBU instruction reads this byte
 * // Result: 0x00000080 (128 in unsigned 32-bit)
 * @endcode
 *
 * Scenario 3: Loading a word (little-endian)
 * @code
 * // Memory at 0x2000 contains: 78 56 34 12 (bytes in order)
 * // LW instruction reads 4 bytes
 * // Result: 0x12345678 (little-endian interpretation)
 * @endcode
 *
 * <b>Alignment Considerations:</b>
 *
 * RISC-V allows unaligned memory accesses in RV32I. This implementation
 * does not enforce alignment requirements. However, real hardware may
 * suffer performance penalties or generate alignment exceptions for
 * unaligned accesses. Future enhancements could:
 * - Detect misaligned accesses
 * - Add performance penalties for misalignment
 * - Optionally generate alignment exceptions
 *
 * @note The _when parameter can be used for timing analysis in derived
 *       classes that implement memory latency modeling.
 *
 * @warning The packet pointer is recycled after processing. Do not use
 *          _memReqPkt after this function returns.
 *
 * @pre _memReqPkt must be a valid, non-null pointer
 * @pre The operation type must be a valid load instruction (LB/LBU/LH/LHU/LW)
 * @pre The memory address must be within bounds
 *
 * @post The request packet is returned to the recycle container
 *
 * @see MemReadReqPacket for request packet structure
 * @see BaseMemory::readData() for underlying read operation
 * @see memWriteReqHandler() for write operation handling
 */
uint32_t DataMemory::memReadReqHandler(acalsim::Tick _when, MemReadReqPacket* _memReqPkt) {
	instr      i    = _memReqPkt->getInstr();
	instr_type op   = _memReqPkt->getOP();
	uint32_t   addr = _memReqPkt->getAddr();
	operand    a1   = _memReqPkt->getA1();

	size_t   bytes = 0;
	uint32_t ret   = 0;

	switch (op) {
		case LB:
		case LBU: bytes = 1; break;
		case LH:
		case LHU: bytes = 2; break;
		case LW: bytes = 4; break;
	}

	void* data = this->readData(addr, bytes, false);

	switch (op) {
		case LB: ret = static_cast<uint32_t>(*(int8_t*)data); break;
		case LBU: ret = *(uint8_t*)data; break;
		case LH: ret = static_cast<uint32_t>(*(int16_t*)data); break;
		case LHU: ret = *(uint16_t*)data; break;
		case LW: ret = *(uint32_t*)data; break;
	}

	auto rc = acalsim::top->getRecycleContainer();
	rc->recycle(_memReqPkt);
	return ret;
}

/**
 * @brief Handles memory write request packets for RISC-V store instructions
 *
 * @param _when Simulation tick when the request is being processed
 * @param _memReqPkt Pointer to the memory write request packet
 *
 * @details
 * This handler processes memory write requests generated by RISC-V store instructions.
 * It performs the following operations:
 *
 * <b>Processing Steps:</b>
 * 1. Extracts instruction metadata from packet (instruction, operation, address, data)
 * 2. Determines access size based on instruction type (1, 2, or 4 bytes)
 * 3. Extracts appropriate number of bytes from the source data
 * 4. Writes bytes to memory using BaseMemory::writeData()
 * 5. Invokes completion callback if registered
 * 6. Recycles the request packet to object pool
 *
 * <b>Supported Store Operations:</b>
 *
 * - <b>SB (Store Byte):</b>
 *   Stores the lower 8 bits of the source register to memory.
 *   Upper 24 bits are ignored.
 *   Example: Register=0x12345678 → Memory[addr]=0x78
 *
 * - <b>SH (Store Halfword):</b>
 *   Stores the lower 16 bits of the source register to memory.
 *   Upper 16 bits are ignored.
 *   Example: Register=0x12345678 → Memory[addr..addr+1]=0x5678
 *
 * - <b>SW (Store Word):</b>
 *   Stores all 32 bits of the source register to memory.
 *   Example: Register=0x12345678 → Memory[addr..addr+3]=0x12345678
 *
 * <b>Byte Truncation:</b>
 *
 * For SB and SH instructions, only the relevant lower bits are stored.
 * The implementation uses C++ type casting to truncate values:
 *
 * @code
 * // Store Byte: Extract lower 8 bits
 * uint8_t val8 = static_cast<uint8_t>(data);  // Truncates to 8 bits
 * this->writeData(&val8, addr, 1);
 *
 * // Store Halfword: Extract lower 16 bits
 * uint16_t val16 = static_cast<uint16_t>(data);  // Truncates to 16 bits
 * this->writeData(&val16, addr, 2);
 * @endcode
 *
 * <b>Little-Endian Byte Ordering:</b>
 *
 * RISC-V uses little-endian byte ordering. Multi-byte stores write the
 * least significant byte at the lowest address:
 *
 * @code
 * // Example: SW with data=0x12345678 at address 0x1000
 * // Memory layout after store:
 * // Address 0x1000: 0x78 (LSB)
 * // Address 0x1001: 0x56
 * // Address 0x1002: 0x34
 * // Address 0x1003: 0x12 (MSB)
 * @endcode
 *
 * <b>Write Completion Callback:</b>
 *
 * The request packet may contain a callback function that is invoked when
 * the write completes. This allows the pipeline to:
 * - Continue execution after memory write
 * - Track outstanding memory operations
 * - Implement write buffers or write-through caches
 *
 * Currently, the callback is retrieved but not explicitly called in this
 * implementation, as writes complete synchronously (zero latency).
 *
 * <b>Packet Recycling:</b>
 *
 * After processing, the request packet is returned to the recycle container:
 * @code
 * auto rc = acalsim::top->getRecycleContainer();
 * rc->recycle(_memReqPkt);
 * @endcode
 *
 * This object pool pattern reduces memory allocation overhead and improves
 * cache locality for frequently allocated packet objects.
 *
 * <b>Example Scenarios:</b>
 *
 * Scenario 1: Store byte
 * @code
 * // Register x5 contains: 0xABCDEF12
 * // SB x5, 100(x0)  # Store to address 100
 * // Result: Memory[100] = 0x12 (only lower 8 bits)
 * // Memory[101..103] unchanged
 * @endcode
 *
 * Scenario 2: Store halfword
 * @code
 * // Register x6 contains: 0x87654321
 * // SH x6, 200(x0)  # Store to address 200
 * // Result: Memory[200] = 0x21 (LSB)
 * //         Memory[201] = 0x43
 * // Memory[202..203] unchanged
 * @endcode
 *
 * Scenario 3: Store word
 * @code
 * // Register x7 contains: 0x12345678
 * // SW x7, 300(x0)  # Store to address 300
 * // Result: Memory[300] = 0x78 (LSB)
 * //         Memory[301] = 0x56
 * //         Memory[302] = 0x34
 * //         Memory[303] = 0x12 (MSB)
 * @endcode
 *
 * <b>Memory Consistency:</b>
 *
 * This implementation provides sequential consistency - all stores are
 * immediately visible to subsequent loads. There is no:
 * - Write buffering
 * - Out-of-order completion
 * - Memory ordering relaxation
 *
 * Advanced implementations might model:
 * - Write buffers with delayed visibility
 * - Weak memory ordering (RISC-V RVWMO)
 * - Fence instructions for synchronization
 *
 * <b>Alignment Considerations:</b>
 *
 * RISC-V allows unaligned stores. This implementation does not enforce
 * alignment or impose penalties. Real hardware behavior may vary:
 * - Some implementations trap on misaligned access
 * - Others complete with performance penalty
 * - Atomic operations require alignment
 *
 * @note The _when parameter can be used for timing analysis in derived
 *       classes that implement write latency or write buffering.
 *
 * @warning The packet pointer is recycled after processing. Do not use
 *          _memReqPkt after this function returns.
 *
 * @pre _memReqPkt must be a valid, non-null pointer
 * @pre The operation type must be a valid store instruction (SB/SH/SW)
 * @pre The memory address must be within bounds
 *
 * @post Memory at the specified address is updated with the data
 * @post The request packet is returned to the recycle container
 *
 * @see MemWriteReqPacket for request packet structure
 * @see BaseMemory::writeData() for underlying write operation
 * @see memReadReqHandler() for read operation handling
 */
void DataMemory::memWriteReqHandler(acalsim::Tick _when, MemWriteReqPacket* _memReqPkt) {
	instr      i        = _memReqPkt->getInstr();
	instr_type op       = _memReqPkt->getOP();
	uint32_t   addr     = _memReqPkt->getAddr();
	uint32_t   data     = _memReqPkt->getData();
	auto       callback = _memReqPkt->getCallback();

	size_t bytes = 0;

	switch (op) {
		case SB: bytes = 1; break;
		case SH: bytes = 2; break;
		case SW: bytes = 4; break;
	}

	switch (op) {
		case SB: {
			uint8_t val8 = static_cast<uint8_t>(data);
			this->writeData(&val8, addr, 1);
			break;
		}
		case SH: {
			uint16_t val16 = static_cast<uint16_t>(data);
			this->writeData(&val16, addr, 2);
			break;
		}
		case SW: {
			uint32_t val32 = static_cast<uint32_t>(data);
			this->writeData(&val32, addr, 4);
			break;
		}
	}

	auto rc = acalsim::top->getRecycleContainer();
	rc->recycle(_memReqPkt);
}
