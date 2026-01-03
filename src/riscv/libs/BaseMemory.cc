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
 * @file BaseMemory.cc
 * @brief Implementation of the BaseMemory class for RISC-V memory simulation
 *
 * @details
 * This file provides the core implementation of the BaseMemory class, which serves as
 * the foundational building block for memory models in the ACALSim RISC-V simulator.
 * BaseMemory implements a simple, flat memory model with byte-addressable storage
 * and provides essential read/write operations.
 *
 * <b>Memory Model Overview:</b>
 *
 * The BaseMemory class abstracts a contiguous memory space that can be accessed using
 * byte-level addressing. It serves as the base class for more specialized memory types
 * such as instruction memory and data memory. The implementation uses dynamically
 * allocated memory (via std::calloc) to simulate the physical memory space.
 *
 * <b>Key Features:</b>
 * - Byte-addressable memory with configurable size
 * - Zero-initialized memory allocation using std::calloc
 * - Support for both shallow (pointer) and deep (copy) memory reads
 * - Bounds checking for all memory accesses
 * - Direct memory pointer access for efficient operations
 *
 * <b>Memory Addressing:</b>
 *
 * Memory addresses are represented as 32-bit unsigned integers (uint32_t), consistent
 * with the RISC-V 32-bit address space. All memory operations use byte addressing:
 * - Address 0x00000000: First byte of memory
 * - Address 0x00000001: Second byte of memory
 * - etc.
 *
 * <b>Read Operations:</b>
 *
 * The readData() method supports two modes:
 * 1. Shallow read (default): Returns a pointer directly into the memory array.
 *    - Advantages: Zero-copy operation, very fast
 *    - Disadvantages: Caller must not modify data if memory should remain unchanged
 *    - Use case: Reading const data or when performance is critical
 *
 * 2. Deep read: Allocates new memory and copies the requested data.
 *    - Advantages: Caller owns the data and can freely modify it
 *    - Disadvantages: Memory allocation overhead, caller responsible for freeing
 *    - Use case: When data needs to be modified without affecting memory
 *
 * <b>Write Operations:</b>
 *
 * The writeData() method copies data from a source buffer into the memory array.
 * The operation uses std::memcpy for efficient bulk copying. Bounds checking
 * ensures that write operations do not exceed the allocated memory size.
 *
 * <b>Memory Safety:</b>
 *
 * All memory operations include assertions to verify:
 * - Memory accesses are within bounds (address + size <= memory size)
 * - Pointers are valid (not nullptr) for write operations
 * - Memory is properly allocated before use
 *
 * These checks help catch programming errors during development and simulation.
 *
 * <b>Usage Example:</b>
 *
 * @code
 * // Create a 64KB memory
 * BaseMemory* memory = new BaseMemory(65536);
 *
 * // Write a 32-bit value at address 0x1000
 * uint32_t value = 0x12345678;
 * memory->writeData(&value, 0x1000, sizeof(uint32_t));
 *
 * // Read the value back (shallow read)
 * uint32_t* ptr = (uint32_t*)memory->readData(0x1000, sizeof(uint32_t), false);
 * printf("Value: 0x%08x\n", *ptr);
 *
 * // Read with deep copy (caller must free)
 * uint32_t* copy = (uint32_t*)memory->readData(0x1000, sizeof(uint32_t), true);
 * *copy = 0x87654321;  // Modify copy without affecting memory
 * free(copy);
 *
 * // Clean up
 * delete memory;
 * @endcode
 *
 * <b>Memory Hierarchy Integration:</b>
 *
 * BaseMemory is designed to be extended by specialized memory classes:
 * - DataMemory: Adds RISC-V load/store instruction handling
 * - InstructionMemory: Specialized for instruction fetch operations
 * - CacheMemory: Could add caching behavior on top of BaseMemory
 *
 * Derived classes inherit the basic memory storage and access methods, then
 * add protocol-specific functionality such as:
 * - Instruction decoding for instruction memory
 * - Load/store size handling for data memory
 * - Cache coherency protocols for cache implementations
 * - Timing models for latency simulation
 *
 * <b>Performance Considerations:</b>
 *
 * - Memory is zero-initialized at construction time (std::calloc)
 * - Read operations are O(1) for shallow reads, O(n) for deep reads
 * - Write operations are O(n) where n is the number of bytes written
 * - No alignment requirements are enforced (relies on memcpy)
 * - Memory is allocated as a single contiguous block
 *
 * <b>Thread Safety:</b>
 *
 * BaseMemory is NOT thread-safe. External synchronization is required if
 * multiple threads access the same memory instance concurrently. In the
 * ACALSim simulator, memory accesses are serialized through the event-driven
 * simulation kernel.
 *
 * @see DataMemory for RISC-V data memory with load/store operations
 * @see BaseMemory.hh for class declaration and method documentation
 * @see MemPacket.hh for memory request/response packet definitions
 *
 * @author Playlab/ACAL
 * @date 2023-2025
 */

#include "BaseMemory.hh"

#include <cstdlib>
#include <cstring>

#include "ACALSim.hh"

/**
 * @brief Constructs a BaseMemory object with the specified size
 *
 * @param _size The size of the memory in bytes
 *
 * @details
 * Allocates a contiguous block of memory using std::calloc, which zeros out
 * all bytes. This ensures that the memory starts in a known state (all zeros),
 * which is useful for simulating power-on initialization of physical memory.
 *
 * The memory is allocated as a single contiguous block, which provides:
 * - Fast sequential access
 * - Simple address calculation
 * - Cache-friendly memory layout
 *
 * @note The size is immutable after construction. To change memory size,
 *       create a new BaseMemory instance.
 *
 * @throws May throw std::bad_alloc if memory allocation fails
 */
BaseMemory::BaseMemory(size_t _size) : size(_size) { this->mem = std::calloc(this->size, 1); }

/**
 * @brief Destroys the BaseMemory object and frees allocated memory
 *
 * @details
 * Releases the memory block allocated during construction using std::free.
 * This destructor is virtual in derived classes to ensure proper cleanup
 * when deleting through base class pointers.
 *
 * @note After destruction, any pointers obtained through readData() with
 *       _deep_copy=false become invalid (dangling pointers).
 */
BaseMemory::~BaseMemory() { std::free(this->mem); }

/**
 * @brief Returns the total size of the memory in bytes
 *
 * @return size_t The memory size in bytes
 *
 * @details
 * Returns the size specified during construction. This value is immutable
 * throughout the lifetime of the BaseMemory object.
 *
 * Useful for:
 * - Bounds checking before memory operations
 * - Determining valid address ranges
 * - Reporting memory configuration
 */
size_t BaseMemory::getSize() const { return this->size; }

/**
 * @brief Reads data from memory at the specified address
 *
 * @param _addr The starting address for the read operation (byte offset)
 * @param _size The number of bytes to read
 * @param _deep_copy If true, allocates new memory and copies data; if false,
 *                   returns pointer directly into memory array (default: false)
 * @return void* Pointer to the requested data
 *
 * @details
 * This method provides flexible memory read semantics with two modes:
 *
 * <b>Shallow Read (_deep_copy = false):</b>
 * Returns a pointer directly into the memory array. This is a zero-copy
 * operation that provides maximum performance. However, the caller must
 * ensure they do not modify the data if memory integrity is required.
 * The returned pointer remains valid until the BaseMemory object is destroyed.
 *
 * <b>Deep Read (_deep_copy = true):</b>
 * Allocates new memory using std::malloc and copies the requested data.
 * The caller takes ownership of the allocated memory and MUST free it
 * using std::free when done. This mode is safe for modifications but
 * incurs allocation and copy overhead.
 *
 * <b>Address Calculation:</b>
 * The address is interpreted as a byte offset from the start of memory:
 * - _addr=0: First byte of memory
 * - _addr=1: Second byte of memory
 * - etc.
 *
 * <b>Bounds Checking:</b>
 * An assertion verifies that _addr + _size <= memory size. Violations
 * indicate a programming error and will trigger an assertion failure.
 *
 * <b>Usage Examples:</b>
 * @code
 * // Shallow read - zero copy, fast
 * uint32_t* data_ptr = (uint32_t*)memory->readData(0x1000, 4, false);
 * uint32_t value = *data_ptr;  // OK to read
 * // *data_ptr = 0x123;        // AVOID: modifies memory directly!
 *
 * // Deep read - caller owns memory
 * uint32_t* data_copy = (uint32_t*)memory->readData(0x1000, 4, true);
 * *data_copy = 0x456;           // OK: modifying copy
 * free(data_copy);              // REQUIRED: caller must free
 * @endcode
 *
 * @warning When _deep_copy=true, the caller is responsible for freeing
 *          the returned pointer using std::free(). Memory leaks will occur
 *          if the pointer is not freed.
 *
 * @warning The caller must ensure all operations on the returned data
 *          are within the requested _size bytes. Buffer overruns will
 *          corrupt adjacent memory.
 *
 * @see writeData() for writing data to memory
 * @see getSize() for obtaining valid address ranges
 */
void* BaseMemory::readData(uint32_t _addr, size_t _size, bool _deep_copy) const {
	ASSERT_MSG(_addr + _size <= this->getSize(), "The memory region to be accessed is out of range.");
	if (_deep_copy) {
		size_t size = sizeof(uint8_t) * _size;
		void*  data = std::malloc(size);
		std::memcpy(data, (uint8_t*)this->mem + _addr, size);
		return data;
	} else {
		return (uint8_t*)this->mem + _addr;
	}
}

/**
 * @brief Writes data to memory at the specified address
 *
 * @param _data Pointer to the source data to be written
 * @param _addr The starting address for the write operation (byte offset)
 * @param _size The number of bytes to write from _data
 *
 * @details
 * This method copies data from the source buffer into the memory array at
 * the specified address. The operation uses std::memcpy for efficient bulk
 * data transfer.
 *
 * <b>Write Operation:</b>
 * - Copies _size bytes from _data buffer into memory at offset _addr
 * - Uses memcpy for efficient byte-level copying
 * - Supports writing any data type (cast to void*)
 * - No alignment requirements enforced
 *
 * <b>Address Calculation:</b>
 * The destination address is calculated as: base_memory_ptr + _addr
 * - _addr=0x0000: Write to first byte of memory
 * - _addr=0x1000: Write to byte at offset 4096
 * - etc.
 *
 * <b>Bounds Checking:</b>
 * Two assertions protect memory integrity:
 * 1. _data must not be nullptr (checks for null pointer dereference)
 * 2. _addr + _size must be <= memory size (prevents buffer overflow)
 *
 * <b>Usage Examples:</b>
 * @code
 * BaseMemory memory(65536);  // 64KB memory
 *
 * // Write a 32-bit integer
 * uint32_t value = 0x12345678;
 * memory.writeData(&value, 0x1000, sizeof(uint32_t));
 *
 * // Write an array of bytes
 * uint8_t buffer[] = {0x01, 0x02, 0x03, 0x04};
 * memory.writeData(buffer, 0x2000, sizeof(buffer));
 *
 * // Write a structure
 * struct MyData {
 *     uint32_t a;
 *     uint16_t b;
 *     uint8_t  c;
 * } my_data = {0x123, 0x45, 0x67};
 * memory.writeData(&my_data, 0x3000, sizeof(MyData));
 * @endcode
 *
 * <b>Byte Ordering:</b>
 * The method copies bytes as-is without any endianness conversion. The
 * byte order in memory will match the host system's byte order. For
 * cross-platform simulation, endianness should be handled at a higher
 * level (e.g., in load/store instruction handlers).
 *
 * <b>Overlapping Writes:</b>
 * Multiple writes to the same address will overwrite previous data.
 * The most recent write wins. There is no write protection or
 * read-only memory support in BaseMemory.
 *
 * @warning If _size exceeds the actual size of the _data buffer,
 *          undefined behavior will occur (reading beyond buffer).
 *          The caller must ensure _size accurately reflects the
 *          source buffer size.
 *
 * @warning This method does not perform any endianness conversion.
 *          Multi-byte values are written in host byte order.
 *
 * @pre _data must point to a valid buffer of at least _size bytes
 * @pre _addr + _size must be within memory bounds
 *
 * @see readData() for reading data from memory
 * @see getSize() for obtaining valid address ranges
 */
void BaseMemory::writeData(void* _data, uint32_t _addr, size_t _size) {
	ASSERT_MSG(_data, "The received argument `_data` is a nullptr.");
	ASSERT_MSG(_addr + _size <= this->getSize(), "The memory region to be accessed is out of range.");

	std::memcpy((uint8_t*)this->mem + _addr, _data, sizeof(uint8_t) * _size);
}
