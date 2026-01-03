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
 * @brief Base memory implementation for the riscvSimTemplate educational simulator
 *
 * @details This file provides the foundational memory abstraction for the RISC-V simulator template.
 * It implements a simple, flat memory model that serves as the base for both instruction and data
 * memory hierarchies. This is the template version designed for educational purposes and provides
 * a simplified but complete memory interface.
 *
 * Key Features:
 * - Contiguous byte-addressable memory allocation using calloc (zero-initialized)
 * - Configurable memory size at construction time
 * - Support for both shallow (pointer-based) and deep copy read operations
 * - Bounds checking with assertions to detect out-of-range accesses
 * - Clean RAII-based memory management (allocation in constructor, deallocation in destructor)
 *
 * Memory Model:
 * The BaseMemory class maintains a single contiguous block of memory that can be accessed
 * using byte addresses. All read and write operations operate on byte boundaries, allowing
 * for flexible access patterns required by RISC-V load/store instructions.
 *
 * Thread Safety:
 * This implementation is NOT thread-safe. External synchronization is required if multiple
 * threads access the same memory instance concurrently.
 *
 * Template Design Philosophy:
 * This is the simplified template version - the same structure is used in the full src/riscv/
 * implementation but may include additional features like memory hierarchy, caching, and
 * performance modeling. This template version focuses on correctness and clarity for
 * educational purposes.
 *
 * @note Part of the riscvSimTemplate educational framework
 * @see DataMemory for specialized data memory implementation
 * @see BaseMemory.hh for class interface documentation
 *
 * @author Playlab/ACAL
 * @version 1.0
 * @date 2023-2025
 */

#include "BaseMemory.hh"

#include <cstdlib>
#include <cstring>

#include "ACALSim.hh"

/**
 * @brief Constructs a BaseMemory object with specified size
 *
 * @param _size The total size of the memory in bytes
 *
 * @details Allocates a contiguous block of memory using std::calloc, which ensures all
 * bytes are initialized to zero. This zero-initialization is important for predictable
 * simulation behavior and matches real hardware power-on state assumptions.
 *
 * The memory is allocated as a byte array to support byte-addressable access patterns
 * required by RISC-V load/store instructions (LB, LH, LW, SB, SH, SW, etc.).
 *
 * @note The memory size is fixed at construction time and cannot be changed later
 * @warning Memory allocation may fail for very large sizes, but this is not explicitly checked
 */
BaseMemory::BaseMemory(size_t _size) : size(_size) { this->mem = std::calloc(this->size, 1); }

/**
 * @brief Destructor that releases allocated memory
 *
 * @details Frees the memory block allocated in the constructor using std::free.
 * This ensures proper cleanup following RAII principles. The destructor is automatically
 * called when the BaseMemory object goes out of scope or is explicitly deleted.
 */
BaseMemory::~BaseMemory() { std::free(this->mem); }

/**
 * @brief Returns the total size of the memory
 *
 * @return size_t The size of the memory in bytes
 *
 * @details This is a simple accessor that returns the memory size that was specified
 * during construction. The size is immutable and stored as a const member variable.
 */
size_t BaseMemory::getSize() const { return this->size; }

/**
 * @brief Reads data from memory at the specified address
 *
 * @param _addr The starting byte address to read from (must be within bounds)
 * @param _size The number of bytes to read
 * @param _deep_copy If true, creates a deep copy of the data; if false, returns a pointer
 *                   to the data in the memory buffer (default: false)
 *
 * @return void* Pointer to the requested data. If _deep_copy is true, caller is responsible
 *              for freeing the returned memory using std::free(). If false, the pointer is
 *              valid only as long as the BaseMemory object exists and no writes occur to
 *              that region.
 *
 * @details This method provides flexible read semantics:
 * - Shallow copy (_deep_copy=false): Returns a direct pointer into the internal memory buffer.
 *   This is efficient but requires careful lifetime management.
 * - Deep copy (_deep_copy=true): Allocates new memory and copies the requested data.
 *   This is safer for long-term storage but requires the caller to manage deallocation.
 *
 * Bounds checking is performed via assertion to catch programming errors during development.
 * The assertion will trigger if _addr + _size exceeds the total memory size.
 *
 * @warning When using shallow copy, the returned pointer becomes invalid if:
 *          - The BaseMemory object is destroyed
 *          - The memory region is overwritten by writeData()
 * @warning When using deep copy, the caller MUST free the returned pointer with std::free()
 *
 * @note This method is const because it does not modify the internal state, even though
 *       it may allocate new memory for deep copies
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
 * @param _data Pointer to the source data to be written (must not be null)
 * @param _addr The starting byte address to write to (must be within bounds)
 * @param _size The number of bytes to write from _data
 *
 * @details Copies _size bytes from the _data buffer into the internal memory buffer
 * starting at address _addr. The operation uses std::memcpy for efficient bulk copying.
 *
 * Bounds checking is performed via assertions:
 * - Checks that _data is not a null pointer
 * - Checks that the write operation does not exceed memory boundaries
 *
 * This method supports writing any size of data, from single bytes (for SB instructions)
 * to multiple words, making it suitable for implementing all RISC-V store operations.
 *
 * @warning The caller must ensure that _data points to at least _size bytes of valid memory.
 *          If _size exceeds the actual size of the data buffer, undefined behavior occurs
 *          (buffer overread).
 * @warning No alignment checking is performed. RISC-V typically requires aligned accesses
 *          for multi-byte operations, but this is not enforced at the BaseMemory level.
 *
 * @note This operation always performs a copy of the data; it does not store a pointer
 *       to the source data
 */
void BaseMemory::writeData(void* _data, uint32_t _addr, size_t _size) {
	ASSERT_MSG(_data, "The received argument `_data` is a nullptr.");
	ASSERT_MSG(_addr + _size <= this->getSize(), "The memory region to be accessed is out of range.");

	std::memcpy((uint8_t*)this->mem + _addr, _data, sizeof(uint8_t) * _size);
}
