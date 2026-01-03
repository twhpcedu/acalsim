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

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace acalsim {

/**
 * @file BitVector.hh
 * @brief Space-efficient bit vector implementation using uint64_t backing storage
 *
 * @details
 * BitVector provides a compact representation of boolean arrays by packing bits
 * into 64-bit unsigned integers. This reduces memory usage by 64x compared to
 * std::vector<bool> while providing similar functionality.
 *
 * **Memory Layout:**
 * - Each uint64_t stores 64 bits
 * - Total storage: ceil(size / 64) * 8 bytes
 * - Bit ordering: LSB-first within each uint64_t
 *
 * **Use Cases:**
 * - Active simulator tracking in ThreadManager
 * - Port state bitmasks (ready/valid signals)
 * - Resource availability flags
 * - Test coverage masks
 *
 * **Performance:**
 * - Set/Get: O(1) with bit manipulation
 * - All Equal: O(n/64) with word-level comparison
 * - Reset: O(n/64) with memset equivalent
 *
 * **Thread Safety:**
 * - Not thread-safe - external synchronization required
 * - Use separate BitVector instances per thread for lock-free operation
 *
 * @code{.cpp}
 * // Example: Track active simulators in a multi-threaded simulation
 * BitVector activeSimulators(128, false);  // 128 simulators, initially inactive
 *
 * // Mark simulator 42 as active
 * activeSimulators.setBit(42, true);
 *
 * // Check if simulator 42 is active
 * if (activeSimulators.getBit(42)) {
 *     // Execute simulator 42
 * }
 *
 * // Check if all simulators are inactive
 * if (activeSimulators.allEqual(false)) {
 *     // Simulation can terminate
 * }
 *
 * // Reset all bits to false
 * activeSimulators.reset();
 * @endcode
 *
 * @note Accessing bits beyond the vector size results in undefined behavior
 * @warning Not compatible with std::vector<bool> - direct replacement requires API changes
 *
 * @see UpdateablePriorityQueue for usage in ThreadManagerV1
 * @since ACALSim 0.1.0
 */
class BitVector {
private:
	/** @brief Number of bits in the vector (fixed at construction) */
	const size_t size;

	/**
	 * @brief Backing storage using 64-bit words
	 * @details Length: ceil(size / 64) uint64_t elements
	 */
	std::vector<uint64_t> bitvec;

	/** @brief Constant for bitmask operations (0xFFFFFFFFFFFFFFFF) */
	const static uint64_t ALL_ONE;

public:
	/**
	 * @brief Construct a BitVector with the specified size
	 *
	 * @param _size Number of bits in the vector (must be > 0)
	 * @param _initial Initial value for all bits (default: false)
	 *
	 * @note Memory allocated: ceil(_size / 64) * 8 bytes
	 * @note Complexity: O(n/64) where n = _size
	 *
	 * @code{.cpp}
	 * BitVector bv1(100);           // 100 bits, all false
	 * BitVector bv2(200, true);     // 200 bits, all true
	 * @endcode
	 */
	BitVector(size_t _size, bool _initial = false);

	/**
	 * @brief Copy constructor - creates a deep copy
	 *
	 * @param _other BitVector to copy from
	 *
	 * @note Complexity: O(n/64) where n = other.size
	 * @note Both vectors are independent after copy
	 *
	 * @code{.cpp}
	 * BitVector original(128, false);
	 * original.setBit(42, true);
	 *
	 * BitVector copy(original);  // Deep copy
	 * copy.setBit(42, false);    // Does not affect original
	 * @endcode
	 */
	BitVector(const BitVector& _other);

	/**
	 * @brief Default destructor
	 * @note Automatically cleans up std::vector backing storage
	 */
	~BitVector() = default;

	/**
	 * @brief Set the value of a bit at the specified index
	 *
	 * @param _idx Bit index (0-based, must be < size)
	 * @param _value Value to set (true or false)
	 *
	 * @note Complexity: O(1)
	 * @note Thread-unsafe - no bounds checking in release builds
	 *
	 * @warning Accessing _idx >= size results in undefined behavior
	 *
	 * @code{.cpp}
	 * BitVector bv(64);
	 * bv.setBit(0, true);   // Set first bit
	 * bv.setBit(63, true);  // Set last bit
	 * @endcode
	 */
	void setBit(size_t _idx, bool _value);

	/**
	 * @brief Get the value of a bit at the specified index
	 *
	 * @param _idx Bit index (0-based, must be < size)
	 * @return true if bit is set, false otherwise
	 *
	 * @note Complexity: O(1)
	 * @note Thread-unsafe - no bounds checking in release builds
	 *
	 * @warning Accessing _idx >= size results in undefined behavior
	 *
	 * @code{.cpp}
	 * BitVector bv(64, false);
	 * bv.setBit(42, true);
	 *
	 * bool isSet = bv.getBit(42);  // Returns true
	 * bool notSet = bv.getBit(0);  // Returns false
	 * @endcode
	 */
	bool getBit(size_t _idx) const;

	/**
	 * @brief Check if all bits are equal to the given value
	 *
	 * @param _value Value to compare against (true or false)
	 * @return true if all bits equal _value, false otherwise
	 *
	 * @note Complexity: O(n/64) where n = size
	 * @note Uses word-level comparison for efficiency
	 * @note Handles partial words (size not multiple of 64) correctly
	 *
	 * @code{.cpp}
	 * BitVector bv(128, false);
	 *
	 * bool allFalse = bv.allEqual(false);  // Returns true
	 * bool allTrue = bv.allEqual(true);    // Returns false
	 *
	 * bv.setBit(0, true);
	 * bool stillAllFalse = bv.allEqual(false);  // Returns false
	 * @endcode
	 */
	bool allEqual(bool _value) const;

	/**
	 * @brief Check if the first 64 bits equal the given bitmask (for testing)
	 *
	 * @param _value 64-bit mask to compare against
	 * @return true if bitvec[0] == _value, false otherwise
	 *
	 * @note Complexity: O(1)
	 * @note Only checks the first uint64_t word
	 * @note Primarily used in Google Test unit tests
	 *
	 * @warning Undefined behavior if size < 64
	 *
	 * @code{.cpp}
	 * BitVector bv(64, false);
	 * bv.setBit(0, true);
	 * bv.setBit(63, true);
	 *
	 * uint64_t expected = 0x8000000000000001ULL;
	 * EXPECT_TRUE(bv.gTestBitMaskEqual(expected));
	 * @endcode
	 */
	bool gTestBitMaskEqual(uint64_t _value) const { return bitvec[0] == _value; }

	/**
	 * @brief Get the first 64 bits as a uint64_t bitmask (for testing)
	 *
	 * @return Value of bitvec[0]
	 *
	 * @note Complexity: O(1)
	 * @note Primarily used in Google Test unit tests
	 *
	 * @warning Undefined behavior if size < 64
	 *
	 * @code{.cpp}
	 * BitVector bv(64, false);
	 * bv.setBit(0, true);
	 *
	 * uint64_t mask = bv.getGTestBitMask();  // Returns 0x0000000000000001ULL
	 * @endcode
	 */
	uint64_t getGTestBitMask() const { return bitvec[0]; }

	/**
	 * @brief Reset all bits to false
	 *
	 * @note Complexity: O(n/64) where n = size
	 * @note Equivalent to filling with zeros
	 *
	 * @code{.cpp}
	 * BitVector bv(128, true);  // All bits true
	 * bv.reset();               // All bits now false
	 *
	 * EXPECT_TRUE(bv.allEqual(false));
	 * @endcode
	 */
	void reset();
};

}  // end of namespace acalsim
