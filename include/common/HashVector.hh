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
#include <unordered_map>
#include <vector>

namespace acalsim {

/**
 * @file HashVector.hh
 * @brief Hybrid data structure combining hash map lookup with vector iteration
 *
 * @details
 * HashVector is a specialized container that provides both:
 * - **O(1) key-based lookup** (via internal unordered_map)
 * - **Cache-friendly sequential iteration** (via internal vector)
 *
 * This is particularly useful when you need to frequently iterate over all
 * elements (e.g., in simulation loops) while also needing fast random access
 * by key (e.g., looking up simulators by name).
 *
 * **Architecture:**
 * ```
 * HashVector internally maintains two data structures:
 *
 * 1. unordered_map<Key, VecElemRef>
 *    - Maps keys to vector indices
 *    - Provides O(1) lookup
 *
 * 2. vector<T>
 *    - Stores actual elements contiguously
 *    - Enables cache-friendly iteration
 *
 * Example with Key=string, T=Simulator*:
 *   umap_: {"GPU0" -> VecRef[0], "CPU0" -> VecRef[1], "NPU0" -> VecRef[2]}
 *   vec_:  [GPU0_ptr, CPU0_ptr, NPU0_ptr]
 * ```
 *
 * **Performance Comparison:**
 *
 * | Operation | std::unordered_map | std::vector | HashVector |
 * |-----------|-------------------|-------------|------------|
 * | Insert | O(1) avg | O(1) amortized | O(1) avg |
 * | Lookup by key | O(1) | O(n) | O(1) |
 * | Iteration | O(n) (poor cache) | O(n) (good cache) | O(n) (good cache) |
 * | Random access | O(1) by key | O(1) by index | O(1) by key |
 * | Memory overhead | High (nodes) | Low (contiguous) | Medium (both) |
 *
 * **Use Cases:**
 *
 * - **Simulator Collections**: Fast lookup by name + fast iteration for execution
 * - **Module Registries**: Register modules by name, iterate for updates
 * - **Resource Management**: Lookup resources by ID, iterate for allocation
 * - **Configuration Maps**: Access config by key, iterate all configs
 *
 * **Trade-offs:**
 *
 * ✅ **Advantages:**
 * - Fast lookup: O(1) average case
 * - Fast iteration: Cache-friendly vector storage
 * - Preserves insertion order (unlike unordered_map)
 *
 * ❌ **Disadvantages:**
 * - No element removal (would invalidate vector indices)
 * - Higher memory usage (stores both map and vector)
 * - Insert slightly slower than plain vector (updates both structures)
 *
 * **Memory Usage:**
 * - Map overhead: ~32 bytes per element (hash table buckets + nodes)
 * - Vector overhead: ~sizeof(T) per element
 * - Total: ~(32 + sizeof(T)) bytes per element + bucket array
 *
 * **Thread Safety:**
 * - Not thread-safe - external synchronization required
 * - Read-only operations can be concurrent if guaranteed no modifications
 * - Use separate HashVector instances per thread for lock-free operation
 *
 * @tparam Key Key type for lookup (must be hashable)
 * @tparam T Value type stored in vector
 * @tparam Hash Hash function object (default: std::hash<Key>)
 * @tparam KeyEqual Equality comparison (default: std::equal_to<Key>)
 * @tparam Allocator Allocator type (default: std::allocator<T>)
 *
 * @code{.cpp}
 * // Example: Simulator registry with fast lookup and iteration
 * HashVector<std::string, Simulator*> simulators;
 *
 * // Insert simulators (preserves insertion order)
 * simulators.insert({"GPU0", new GPUSimulator()});
 * simulators.insert({"CPU0", new CPUSimulator()});
 * simulators.insert({"NPU0", new NPUSimulator()});
 *
 * // Fast O(1) lookup by name
 * Simulator* gpu = simulators["GPU0"];
 * gpu->execute();
 *
 * // Cache-friendly iteration (insertion order preserved)
 * for (auto* sim : simulators) {
 *     sim->tick();  // Sequential access, good cache locality
 * }
 * @endcode
 *
 * @note Keys must be unique - duplicate inserts are silently ignored
 * @note No element removal supported - indices would become invalid
 * @note Iteration order matches insertion order
 *
 * @warning operator[] does NOT insert on missing keys (unlike std::map)
 * @warning No bounds checking - accessing invalid keys is undefined behavior
 *
 * @see SimTop for usage in simulator registry
 * @since ACALSim 0.1.0
 */
template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
          class Allocator = std::allocator<T>>
class HashVector {
	/**
	 * @brief Internal helper class - reference to vector element via index
	 *
	 * @details
	 * VecElemRef stores a pointer to the vector and an index, allowing the
	 * unordered_map to indirectly reference vector elements. This enables
	 * operator[] to return T& directly.
	 *
	 * **Why not store T* directly?**
	 * - Vector may reallocate, invalidating pointers
	 * - Index-based reference remains valid after vector growth
	 *
	 * @note This class is an implementation detail - not exposed to users
	 */
	class VecElemRef {
	public:
		/**
		 * @brief Default constructor - creates null reference
		 */
		inline VecElemRef() = default;

		/**
		 * @brief Construct reference to vector element at given index
		 *
		 * @param _vec Reference to the backing vector
		 * @param _index Index of the element in the vector
		 */
		inline VecElemRef(std::vector<T, Allocator>& _vec, const size_t& _index) : vec_(&_vec), index_(_index) {}

		/**
		 * @brief Dereference to get the actual vector element
		 *
		 * @return Reference to the element at index_ in vec_
		 *
		 * @note Returns reference, enabling modification via operator[]
		 */
		inline T& get() const { return (*this->vec_)[this->index_]; }

	private:
		/** @brief Pointer to the backing vector */
		std::vector<T, Allocator>* vec_ = nullptr;

		/** @brief Index of the element in the vector */
		size_t index_ = 0;
	};

public:
	/**
	 * @brief Default constructor - creates an empty HashVector
	 *
	 * @note No memory allocated until first insert()
	 * @note Default hash table size determined by std::unordered_map
	 *
	 * @code{.cpp}
	 * HashVector<std::string, int> hv;
	 * assert(hv.size() == 0);
	 * @endcode
	 */
	HashVector() = default;

	/**
	 * @brief Construct HashVector with custom hash table parameters
	 *
	 * @param _bucket_count Initial number of hash table buckets
	 * @param _hash Hash function object
	 * @param _equal Key equality comparison function
	 * @param _alloc Allocator for vector elements
	 *
	 * @note Pre-sizing hash table improves insert performance
	 * @note Vector not pre-allocated - use reserve() for that
	 *
	 * @code{.cpp}
	 * // Reserve space for 1000 elements to avoid rehashing
	 * HashVector<int, std::string> hv(1000);
	 * hv.reserve(1000);  // Also reserve vector space
	 * @endcode
	 */
	HashVector(const size_t& _bucket_count, const Hash& _hash = Hash(), const KeyEqual& _equal = KeyEqual(),
	           const Allocator& _alloc = Allocator());

	/**
	 * @brief Virtual destructor - allows safe polymorphic use
	 *
	 * @note Does NOT delete elements (pointer semantics)
	 * @note Cleans up internal map and vector structures
	 */
	virtual ~HashVector() = default;

	/**
	 * @brief Insert a key-value pair into the HashVector
	 *
	 * @param _value Pair containing {key, value}
	 * @return true if inserted, false if key already exists
	 *
	 * @note Duplicate keys are silently ignored (returns false)
	 * @note Complexity: O(1) average case (amortized for vector growth)
	 * @note Preserves insertion order for iteration
	 *
	 * @code{.cpp}
	 * HashVector<std::string, int> hv;
	 *
	 * bool ok1 = hv.insert({"alice", 100});  // true - inserted
	 * bool ok2 = hv.insert({"bob", 200});    // true - inserted
	 * bool ok3 = hv.insert({"alice", 300});  // false - key exists
	 *
	 * assert(hv["alice"] == 100);  // Original value preserved
	 * @endcode
	 */
	bool insert(const std::pair<Key, T>& _value);

	/**
	 * @brief Reserve capacity for both internal structures
	 *
	 * @param _count Expected number of elements
	 *
	 * @note Reserves space in both hash table and vector
	 * @note Avoids rehashing and reallocation during inserts
	 * @note Does not change size(), only capacity
	 *
	 * @code{.cpp}
	 * HashVector<int, std::string> hv;
	 * hv.reserve(10000);  // Reserve for 10k elements
	 *
	 * // Now 10k inserts won't trigger reallocation
	 * for (int i = 0; i < 10000; i++) {
	 *     hv.insert({i, std::to_string(i)});
	 * }
	 * @endcode
	 */
	inline void reserve(size_t _count);

	/**
	 * @brief Remove all elements from the HashVector
	 *
	 * @note Clears both map and vector
	 * @note Does NOT delete elements (pointer semantics)
	 * @note Capacity is preserved (no memory deallocation)
	 * @note Complexity: O(n)
	 *
	 * @code{.cpp}
	 * HashVector<int, std::string> hv;
	 * hv.insert({1, "one"});
	 * hv.insert({2, "two"});
	 *
	 * hv.clear();
	 * assert(hv.size() == 0);
	 * @endcode
	 */
	inline void clear() noexcept;

	/**
	 * @brief Get the number of elements in the HashVector
	 *
	 * @return Number of key-value pairs stored
	 *
	 * @note Complexity: O(1)
	 * @note size() == vec_.size() == umap_.size()
	 *
	 * @code{.cpp}
	 * HashVector<int, std::string> hv;
	 * assert(hv.size() == 0);
	 *
	 * hv.insert({1, "one"});
	 * assert(hv.size() == 1);
	 * @endcode
	 */
	inline size_t size() const { return this->vec_.size(); }

	/**
	 * @brief Access element by key (mutable reference)
	 *
	 * @param _key Key to lookup
	 * @return Mutable reference to the value
	 *
	 * @note Complexity: O(1) average case
	 * @note Does NOT insert on missing key (unlike std::map)
	 * @note Accessing missing key is undefined behavior
	 *
	 * @warning No bounds checking - key must exist
	 * @warning Undefined behavior if key not found
	 *
	 * @code{.cpp}
	 * HashVector<std::string, int> hv;
	 * hv.insert({"score", 100});
	 *
	 * hv["score"] = 200;      // Modify existing
	 * int x = hv["score"];    // Read value (200)
	 *
	 * // hv["missing"] = 0;   // UNDEFINED BEHAVIOR - key doesn't exist!
	 * @endcode
	 */
	inline T& operator[](const Key& _key) { return this->umap_[_key].get(); }

	/**
	 * @brief Get const reference to internal unordered_map
	 *
	 * @return Const reference to the map
	 *
	 * @note Useful for advanced operations (checking existence, etc.)
	 * @note Map values are VecElemRef, not T directly
	 *
	 * @code{.cpp}
	 * HashVector<std::string, int> hv;
	 * hv.insert({"key", 42});
	 *
	 * const auto& map = hv.getUMapRef();
	 * bool exists = map.find("key") != map.end();  // Check existence
	 * @endcode
	 */
	inline const std::unordered_map<Key, VecElemRef, Hash, KeyEqual>& getUMapRef() const { return this->umap_; }

	/**
	 * @brief Get const reference to internal vector
	 *
	 * @return Const reference to the vector
	 *
	 * @note Useful for accessing elements by insertion order
	 * @note Same as iterating with begin()/end()
	 *
	 * @code{.cpp}
	 * HashVector<std::string, int> hv;
	 * hv.insert({"a", 1});
	 * hv.insert({"b", 2});
	 *
	 * const auto& vec = hv.getVecRef();
	 * assert(vec[0] == 1);  // First inserted
	 * assert(vec[1] == 2);  // Second inserted
	 * @endcode
	 */
	inline const std::vector<T, Allocator>& getVecRef() const { return this->vec_; }

	/**
	 * @brief Get iterator to beginning of vector (mutable)
	 *
	 * @return Iterator to first element (insertion order)
	 *
	 * @note Iteration order matches insertion order
	 * @note Cache-friendly sequential access
	 *
	 * @code{.cpp}
	 * for (auto it = hv.begin(); it != hv.end(); ++it) {
	 *     // Process *it
	 * }
	 * @endcode
	 */
	inline typename std::vector<T, Allocator>::iterator begin() { return this->vec_.begin(); }

	/**
	 * @brief Get iterator to end of vector (mutable)
	 * @return Iterator past the last element
	 */
	inline typename std::vector<T, Allocator>::iterator end() { return this->vec_.end(); }

	/**
	 * @brief Get const iterator to beginning of vector
	 * @return Const iterator to first element
	 */
	inline typename std::vector<T, Allocator>::const_iterator begin() const { return this->vec_.begin(); }

	/**
	 * @brief Get const iterator to end of vector
	 * @return Const iterator past the last element
	 */
	inline typename std::vector<T, Allocator>::const_iterator end() const { return this->vec_.end(); }

private:
	/**
	 * @brief Hash map for O(1) key-based lookup
	 * @details Maps Key -> VecElemRef (index into vec_)
	 */
	std::unordered_map<Key, VecElemRef, Hash, KeyEqual> umap_;

	/**
	 * @brief Vector for cache-friendly sequential iteration
	 * @details Stores actual elements in insertion order
	 */
	std::vector<T, Allocator> vec_;
};

}  // namespace acalsim

#include "common/HashVector.inl"
