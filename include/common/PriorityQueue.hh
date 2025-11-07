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

#pragma once

#include <functional>
#include <map>
#include <unordered_set>
#include <vector>

namespace acalsim {

/**
 * @file PriorityQueue.hh
 * @brief Priority queue supporting multiple elements per priority level with set recycling
 *
 * @details
 * PriorityQueue is a specialized priority queue implementation that allows multiple
 * elements to share the same priority. Unlike std::priority_queue, this implementation:
 *
 * - Groups elements by priority using std::map (sorted priorities)
 * - Stores elements with same priority in std::unordered_set (fast lookup, no duplicates)
 * - Recycles unordered_set containers to reduce allocations
 * - Provides batch processing of all elements at the top priority
 *
 * **Key Design Decisions:**
 *
 * 1. **Multi-Element Priorities**: Elements with the same priority are stored together,
 *    enabling efficient batch processing
 * 2. **Set Recycling**: Empty unordered_sets are recycled rather than deleted, reducing
 *    allocation overhead in long-running simulations
 * 3. **Sorted Map**: Uses std::map for O(log n) priority operations with guaranteed ordering
 *
 * **Data Structure:**
 * ```
 * prioritySetMap: map<Priority, unordered_set<Elem>*>
 *   Priority 1 --> {Elem A, Elem B, Elem C}
 *   Priority 5 --> {Elem D, Elem E}
 *   Priority 10 --> {Elem F}
 *
 * elemSetReclcyeBin: vector of recycled unordered_set*
 *   [empty_set*, empty_set*, ...]
 * ```
 *
 * **Use Cases:**
 *
 * - **Event Scheduling**: Multiple events at the same simulation time
 * - **Task Scheduling**: Multiple tasks with the same deadline
 * - **Request Queues**: Multiple requests with the same priority class
 * - **Packet Scheduling**: Multiple packets in the same QoS class
 *
 * **Performance:**
 *
 * | Operation | Time Complexity | Notes |
 * |-----------|----------------|--------|
 * | insert() | O(log P + 1) | P = # unique priorities |
 * | getTopElem() | O(1) | |
 * | popTopElem() | O(log P) | |
 * | getTopElements() | O(E) | E = # elements at top priority |
 * | getTopPriority() | O(1) | |
 * | empty() | O(1) | |
 * | remove() | O(P × E) | Linear search, use sparingly |
 *
 * **Memory:**
 * - Base overhead: O(P) for map nodes
 * - Element storage: O(N) for all elements
 * - Recycled sets: O(R) where R = max historic unique priorities
 *
 * **Thread Safety:**
 * - Not thread-safe - external synchronization required
 * - Each thread should maintain its own PriorityQueue instance
 *
 * @tparam TPriority Priority type (must support operator< and std::map ordering)
 * @tparam TElem Element type (must be hashable for std::unordered_set)
 *
 * @code{.cpp}
 * // Example: Event scheduling with simulation time priority
 * PriorityQueue<uint64_t, Event*> eventQueue;
 *
 * // Insert events at different times
 * eventQueue.insert(event1, 100);  // Time 100
 * eventQueue.insert(event2, 100);  // Time 100 (same priority)
 * eventQueue.insert(event3, 200);  // Time 200
 *
 * // Process all events at top priority (time 100)
 * eventQueue.getTopElements([](const Event* e) {
 *     e->execute();
 * });
 *
 * // Top priority is now 200
 * assert(eventQueue.getTopPriority() == 200);
 * @endcode
 *
 * @note TPriority must be comparable with operator<
 * @note TElem must be hashable and equality comparable for std::unordered_set
 * @note Duplicate elements at the same priority are automatically deduplicated
 *
 * @warning remove() has O(P × E) complexity - use sparingly
 * @warning getTop() and popTop() on empty queue results in undefined behavior
 *
 * @see UpdateablePriorityQueue for updateable priorities (ThreadManagerV1)
 * @since ACALSim 0.1.0
 */
template <typename TPriority, typename TElem>
class PriorityQueue {
public:
	/**
	 * @brief Default constructor - creates an empty priority queue
	 *
	 * @note No memory allocated until first insert()
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * PriorityQueue<int, std::string> pq;
	 * assert(pq.empty());
	 * @endcode
	 */
	PriorityQueue();

	/**
	 * @brief Destructor - frees all allocated unordered_sets
	 *
	 * @note Destroys both active sets in prioritySetMap and recycled sets
	 * @note Does NOT delete the elements themselves (pointer semantics)
	 * @note Complexity: O(P + R) where P = unique priorities, R = recycled sets
	 */
	~PriorityQueue();

	/**
	 * @brief Insert an element with the specified priority
	 *
	 * @param _elem Element to insert
	 * @param _priority Priority of the element
	 *
	 * @note If element already exists at this priority, it is not duplicated
	 * @note Complexity: O(log P + 1) where P = # unique priorities
	 * @note Amortized O(1) for set insertion due to hash table
	 *
	 * @code{.cpp}
	 * PriorityQueue<int, std::string> pq;
	 * pq.insert("task1", 10);
	 * pq.insert("task2", 10);  // Same priority
	 * pq.insert("task3", 5);   // Higher priority (lower value)
	 * @endcode
	 */
	void insert(TElem _elem, TPriority _priority);

	/**
	 * @brief Get the first element at the top priority (without removing)
	 *
	 * @return An element from the top priority set
	 *
	 * @note Returns an arbitrary element if multiple exist at top priority
	 * @note Complexity: O(1)
	 * @note Does not remove the element
	 *
	 * @warning Undefined behavior if queue is empty - check empty() first
	 *
	 * @code{.cpp}
	 * PriorityQueue<int, std::string> pq;
	 * pq.insert("high", 1);
	 * pq.insert("low", 10);
	 *
	 * std::string top = pq.getTopElem();  // Returns "high"
	 * assert(!pq.empty());  // Element not removed
	 * @endcode
	 */
	TElem getTopElem() const;

	/**
	 * @brief Get and remove the first element at the top priority
	 *
	 * @return An element from the top priority set
	 *
	 * @note Returns an arbitrary element if multiple exist at top priority
	 * @note Complexity: O(log P) where P = # unique priorities
	 * @note Removes only one element, others at same priority remain
	 *
	 * @warning Undefined behavior if queue is empty - check empty() first
	 *
	 * @code{.cpp}
	 * PriorityQueue<int, std::string> pq;
	 * pq.insert("task1", 5);
	 * pq.insert("task2", 5);
	 *
	 * std::string task = pq.popTopElem();  // Returns "task1" or "task2"
	 * assert(!pq.empty());  // Other task still in queue
	 * @endcode
	 */
	TElem popTopElem();

	/**
	 * @brief Process all elements at the top priority (const reference callback)
	 *
	 * @param _func Callback function called for each element
	 *
	 * @note Removes ALL elements at the top priority after processing
	 * @note Callback receives const TElem& for read-only access
	 * @note Complexity: O(E) where E = # elements at top priority
	 * @note Set is recycled after processing
	 *
	 * @code{.cpp}
	 * PriorityQueue<int, Event*> pq;
	 * pq.insert(event1, 100);
	 * pq.insert(event2, 100);
	 * pq.insert(event3, 100);
	 *
	 * // Process all events at time 100
	 * pq.getTopElements([](const Event* e) {
	 *     e->execute();
	 * });
	 *
	 * // All three events removed, queue may be empty
	 * @endcode
	 */
	void getTopElements(std::function<void(const TElem&)> _func);

	/**
	 * @brief Process all elements at the top priority (const set callback)
	 *
	 * @param _func Callback function receiving the entire set
	 *
	 * @note Removes ALL elements after processing
	 * @note Callback receives const std::unordered_set<TElem>& for batch read
	 * @note Complexity: O(E) where E = # elements at top priority
	 * @note Useful for batch processing or statistics collection
	 *
	 * @code{.cpp}
	 * pq.getTopElements([](const std::unordered_set<Event*>& events) {
	 *     std::cout << "Processing " << events.size() << " events\n";
	 *     for (const auto& e : events) {
	 *         e->execute();
	 *     }
	 * });
	 * @endcode
	 */
	void getTopElements(std::function<void(const std::unordered_set<TElem>&)> _func);

	/**
	 * @brief Process all elements at the top priority (mutable set callback)
	 *
	 * @param _func Callback function receiving mutable set reference
	 *
	 * @note Removes ALL elements after processing
	 * @note Callback receives std::unordered_set<TElem>& for modification
	 * @note Complexity: O(E) where E = # elements at top priority
	 * @note Allows modification of elements before removal (advanced use)
	 *
	 * @warning Modifying the set during callback may have unexpected behavior
	 *
	 * @code{.cpp}
	 * pq.getTopElements([](std::unordered_set<Task*>& tasks) {
	 *     // Can modify tasks before they're removed
	 *     for (auto& task : tasks) {
	 *         task->markProcessed();
	 *     }
	 * });
	 * @endcode
	 */
	void getTopElements(std::function<void(std::unordered_set<TElem>&)> _func);

	/**
	 * @brief Get the top priority value
	 *
	 * @return Top priority, or std::numeric_limits<TPriority>::max() if empty
	 *
	 * @note Does not remove any elements
	 * @note Complexity: O(1)
	 * @note Safe to call on empty queue (returns max value)
	 *
	 * @code{.cpp}
	 * PriorityQueue<uint64_t, Event*> pq;
	 * pq.insert(event1, 100);
	 * pq.insert(event2, 200);
	 *
	 * uint64_t nextTime = pq.getTopPriority();  // Returns 100
	 *
	 * // Empty queue returns max value
	 * PriorityQueue<int, std::string> empty;
	 * int maxPri = empty.getTopPriority();  // Returns INT_MAX
	 * @endcode
	 */
	TPriority getTopPriority() const;

	/**
	 * @brief Check if the queue is empty
	 *
	 * @return true if no elements exist, false otherwise
	 *
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * PriorityQueue<int, std::string> pq;
	 * assert(pq.empty());
	 *
	 * pq.insert("task", 5);
	 * assert(!pq.empty());
	 * @endcode
	 */
	bool empty() const;

	/**
	 * @brief Remove a specific element from the queue (any priority)
	 *
	 * @param _elem Element to remove
	 *
	 * @note Searches through all priorities to find the element
	 * @note Complexity: O(P × E) where P = priorities, E = avg elements per priority
	 * @note If element not found, does nothing (silent failure)
	 * @note Recycles set if priority level becomes empty
	 *
	 * @warning Expensive operation - use sparingly
	 * @warning If element exists at multiple priorities, only first found is removed
	 *
	 * @code{.cpp}
	 * PriorityQueue<int, std::string> pq;
	 * pq.insert("task1", 10);
	 * pq.insert("task2", 20);
	 *
	 * pq.remove("task1");  // O(P × E) search
	 * assert(!pq.empty());  // task2 still exists
	 * @endcode
	 */
	void remove(const TElem& _elem);

protected:
	/**
	 * @brief Get a new or recycled unordered_set for storing elements
	 *
	 * @return Pointer to an empty unordered_set<TElem>
	 *
	 * @note Tries to reuse recycled sets before allocating new ones
	 * @note Complexity: O(1)
	 * @note Returned set is guaranteed to be empty
	 *
	 * @code{.cpp}
	 * std::unordered_set<Event*>* set = getNewElemSet();
	 * set->insert(event);
	 * prioritySetMap[priority] = set;
	 * @endcode
	 */
	std::unordered_set<TElem>* getNewElemSet();

private:
	/**
	 * @brief Map from priority to set of elements at that priority
	 * @details Sorted by priority (std::map), enabling O(1) access to minimum
	 */
	std::map<TPriority, std::unordered_set<TElem>*> prioritySetMap;

	/**
	 * @brief Pool of recycled unordered_sets to reduce allocations
	 * @details Sets are recycled when priority levels become empty
	 * @note Typo in variable name preserved for compatibility (should be "recycleBin")
	 */
	std::vector<std::unordered_set<TElem>*> elemSetReclcyeBin;
};

}  // namespace acalsim

#include "common/PriorityQueue.inl"
