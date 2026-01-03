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

#include <queue>
#include <string>

#include "utils/HashableType.hh"

namespace acalsim {

/**
 * @file FifoQueue.hh
 * @brief Hardware-realistic FIFO queue with ready/valid handshaking
 *
 * @details
 * FifoQueue implements a bounded FIFO queue that models hardware buffering with
 * ready/valid handshaking signals. Unlike std::queue, this provides explicit
 * backpressure handling to accurately model hardware behavior.
 *
 * **Hardware Modeling:**
 *
 * - **pushReady (push-side ready)**: Queue has space for new data
 * - **popValid (pop-side valid)**: Queue has data available to read
 * - **Bounded Capacity**: Fixed maximum size (like hardware FIFOs)
 * - **Backpressure**: Automatic flow control via ready/valid signals
 *
 * **Ready/Valid Protocol:**
 * ```
 * Producer                    FifoQueue                    Consumer
 *    |                            |                            |
 *    |-------- isPushReady() ---->|                            |
 *    |<-------- true -------------|                            |
 *    |                            |                            |
 *    |-------- push(data) ------->|                            |
 *    |<-------- true -------------|                            |
 *    |                            |                            |
 *    |                            |<------- isPopValid() ------|
 *    |                            |-------- true ------------->|
 *    |                            |                            |
 *    |                            |<------- pop() -------------|
 *    |                            |-------- data ------------->|
 * ```
 *
 * **State Transitions:**
 * ```
 * Empty:  pushReady=true,  popValid=false
 * Partial: pushReady=true,  popValid=true
 * Full:   pushReady=false, popValid=true
 * ```
 *
 * **Use Cases:**
 *
 * | Component | Purpose |
 * |-----------|---------|
 * | **NoC Buffers** | Model router input/output queues with bounded capacity |
 * | **Pipeline Buffers** | Model stage-to-stage buffers with backpressure |
 * | **Request Queues** | Model pending request buffers in controllers |
 * | **Packet Buffers** | Model network packet queuing with finite capacity |
 * | **Command Queues** | Model hardware command FIFOs (e.g., DMA, GPU) |
 *
 * **Performance:**
 *
 * | Operation | Complexity | Notes |
 * |-----------|-----------|-------|
 * | push() | O(1) | std::queue push |
 * | pop() | O(1) | std::queue pop |
 * | isPushReady() | O(1) | Check pushReady flag |
 * | isPopValid() | O(1) | Check popValid flag |
 * | empty() | O(1) | Check queue empty |
 * | size() | O(1) | Get queue size |
 *
 * **Memory:** O(n) where n = queueSize (maximum capacity)
 *
 * **Thread Safety:**
 * - Not thread-safe - external synchronization required
 * - Common pattern: single producer + single consumer
 *
 * @tparam T Type of elements stored in the queue
 *
 * @code{.cpp}
 * // Example: NoC router input buffer with backpressure
 * FifoQueue<Packet*> inputBuffer(8, "router_input");
 *
 * // Producer side (upstream port)
 * void sendPacket(Packet* pkt) {
 *     if (inputBuffer.isPushReady()) {
 *         bool success = inputBuffer.push(pkt);
 *         assert(success);  // Should succeed if pushReady was true
 *     } else {
 *         // Backpressure! Stall the upstream port
 *         stallUpstream();
 *     }
 * }
 *
 * // Consumer side (router logic)
 * void processBuffer() {
 *     if (inputBuffer.isPopValid()) {
 *         Packet* pkt = inputBuffer.pop();
 *         routePacket(pkt);
 *     }
 * }
 * @endcode
 *
 * @note Queue size is fixed at construction - cannot be resized
 * @note pop() on empty queue returns default-constructed T and logs error
 * @note push() on full queue returns false (element not added)
 *
 * @warning Always check isPushReady() before push() to avoid data loss
 * @warning Always check isPopValid() before pop() to avoid invalid reads
 *
 * @see SimPort for port-level ready/valid interface
 * @see SimChannel for inter-component communication
 * @since ACALSim 0.1.0
 */
template <typename T>
class FifoQueue : virtual public HashableType {
public:
	/**
	 * @brief Construct a bounded FIFO queue with the specified capacity
	 *
	 * @param _qSize Maximum number of elements the queue can hold
	 * @param _name Name of the queue (for debugging/logging)
	 *
	 * @note Initializes with pushReady=true, popValid=false (empty state)
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * FifoQueue<int> cmdQueue(16, "gpu_cmd_queue");
	 * FifoQueue<Packet*> nocBuffer(32, "router_north_input");
	 * @endcode
	 */
	FifoQueue(size_t _qSize, std::string _name);

	/**
	 * @brief Destructor - clears all elements from the queue
	 *
	 * @note Calls reset() to clear queue
	 * @note Does NOT delete elements (pointer semantics)
	 */
	~FifoQueue();

	/**
	 * @brief Remove and return the front element from the queue
	 *
	 * @return Front element, or default-constructed T if empty
	 *
	 * @note Complexity: O(1)
	 * @note Automatically updates pushReady and popValid flags
	 * @note Logs error if called when popValid==false
	 *
	 * @warning Returns default T() if queue is empty (check isPopValid() first)
	 * @warning Modifies queue state - not const
	 *
	 * @code{.cpp}
	 * FifoQueue<Request*> queue(10, "mem_queue");
	 *
	 * if (queue.isPopValid()) {
	 *     Request* req = queue.pop();
	 *     processRequest(req);
	 * }
	 * @endcode
	 */
	inline T pop();

	/**
	 * @brief Add an element to the back of the queue
	 *
	 * @param t Element to add to the queue
	 * @return true if element was added, false if queue is full
	 *
	 * @note Complexity: O(1)
	 * @note Automatically updates pushReady and popValid flags
	 * @note Returns false if pushReady==false (queue full)
	 *
	 * @code{.cpp}
	 * FifoQueue<Packet*> buffer(8, "tx_buffer");
	 * Packet* pkt = new Packet();
	 *
	 * if (buffer.isPushReady()) {
	 *     bool ok = buffer.push(pkt);
	 *     assert(ok);  // Should succeed
	 * } else {
	 *     // Handle backpressure
	 *     delete pkt;
	 * }
	 * @endcode
	 */
	inline bool push(T t);

	/**
	 * @brief Update ready/valid flags based on current queue state
	 *
	 * @note Called automatically by push(), pop(), and reset()
	 * @note Sets pushReady = (size < queueSize)
	 * @note Sets popValid = !empty()
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * // Usually not needed - push/pop call this automatically
	 * queue.update();
	 * @endcode
	 */
	inline void update();

	/**
	 * @brief Clear all elements from the queue
	 *
	 * @note Removes all elements (pops until empty)
	 * @note Does NOT delete elements (pointer semantics)
	 * @note Resets to initial state: pushReady=true, popValid=false
	 * @note Complexity: O(n) where n = current size
	 *
	 * @code{.cpp}
	 * queue.reset();
	 * assert(queue.empty());
	 * assert(queue.isPushReady());
	 * assert(!queue.isPopValid());
	 * @endcode
	 */
	inline void reset();

	/**
	 * @brief Check if the queue is empty
	 *
	 * @return true if queue contains no elements, false otherwise
	 *
	 * @note Complexity: O(1)
	 * @note Equivalent to !isPopValid()
	 *
	 * @code{.cpp}
	 * if (queue.empty()) {
	 *     // No data available
	 * }
	 * @endcode
	 */
	inline bool empty() { return this->queue.empty(); }

	/**
	 * @brief Check if the queue can accept new elements (push-side ready)
	 *
	 * @return true if push() will succeed, false if queue is full
	 *
	 * @note Complexity: O(1)
	 * @note Updated automatically by push(), pop(), and update()
	 * @note Hardware equivalent: "ready" signal on producer interface
	 *
	 * @code{.cpp}
	 * // Hardware-style ready/valid handshaking
	 * if (buffer.isPushReady()) {
	 *     buffer.push(data);  // Guaranteed to succeed
	 * } else {
	 *     // Apply backpressure to upstream
	 * }
	 * @endcode
	 */
	inline bool isPushReady() { return this->pushReady; }

	/**
	 * @brief Check if the queue has data available (pop-side valid)
	 *
	 * @return true if pop() will return valid data, false if queue is empty
	 *
	 * @note Complexity: O(1)
	 * @note Updated automatically by push(), pop(), and update()
	 * @note Hardware equivalent: "valid" signal on consumer interface
	 *
	 * @code{.cpp}
	 * // Hardware-style ready/valid handshaking
	 * if (buffer.isPopValid()) {
	 *     Packet* pkt = buffer.pop();  // Guaranteed to have data
	 *     process(pkt);
	 * }
	 * @endcode
	 */
	inline bool isPopValid() { return this->popValid; }

	/**
	 * @brief Get the current number of elements in the queue
	 *
	 * @return Number of elements (0 to queueSize)
	 *
	 * @note Complexity: O(1)
	 *
	 * @code{.cpp}
	 * size_t fillLevel = queue.size();
	 * double utilization = (double)fillLevel / queueSize;
	 * @endcode
	 */
	inline size_t size() { return this->queue.size(); }

	/**
	 * @brief Get the name of the queue
	 *
	 * @return Queue name (for debugging/logging)
	 *
	 * @code{.cpp}
	 * std::string queueName = queue.getName();
	 * LOG_DEBUG << "Queue " << queueName << " is full";
	 * @endcode
	 */
	inline std::string getName() { return this->name; }

	/**
	 * @brief Get the front element without removing it
	 *
	 * @return Front element, or nullptr if empty (for pointer types)
	 *
	 * @note Does not remove the element
	 * @note Returns nullptr if empty (for pointer types only)
	 * @note For non-pointer types, behavior on empty is type-dependent
	 *
	 * @code{.cpp}
	 * FifoQueue<Packet*> queue(8, "pkt_buffer");
	 *
	 * Packet* pkt = queue.front();
	 * if (pkt != nullptr) {
	 *     inspectPacket(pkt);  // Don't remove yet
	 * }
	 * @endcode
	 */
	inline T front();

protected:
	/** @brief Queue name for debugging and logging */
	std::string name;

	/** @brief Push-side ready flag (true if queue has space) */
	bool pushReady;

	/** @brief Pop-side valid flag (true if queue has data) */
	bool popValid;

	/** @brief Underlying std::queue for FIFO storage */
	std::queue<T> queue;

	/** @brief Maximum queue capacity (fixed at construction) */
	size_t queueSize;
};

}  // namespace acalsim

#include "common/FifoQueue.inl"
