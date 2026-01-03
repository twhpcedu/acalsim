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

#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace acalsim {

/**
 * Lock-free implementation of an Updateable Priority Queue
 * Uses a combination of atomic operations and fine-grained locking
 * for optimal performance and correctness
 */
template <typename T>
class LockFreeUpdateablePriorityQueue {
private:
	// Structure for nodes in our queue
	struct Node {
		T                     value;
		std::atomic<uint64_t> priority;
		std::atomic<bool>     needsUpdate;
		std::atomic<int>      index;  // Position in the heap

		Node(const T& val, uint64_t prio, bool update, int idx)
		    : value(val), priority(prio), needsUpdate(update), index(idx) {}

		// Add move constructor for better performance
		Node(T&& val, uint64_t prio, bool update, int idx)
		    : value(std::move(val)), priority(prio), needsUpdate(update), index(idx) {}
	};

	// Internal data structure
	std::vector<std::shared_ptr<Node>> nodes;
	std::atomic<size_t>                size_;
	std::mutex                         vector_mutex;  // Protects vector operations only
	std::atomic<uint64_t>              version_counter;

public:
	LockFreeUpdateablePriorityQueue() : size_(0), version_counter(0) {
		// Reserve initial capacity
		nodes.reserve(1024);
	}

	bool compare(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
		// Return true if element a has a lower priority value than element b
		return a->priority.load(std::memory_order_acquire) < b->priority.load(std::memory_order_acquire);
	}

	/**
	 * Push an element with its priority
	 */
	void push(const T& value, uint64_t priority) {
		// Create a copy of value to ensure it remains valid
		T value_copy(value);

		// Create new node with moved value for efficiency
		auto new_node = std::make_shared<Node>(std::move(value_copy), priority, false, -1);

		int    node_index;
		size_t current_size;

		// Get vector lock for modifications
		{
			std::lock_guard<std::mutex> lock(vector_mutex);

			// Get current size
			current_size = size_.load(std::memory_order_relaxed);

			// Ensure capacity
			if (current_size >= nodes.capacity()) { nodes.reserve(nodes.capacity() * 2); }

			// If necessary, expand the vector
			if (current_size >= nodes.size()) {
				nodes.push_back(new_node);
			} else {
				nodes[current_size] = new_node;
			}

			// Set the index
			node_index = current_size;

			// Increment size atomically
			size_.store(current_size + 1, std::memory_order_release);
		}

		// Now store the index in the node itself - safe to do outside the lock
		new_node->index.store(node_index, std::memory_order_release);

		// Memory barrier to ensure all previous operations are visible
		std::atomic_thread_fence(std::memory_order_release);

		// Now sift up the node - using our safer siftUp implementation
		siftUp(node_index);
	}

	/**
	 * Update the priority of an element with specific ID
	 */

	void update(int simID, uint64_t newPriority) {
		std::shared_ptr<Node> target_node  = nullptr;
		int                   target_index = -1;
		size_t                current_size = size_.load(std::memory_order_acquire);

		// Find the target node and its index - keep lock for consistency
		{
			std::lock_guard<std::mutex> lock(vector_mutex);
			for (size_t i = 0; i < current_size && i < nodes.size(); i++) {
				auto node = nodes[i];
				if (node && node->value.id == simID) {
					target_node  = node;
					target_index = node->index.load(std::memory_order_relaxed);  // Get index under lock
					break;
				}
			}
		}

		// If element not found, simply return
		if (!target_node) { return; }

		// Update the priority atomically
		uint64_t old_priority = target_node->priority.exchange(newPriority, std::memory_order_acq_rel);

		// Update the execution cycle in value
		// This is safe as long as target_node is valid (which it should be if we found it)
		target_node->value.next_execution_cycle = newPriority;

		// Mark for lazy update
		target_node->needsUpdate.store(true, std::memory_order_release);

		// Increment version counter to notify readers
		version_counter.fetch_add(1, std::memory_order_release);

		// Memory barrier before sift operations
		std::atomic_thread_fence(std::memory_order_acq_rel);

		// If new priority is less than old, sift up; otherwise, sift down
		// Use the index we captured under the lock
		if (target_index != -1) {
			if (newPriority < old_priority) {
				siftUp(target_index);
			} else if (newPriority > old_priority) {
				siftDown(target_index);
			}
		}
	}

	/**
	 * Check if there's a task ready at or before the given priority
	 */
	bool hasReadyTask(uint64_t priority) {
		// Process any pending updates - ensures heap property
		processUpdates();

		// Check if heap is empty
		size_t current_size = size_.load(std::memory_order_acquire);
		if (current_size == 0) return false;

		// Check the top element's priority
		std::shared_ptr<Node> top_node;
		{
			std::lock_guard<std::mutex> lock(vector_mutex);
			if (nodes.empty() || current_size == 0) return false;
			top_node = nodes[0];
		}

		if (!top_node) return false;

		return top_node->priority.load(std::memory_order_acquire) <= priority;
	}

	/**
	 * Get the top element without removing it
	 */
	T top() {
		// Process any pending updates
		processUpdates();

		// Check if heap is empty
		size_t current_size = size_.load(std::memory_order_acquire);
		if (current_size == 0) { throw std::out_of_range("Priority queue is empty"); }

		// Return the top element
		std::shared_ptr<Node> top_node;
		{
			std::lock_guard<std::mutex> lock(vector_mutex);
			if (nodes.empty() || current_size == 0) { throw std::out_of_range("Priority queue is empty"); }
			top_node = nodes[0];
		}

		if (!top_node) { throw std::out_of_range("Priority queue is empty"); }

		return top_node->value;
	}

	/**
	 * Pop the top element
	 */
	void pop() {
		// Process any pending updates
		processUpdates();

		std::unique_lock<std::mutex> lock(vector_mutex);

		// Check if heap is empty
		size_t current_size = size_.load(std::memory_order_acquire);
		if (current_size == 0 || nodes.empty()) { throw std::out_of_range("Priority queue is empty"); }

		// Get the last element
		auto last_node = nodes[current_size - 1];

		// Replace the root with the last element
		if (current_size > 1) {
			// Save the old root for cleanup
			auto old_root = nodes[0];

			// Move last node to root
			nodes[0] = last_node;
			last_node->index.store(0, std::memory_order_release);

			// Remove the last element
			nodes[current_size - 1] = nullptr;

			// Update size
			size_.store(current_size - 1, std::memory_order_release);

			// Release lock before sifting down
			lock.unlock();

			// Memory barrier
			std::atomic_thread_fence(std::memory_order_acq_rel);

			// Sift down the new root
			siftDown(0);
		} else {
			// Only one element, just remove it
			nodes[0] = nullptr;
			size_.store(0, std::memory_order_release);
		}
	}

	/**
	 * Try to get the top task if it's ready, returning true if successful
	 */
	bool tryGetReadyTask(uint64_t currentTick, T& outTask) {
		// Process any pending updates
		processUpdates();

		// Check if heap is empty
		size_t current_size = size_.load(std::memory_order_acquire);
		if (current_size == 0) return false;

		// Check if the top task is ready
		std::shared_ptr<Node> top_node;
		{
			std::lock_guard<std::mutex> lock(vector_mutex);
			if (nodes.empty() || current_size == 0) return false;
			top_node = nodes[0];
		}

		if (!top_node) return false;

		uint64_t top_priority = top_node->priority.load(std::memory_order_acquire);
		if (top_priority > currentTick) {
			return false;  // Not ready yet
		}

		// Task is ready, extract it safely
		try {
			// Copy the task first while we know it's valid
			outTask = top_node->value;

			// Then remove it from the queue
			pop();
			return true;
		} catch (const std::exception&) {
			return false;  // Pop failed
		}
	}

	/**
	 * Try to get any task from the queue (used during termination)
	 */
	bool tryGetAnyTask(T& outTask) {
		if (empty()) return false;

		try {
			// Lock to ensure consistent state
			std::unique_lock<std::mutex> lock(vector_mutex);

			// Check if heap is still non-empty
			size_t current_size = size_.load(std::memory_order_relaxed);
			if (current_size == 0 || nodes.empty()) return false;

			// Get top node
			auto top_node = nodes[0];
			if (!top_node) return false;

			// Copy the task
			outTask = top_node->value;

			// Keep lock during pop operations since we're already locked
			// Get the last element
			auto last_node = nodes[current_size - 1];

			// Replace the root with the last element
			if (current_size > 1) {
				// Move last node to root
				nodes[0] = last_node;
				last_node->index.store(0, std::memory_order_release);

				// Remove the last element
				nodes[current_size - 1] = nullptr;

				// Update size
				size_.store(current_size - 1, std::memory_order_release);
			} else {
				// Only one element, just remove it
				nodes[0] = nullptr;
				size_.store(0, std::memory_order_release);
			}

			// Release lock before sifting
			lock.unlock();

			// Memory barrier
			std::atomic_thread_fence(std::memory_order_acq_rel);

			// Sift down if we replaced the root
			if (current_size > 1) { siftDown(0); }

			return true;
		} catch (const std::exception&) { return false; }
	}

	/**
	 * Check if the priority queue is empty
	 */
	bool empty() const { return size_.load(std::memory_order_acquire) == 0; }

	/**
	 * Get the current size of the priority queue
	 */
	size_t size() const { return size_.load(std::memory_order_acquire); }

	/**
	 * Dump the content of the queue as a string
	 */
	std::string dump() {
		processUpdates();

		size_t current_size = size_.load(std::memory_order_acquire);
		if (current_size == 0) { return "Priority Queue is empty.\n"; }

		std::stringstream ss;
		ss << "Priority Queue Content: size=" << current_size << "\n";

		// Include information about the top element
		std::shared_ptr<Node> top_node;
		{
			std::lock_guard<std::mutex> lock(vector_mutex);
			if (!nodes.empty() && current_size > 0) { top_node = nodes[0]; }
		}

		if (top_node) { ss << "  - Top element data (ID: " << top_node->value.id << "): " << top_node->value << "\n"; }

		return ss.str();
	}

private:
	/**
	 * Helper methods for maintaining heap property
	 */
	int parent(int index) const { return (index - 1) / 2; }

	int leftChild(int index) const { return 2 * index + 1; }

	int rightChild(int index) const { return 2 * index + 2; }

	/**
	 * Sift up operation - move node up the heap as needed
	 * Improved for better lock-free safety
	 */
	void siftUp(int index) {
		size_t current_size = size_.load(std::memory_order_acquire);
		if (index >= current_size) return;

		while (index > 0) {
			int parent_idx = parent(index);

			// Get nodes safely
			std::shared_ptr<Node> current_node;
			std::shared_ptr<Node> parent_node;

			{
				std::lock_guard<std::mutex> lock(vector_mutex);
				if (index >= nodes.size() || parent_idx >= nodes.size()) { break; }

				current_node = nodes[index];
				parent_node  = nodes[parent_idx];

				if (!current_node || !parent_node) { break; }
			}

			// Load priorities atomically
			uint64_t current_priority = current_node->priority.load(std::memory_order_acquire);
			uint64_t parent_priority  = parent_node->priority.load(std::memory_order_acquire);

			// If heap property is satisfied, exit
			if (current_priority >= parent_priority) { break; }

			// Need to swap - lock for the swap operation
			bool swapped = false;
			{
				std::lock_guard<std::mutex> lock(vector_mutex);

				// Recheck conditions with lock
				if (index >= nodes.size() || parent_idx >= nodes.size()) { break; }

				current_node = nodes[index];
				parent_node  = nodes[parent_idx];

				if (!current_node || !parent_node) { break; }

				// Recheck priorities under lock
				current_priority = current_node->priority.load(std::memory_order_relaxed);
				parent_priority  = parent_node->priority.load(std::memory_order_relaxed);

				if (current_priority >= parent_priority) { break; }

				// Perform swap
				std::swap(nodes[index], nodes[parent_idx]);
				swapped = true;
			}

			if (swapped) {
				// Update indices atomically
				current_node->index.store(parent_idx, std::memory_order_release);
				parent_node->index.store(index, std::memory_order_release);

				// Memory barrier
				std::atomic_thread_fence(std::memory_order_release);

				// Continue upward
				index = parent_idx;
			} else {
				// If swap failed or wasn't needed, exit
				break;
			}
		}
	}

	/**
	 * Sift down operation - move node down the heap as needed
	 * Improved for better lock-free safety
	 */
	void siftDown(int index) {
		size_t current_size = size_.load(std::memory_order_acquire);
		if (index >= current_size) return;

		while (true) {
			int smallest = index;
			int left     = leftChild(index);
			int right    = rightChild(index);

			// Get nodes safely
			std::shared_ptr<Node> current_node;
			std::shared_ptr<Node> left_node;
			std::shared_ptr<Node> right_node;

			{
				std::lock_guard<std::mutex> lock(vector_mutex);

				// Re-check size under lock
				current_size = size_.load(std::memory_order_relaxed);

				if (index >= nodes.size() || index >= current_size) { return; }

				current_node = nodes[index];

				if (left < nodes.size() && left < current_size) { left_node = nodes[left]; }

				if (right < nodes.size() && right < current_size) { right_node = nodes[right]; }
			}

			if (!current_node) return;

			uint64_t current_priority = current_node->priority.load(std::memory_order_acquire);

			// Check if left child is smaller
			if (left_node) {
				uint64_t left_priority = left_node->priority.load(std::memory_order_acquire);
				if (left_priority < current_priority) {
					smallest         = left;
					current_priority = left_priority;
				}
			}

			// Check if right child is smaller
			if (right_node) {
				uint64_t right_priority = right_node->priority.load(std::memory_order_acquire);
				if (right_priority < current_priority) { smallest = right; }
			}

			// If no change needed, break
			if (smallest == index) { break; }

			// Need to swap - lock for the swap operation
			bool                  swapped = false;
			std::shared_ptr<Node> smallest_node;

			{
				std::lock_guard<std::mutex> lock(vector_mutex);

				// Re-check conditions with lock
				current_size = size_.load(std::memory_order_relaxed);

				if (index >= nodes.size() || smallest >= nodes.size() || index >= current_size ||
				    smallest >= current_size) {
					break;
				}

				current_node  = nodes[index];
				smallest_node = nodes[smallest];

				if (!current_node || !smallest_node) { break; }

				// Recheck priorities under lock
				current_priority           = current_node->priority.load(std::memory_order_relaxed);
				uint64_t smallest_priority = smallest_node->priority.load(std::memory_order_relaxed);

				if (current_priority <= smallest_priority) { break; }

				// Perform swap
				std::swap(nodes[index], nodes[smallest]);
				swapped = true;
			}

			if (swapped) {
				// Update indices atomically
				current_node->index.store(smallest, std::memory_order_release);
				smallest_node->index.store(index, std::memory_order_release);

				// Memory barrier
				std::atomic_thread_fence(std::memory_order_release);

				// Continue downward
				index = smallest;
			} else {
				// If swap failed or wasn't needed, exit
				break;
			}
		}
	}

	/**
	 * Process any pending updates at the top of the heap
	 * Enhanced for lock-free safety
	 */
	void processUpdates() {
		size_t current_size = size_.load(std::memory_order_acquire);
		if (current_size == 0) return;

		while (true) {
			// Re-check size on each iteration
			current_size = size_.load(std::memory_order_acquire);
			if (current_size == 0) break;

			std::shared_ptr<Node> top_node;
			{
				std::lock_guard<std::mutex> lock(vector_mutex);
				if (nodes.empty() || current_size == 0) break;
				top_node = nodes[0];
			}

			if (!top_node) break;

			bool needs_update = top_node->needsUpdate.load(std::memory_order_acquire);
			if (!needs_update) break;

			// Clear update flag atomically
			if (!top_node->needsUpdate.compare_exchange_strong(needs_update, false, std::memory_order_acq_rel,
			                                                   std::memory_order_acquire)) {
				// Someone else already updated it
				continue;
			}

			// Re-sift this element to maintain heap property
			siftDown(0);
		}
	}
};

}  // namespace acalsim
