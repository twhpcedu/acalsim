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

#include <sstream>
#include <vector>

namespace acalsim {

template <typename T>
class UpdateablePriorityQueue {
public:
	struct Element {
		T        value;
		uint64_t priority;
		bool     needsUpdate;
		int      index;  // Store the element's index in the heap

		Element(const T& val, int prio, bool update, int idx)
		    : value(val), priority(prio), needsUpdate(update), index(idx) {}
	};

	bool compare(const Element& a, const Element& b) {
		// Return true if element a has a lower score than element b
		return a.priority < b.priority;
	}

	// Default constructor with appropriate arguments
	UpdateablePriorityQueue() : data_(), needsUpdate_(false) {}

	// Push an element with its priority
	void push(const T& value, uint64_t priority) {
		data_.emplace_back(value, priority, false, data_.size());  // Store index directly
		siftUp(data_.back().index);                                // Update position using stored index
	}

	// Update the priority of an element with a specific value (O(1) amortized)
	void update(int simID, uint64_t newPriority) {
		auto it = findElement(simID);
		if (it == data_.end()) { throw std::out_of_range("Element not found in the queue"); }
		it->priority                   = newPriority;
		it->value.next_execution_cycle = newPriority;
		it->needsUpdate                = true;  // Mark for lazy update
		siftUp(it->index);                      // Update position directly using stored index
	}

	bool hasReadyTask(uint64_t priority) {
		processUpdates();  // Handle lazy updates before returning the top element
		return (!empty() && data_.front().priority <= priority);
	}

	// Extract the element with the highest priority (considering lazy updates)
	T top() {
		if (empty()) { throw std::out_of_range("Priority queue is empty"); }
		processUpdates();  // Handle lazy updates before returning the top element
		return data_.front().value;
	}

	// Remove the element with the highest priority (considering lazy updates)
	void pop() {
		if (empty()) { throw std::out_of_range("Priority queue is empty"); }
		processUpdates();  // Handle lazy updates before removing the top element
		std::swap(data_.front(), data_.back());
		data_.back().index = -1;  // Mark removed element's index as invalid
		data_.pop_back();
		if (!empty()) { siftDown(0); }
	}

	// Check if the priority queue is empty
	bool empty() const { return data_.empty(); }

	size_t size() const { return data_.size(); }

	// Dump the content of the queue as a string
	std::string dump() {
		if (empty()) { return "Priority Queue is empty.\n"; }

		std::stringstream ss;
		ss << "Priority Queue Content: size=" << data_.size() << "\n";

		// Include information about the top element (optional)
		ss << "  - Top element data (ID: " << data_.front().value.id << "): " << data_.front().value << "\n";

		/*for (const auto& element : data_) {
		    ss << "  - Element (@" << &(element.value) << "): " << element.value << "\n";
		    ss << "    - Priority: " << element.priority << "\n";
		    ss << "    - index: " << element.index << "\n";
		    ss << "    - needsUpdate: " << element.needsUpdate << "\n";
		  }*/

		return ss.str();
	}

private:
	std::vector<Element> data_;
	// Alternative using unordered_set for faster average-case lookups (optional)
	// std::unordered_set<T> indices_;
	bool needsUpdate_;  // Flag for lazy update tracking

	int parent(int index) const { return (index - 1) / 2; }

	int leftChild(int index) const { return 2 * index + 1; }

	int rightChild(int index) const { return 2 * index + 2; }

	void siftUp(int index) {
		while (index > 0 && compare(data_[index], data_[parent(index)])) {
			std::swap(data_[index], data_[parent(index)]);
			data_[index].index         = index;
			data_[parent(index)].index = parent(index);
			index                      = parent(index);
		}
	}

	void siftDown(int index) {
		int maxIndex = index;
		int left     = leftChild(index);
		int right    = rightChild(index);
		if (left < data_.size() && compare(data_[left], data_[maxIndex])) { maxIndex = left; }
		if (right < data_.size() && compare(data_[right], data_[maxIndex])) { maxIndex = right; }
		if (index != maxIndex) {
			std::swap(data_[index], data_[maxIndex]);
			data_[index].index    = index;
			data_[maxIndex].index = maxIndex;  // Update swapped element's index as well
			siftDown(maxIndex);                // Recursively sift down the swapped element if needed
		}
	}

	// Find the iterator pointing to the element with the specified value (optional)
	typename std::vector<Element>::iterator findElement(int simID) {
		for (auto it = data_.begin(); it != data_.end(); ++it) {
			if (it->value.id == simID) { return it; }
		}
		return data_.end();  // Element not found
	}

	void processUpdates() {
		while (!empty() && data_.front().needsUpdate) {
			data_.front().needsUpdate = false;  // Clear flag after processing
			int index                 = 0;
			siftUp(index);  // Move the updated element to its correct position
		}
	}
};

}  // namespace acalsim
