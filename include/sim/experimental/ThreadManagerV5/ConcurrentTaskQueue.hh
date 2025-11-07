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

#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <vector>

#include "profiling/Synchronization.hh"
#include "profiling/Utils.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

/**
 * A fine-grained concurrent priority queue implementation that maintains
 * the same behavior as UpdateablePriorityQueue while reducing contention.
 */
template <typename T>
class ConcurrentUpdateablePriorityQueue {
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

	// Default constructor
	ConcurrentUpdateablePriorityQueue() : data_(), needsUpdate_(false) {}

	// Push an element with its priority
	template <bool EnableProfiling = false, ConstexprStr PhaseName = "">
	void push(const T& value, uint64_t priority) {
		using Lock =
		    std::conditional_t</* cond */ EnableProfiling,
		                       /* true */
		                       ProfiledLock<"TaskManagerV5-TaskQueue-" + PhaseName + "-Unique",
		                                    std::unique_lock<std::shared_mutex>, ProfileMode::ACALSIM_STATISTICS_FLAG>,
		                       /* false */ std::unique_lock<std::shared_mutex>>;

		// Lock the entire structure for push operations
		Lock lock(mutex_);

		MEASURE_TIME_MICROSECONDS(/* var_name */ tq_manip,
		                          /* code_block */ {
			                          data_.emplace_back(value, priority, false, data_.size());  // Store index directly
			                          siftUp(data_.back().index);  // Update position using stored index
		                          });

#ifdef ACALSIM_STATISTICS
		if constexpr (EnableProfiling) { this->tqManipStatistics.push(tq_manip_lat); }
#endif  // ACALSIM_STATISTICS
	}

	// Update the priority of an element with a specific value
	void update(int simID, uint64_t newPriority) {
		// Lock the entire structure for update operations
		std::unique_lock<std::shared_mutex> lock(mutex_);

		auto it = findElement(simID);
		if (it == data_.end()) { throw std::out_of_range("Element not found in the queue"); }

		it->priority                   = newPriority;
		it->value.next_execution_cycle = newPriority;
		it->needsUpdate                = true;  // Mark for lazy update

		// If this is the top element, we won't need to sift
		if (it->index > 0) {
			siftUp(it->index);  // Update position directly using stored index
		}
	}

	template <bool EnableProfiling = false, ConstexprStr PhaseName = "">
	bool hasReadyTask(uint64_t priority) {
		using Lock =
		    std::conditional_t</* cond */ EnableProfiling,
		                       /* true */
		                       ProfiledLock<"TaskManagerV5-TaskQueue-" + PhaseName + "-Shared",
		                                    std::shared_lock<std::shared_mutex>, ProfileMode::ACALSIM_STATISTICS_FLAG>,
		                       /* false */ std::shared_lock<std::shared_mutex>>;

		// Read-only operation can use shared lock
		Lock lock(mutex_);

		bool result = false;

		MEASURE_TIME_MICROSECONDS(/* var_name */ tq_manip,
		                          /* code_block */ {
			                          processUpdates();  // Handle lazy updates before checking
			                          result = (!empty() && data_.front().priority <= priority);
		                          });

#ifdef ACALSIM_STATISTICS
		if constexpr (EnableProfiling) this->tqManipStatistics.push(tq_manip_lat);
#endif  // ACALSIM_STATISTICS

		return result;
	}

	// Extract the element with the highest priority (considering lazy updates)
	T top() {
		// Read-only operation can use shared lock
		std::shared_lock<std::shared_mutex> lock(mutex_);

		if (empty()) { throw std::out_of_range("Priority queue is empty"); }

		processUpdates();  // Handle lazy updates before returning the top element
		return data_.front().value;
	}

	// Remove the element with the highest priority (considering lazy updates)
	void pop() {
		// Need exclusive lock for modification
		std::unique_lock<std::shared_mutex> lock(mutex_);

		if (empty()) { throw std::out_of_range("Priority queue is empty"); }

		processUpdates();  // Handle lazy updates before removing the top element

		std::swap(data_.front(), data_.back());
		data_.back().index = -1;  // Mark removed element's index as invalid
		data_.pop_back();

		if (!empty()) { siftDown(0); }
	}

	// Check if the priority queue is empty
	bool empty() const {
		// No lock needed for atomic check
		return data_.empty();
	}

	size_t size() const {
		// No lock needed for atomic size check
		return data_.size();
	}

	// Try to get the top task if it's ready, returning true if successful
	template <bool EnableProfiling = false, ConstexprStr PhaseName = "">
	bool tryGetReadyTask(uint64_t currentTick, T& outTask) {
		using Lock =
		    std::conditional_t</* cond */ EnableProfiling,
		                       /* true */
		                       ProfiledLock<"TaskManagerV5-TaskQueue-" + PhaseName + "-Unique",
		                                    std::unique_lock<std::shared_mutex>, ProfileMode::ACALSIM_STATISTICS_FLAG>,
		                       /* false */ std::unique_lock<std::shared_mutex>>;

		// Need exclusive lock since we might modify the queue
		Lock lock(mutex_);

		if (empty()) { return false; }

		bool has_ready_task;

		MEASURE_TIME_MICROSECONDS(/* var_name */ tq_manip,
		                          /* code_block */ {
			                          processUpdates();  // Handle lazy updates first

			                          has_ready_task = data_.front().priority <= currentTick;

			                          if (has_ready_task) {
				                          // Task is ready, extract it
				                          outTask = data_.front().value;

				                          // Remove it from the queue
				                          std::swap(data_.front(), data_.back());
				                          data_.back().index = -1;
				                          data_.pop_back();

				                          if (!empty()) { siftDown(0); }
			                          }
		                          });

#ifdef ACALSIM_STATISTICS
		if constexpr (EnableProfiling) this->tqManipStatistics.push(tq_manip_lat);
#endif  // ACALSIM_STATISTICS

		return has_ready_task;
	}

	// Dump the content of the queue as a string
	std::string dump() {
		// Read-only operation can use shared lock
		std::shared_lock<std::shared_mutex> lock(mutex_);

		if (empty()) { return "Priority Queue is empty.\n"; }

		std::stringstream ss;
		ss << "Priority Queue Content: size=" << data_.size() << "\n";

		// Include information about the top element
		ss << "  - Top element data (ID: " << data_.front().value.id << "): " << data_.front().value << "\n";

		return ss.str();
	}
	/**
	 * Get any task from the queue regardless of its execution time
	 * Used during termination when we need to drain the queue
	 */
	template <bool EnableProfiling = false, ConstexprStr PhaseName = "">
	bool tryGetAnyTask(T& outTask) {
		using Lock =
		    std::conditional_t</* cond */ EnableProfiling,
		                       /* true */
		                       ProfiledLock<"TaskManagerV5-TaskQueue-" + PhaseName + "-Unique",
		                                    std::unique_lock<std::shared_mutex>, ProfileMode::ACALSIM_STATISTICS_FLAG>,
		                       /* false */ std::unique_lock<std::shared_mutex>>;

		// Need exclusive lock since we modify the queue
		Lock lock(mutex_);

		if (empty()) { return false; }

		MEASURE_TIME_MICROSECONDS(/* var_name */ tq_manip,
		                          /* code_block */ {
			                          processUpdates();  // Handle lazy updates first

			                          // Get the top task regardless of its priority
			                          outTask = data_.front().value;

			                          // Remove it from the queue
			                          std::swap(data_.front(), data_.back());
			                          data_.back().index = -1;
			                          data_.pop_back();

			                          if (!empty()) { siftDown(0); }
		                          });

#ifdef ACALSIM_STATISTICS
		if constexpr (EnableProfiling) this->tqManipStatistics.push(tq_manip_lat);
#endif  // ACALSIM_STATISTICS

		return true;
	}

private:
	std::vector<Element>      data_;
	mutable std::shared_mutex mutex_;        // Reader-writer lock for fine-grained concurrency
	bool                      needsUpdate_;  // Flag for lazy update tracking

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

	// Find the iterator pointing to the element with the specified value
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
			siftDown(index);  // Move the updated element to its correct position
		}
	}

#ifdef ACALSIM_STATISTICS
public:
	double getUniqueLockWaitingTime() const {
		return NamedTimer<"TaskManagerV5-TaskQueue-Phase1-Unique">::getTimerVal();
	}

	double getSharedLockWaitingTime() const {
		return NamedTimer<"TaskManagerV5-TaskQueue-Phase1-Shared">::getTimerVal();
	}

	double getTqManipTime() const { return this->tqManipStatistics.sum(); }

private:
	// The time spent manipulating the task queue
	Statistics<double, StatisticsMode::Accumulator, true> tqManipStatistics;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim
