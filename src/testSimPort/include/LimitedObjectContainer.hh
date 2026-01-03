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
#include <functional>
#include <map>
#include <queue>

#include "ACALSim.hh"

namespace test_port {

template <typename T>
class LimitedObjectContainer : public acalsim::HashableType {
private:
	struct LimitedObject {
		T             object;
		acalsim::Tick timestamp;

		// Reverse comparison for min-heap behavior
		bool operator<(const LimitedObject& obj) const {
			return timestamp > obj.timestamp;  // Smaller timestamp has higher priority
		}
	};

	size_t                             max_size_;
	std::priority_queue<LimitedObject> object_list_;  // Min-heap based on timestamp

	std::atomic<bool> has_push_object_flag;
	std::atomic<bool> has_pop_object_flag;

public:
	// Constructor to initialize with a maximum size
	LimitedObjectContainer(size_t max_size = SIZE_MAX)
	    : max_size_(max_size), has_push_object_flag(false), has_pop_object_flag(false) {}

	bool push(T object, acalsim::Tick _when) {
		if (!isPushReady()) { return false; }
		object_list_.push({object, _when});
		this->has_push_object_flag.store(true);
		return true;
	}

	T front() const {
		if (empty()) { return nullptr; }
		return object_list_.top().object;
	}

	acalsim::Tick getMinTick() const {
		if (empty()) { return acalsim::Tick(UINT64_MAX); }
		return object_list_.top().timestamp;
	}

	T pop() {
		if (!isPopValid()) { return nullptr; }
		T object = object_list_.top().object;
		object_list_.pop();
		has_pop_object_flag.store(true);
		return object;
	}

	bool empty() const { return object_list_.empty(); }

	// Checks if a new object can be pushed
	bool isPushReady() const { return size() < max_size_ && !has_push_object_flag.load(); }

	// Checks if the pool has any objects to pop
	bool isPopValid() const { return !empty() && !has_pop_object_flag.load(); }

	// Retrieves the current size of the pool
	size_t size() const { return object_list_.size(); }

	// make sure each cycle can only push 1 object in this Queue
	void step() {
		this->has_push_object_flag.store(false);
		this->has_pop_object_flag.store(false);
	}
};

}  // namespace test_port
