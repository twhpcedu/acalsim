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

#include <cstddef>
#include <functional>
#include <map>
#include <queue>

#include "ACALSim.hh"

namespace test_port {

class OutStandingReqQueue : protected acalsim::UnorderedRequestQueue<bool> {
public:
	explicit OutStandingReqQueue(size_t max_size = SIZE_MAX)
	    : acalsim::UnorderedRequestQueue<bool>(), max_size_(max_size), counter_(0) {}
	~OutStandingReqQueue() {}

	bool contains(int _req_id) { return this->acalsim::UnorderedRequestQueue<bool>::contains(_req_id); }

	bool add(int _req_id, bool _obj) {
		bool stat = isPushReady();
		if (isPushReady()) {
			this->counter_++;
			this->acalsim::UnorderedRequestQueue<bool>::add(_req_id, _obj);
		}
		return stat;
	}

	void remove(int _req_id) {
		this->counter_--;
		this->acalsim::UnorderedRequestQueue<bool>::remove(_req_id);
	}

	bool isPushReady() const { return this->counter_ < max_size_; }

	size_t size() const { return counter_; }

private:
	const size_t max_size_;

	size_t counter_;
};

}  // namespace test_port
