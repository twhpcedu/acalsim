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

#include <sstream>
#include <string>

#include "external/gem5/Event.hh"
#include "sim/Task.hh"

namespace acalsim {

class TaskV5 : public Task {
	friend std::ostream& operator<<(std::ostream& os, const TaskV5& t);

public:
	TaskV5() : next_execution_cycle(0), id(0) { ; }
	~TaskV5() { ; }

	// Constructor to initialize the task
	TaskV5(SimBase* base, std::string name) {
		functor.simbase      = base;
		functor.simBaseName  = name;
		next_execution_cycle = 0;
		id                   = functor.getSimBaseId();
	}

	// Operator() to execute the task
	void operator()() {
		functor();
		updateNextTick();
	}

	void updateNextTick() {
		if (!functor.isReadyToTerminate()) {
			next_execution_cycle = functor.getSimNextTick();
		} else {
			next_execution_cycle = functor.getGlobalTick() + 1;
		}
	}

	std::string getSimBaseName() const { return this->functor.getSimBaseName(); }

	bool operator<(const TaskV5& other) const { return next_execution_cycle < other.next_execution_cycle; }
	bool operator==(const TaskV5& other) const { return next_execution_cycle == other.next_execution_cycle; }

	// Function to execute
	TaskFunctor functor;

	// Next execution time
	Tick next_execution_cycle;

	int id;
};

inline std::ostream& operator<<(std::ostream& os, const TaskV5& t) {
	std::stringstream ss;
	ss << "{next_execution_cycle: " << t.next_execution_cycle << ",";
	ss << " id: " << t.id << ",";
	ss << " simBaseName: " << t.functor.getSimBaseName() << "}";
	os << ss.str();
	return os;
}

}  // namespace acalsim
