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

// ACALSim
#include <cstddef>

#include "sim/Task.hh"
#include "sim/TaskManager.hh"
#include "sim/ThreadManager.hh"
#include "sim/ThreadManagerV1/ThreadManagerV1.hh"

namespace acalsim {

// class definition forwarding
template <typename T>
class ThreadManagerV4;

template <typename TFriend>
class TaskManagerV4 : public TaskManager {
	friend class ThreadManagerV4<TFriend>;

public:
	TaskManagerV4(std::string _name) : TaskManager(_name) {}

	virtual ~TaskManagerV4() {}

	// TaskManagerV4 is not used at all. Only override these methods since they are pure virtual in TaskManager.
	void terminateThread() override { ASSERT(false); }

	void addTask(const std::shared_ptr<Task>& task) override { ASSERT(false); }

	std::shared_ptr<Task> getReadyTask() override {
		ASSERT(false);
		return nullptr;
	}

	Tick getNextSimTick() override {
		ASSERT(false);
		return 0;
	}

	void scheduler(const size_t _tidx) override { ASSERT(false); }

protected:
private:
	ThreadManagerV1<TFriend>* getThreadManager() { return static_cast<ThreadManagerV1<TFriend>*>(this->threadManager); }
};

}  // end of namespace acalsim

#include "sim/experimental/ThreadManagerV4/TaskManagerV4.inl"
