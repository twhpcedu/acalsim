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

#include <vector>

// ACALSim
#include "common/PriorityQueue.hh"
#include "sim/SimBase.hh"
#include "sim/TaskManager.hh"

namespace acalsim {

class SimBase;

template <typename T>
class ThreadManagerV7;

template <typename TFriend>
class TaskManagerV7 : public TaskManager {
	friend class ThreadManagerV7<TFriend>;

public:
	TaskManagerV7(std::string _name) : TaskManager(_name) { ; }

	void init() override;

	void addTask(const std::shared_ptr<Task>& task) override { ; }

	std::shared_ptr<Task> getReadyTask() override { return nullptr; }

	/*
	 * @brief task scheduler for each worker thread
	 */
	void scheduler(const size_t _tidx) override;

	/**
	 * @brief Get the next simulation tick.
	 * @return The next execution cycle tick.
	 */
	Tick getNextSimTick() override;

	/**
	 * @brief Terminate a thread and Increment the number of finished threads
	 */
	void terminateThread() override { ++this->nFinishedThreads; }

	ThreadManagerV7<TFriend>* getThreadManager() const override {
		return dynamic_cast<ThreadManagerV7<TFriend>*>(this->threadManager);
	}

private:
	std::vector<std::vector<SimBase*>>         localSimBaseVec;
	std::vector<PriorityQueue<Tick, SimBase*>> localTaskQueue;
};

}  // namespace acalsim

#include "sim/experimental/ThreadManagerV7/TaskManagerV7.inl"
