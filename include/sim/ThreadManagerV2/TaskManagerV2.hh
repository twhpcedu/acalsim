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

/**
 * @file TaskManagerV2.hh
 * @brief Defines the TaskManagerV2 class for managing and scheduling tasks.
 *
 * This file contains the definition of the TaskManagerV2 class, which is responsible
 * for managing tasks and scheduling their execution. It inherits from TaskManager
 * and includes methods for adding tasks, scheduling, and executing task phases.
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>

// ACALSim
#include "sim/TaskManager.hh"
#include "sim/ThreadManagerV2/TaskQueueV2.hh"
#include "sim/ThreadManagerV2/TaskV2.hh"

namespace acalsim {

template <typename T>
class ThreadManagerV2;

/**
 * @class TaskManagerV2
 * @brief A class that manages tasks and schedules them for execution.
 */

template <typename TFriend>
class TaskManagerV2 : public TaskManager {
	friend class ThreadManagerV2<TFriend>;

public:
	/**
	 * @brief Constructor for TaskManagerV2.
	 * @param _name The name of the TaskManagerV2 instance.
	 */
	TaskManagerV2(std::string _name) : TaskManager(_name) { ; }

	~TaskManagerV2() {}

	/**
	 * @brief Initialize the task manager after worker threads are launched.
	 */
	void init() override;

	/// @brief Terminate a thread and Increment the number of finished threads
	void terminateThread() override;

	/**
	 * @brief Add a Task object to the TaskManagerV2.
	 * @param task The shared pointer to the Task object to be added.
	 */
	void addTask(const std::shared_ptr<Task>& task) override;

	/**
	 * @brief Get the next simulation tick.
	 * @return The next execution cycle tick.
	 */
	Tick getNextSimTick() override { return this->taskQueue.top()->next_execution_cycle; }

	/**
	 * @brief Get a task that is ready for execution.
	 * @return A shared pointer to the ready task.
	 */
	std::shared_ptr<Task> getReadyTask() override;

	/// @brief Task scheduler for each worker thread.
	void scheduler(const size_t _tidx) override;

	ThreadManagerV2<TFriend>* getThreadManager() const override {
		return dynamic_cast<ThreadManagerV2<TFriend>*>(this->threadManager);
	}

	/**
	 * @brief Promote a task to the top of the TaskQueueV2.
	 * @param task The shared pointer to the Task object to be promoted.
	 */
	void promoteTaskToTop(std::shared_ptr<TaskV2> task);

	/**
	 * @brief Execute phase 1 of the task manager's operation.
	 * @param readyToTerminate A flag indicating if the task manager is ready to terminate.
	 * @param tid Identifier for the task to be executed
	 */
	void execPhase1(bool readyToTerminate, uint64_t tidx);

	/**
	 * @brief Execute phase 2 of the task manager's operation.
	 * @param readyToTerminate A flag indicating if the task manager is ready to terminate.
	 * @param tid Identifier for the task to be executed
	 */
	void execPhase2(bool readyToTerminate, uint64_t tidx);

protected:
	TaskQueueV2 taskQueue;  // Task priority queue sorted by the next execution time.

	std::mutex taskQueueMutex;  // Mutex for the taskQueue.

private:
	ThreadManagerV2<TFriend>* getThreadManager() { return static_cast<ThreadManagerV2<TFriend>*>(this->threadManager); }
};

}  // end of namespace acalsim

#include "sim/ThreadManagerV2/TaskManagerV2.inl"
