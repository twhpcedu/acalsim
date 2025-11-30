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

#include <atomic>
#include <condition_variable>

// ACALSim
#include "sim/TaskManager.hh"
#include "sim/ThreadManager.hh"
#include "sim/ThreadManagerV1/TaskQueueV1.hh"
#include "sim/ThreadManagerV1/TaskV1.hh"

namespace acalsim {

// class definition forwarding
template <typename T>
class ThreadManagerV1;

template <typename TFriend>
class TaskManagerV1 : public TaskManager {
	friend class ThreadManagerV1<TFriend>;

public:
	TaskManagerV1(std::string _name)
	    : TaskManager(_name),
	      allThreadsDone(false),
	      startPhase1(false),
	      nWakeupThreads(0),
	      pendingInboundRequests(false) {}

	virtual ~TaskManagerV1() {}

	void promoteTaskToTop(int simID);

	/**
	 * @brief Terminate a thread and Increment the number of finished threads
	 */
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
	Tick getNextSimTick() override { return this->taskQueue.top().next_execution_cycle; }

	/**
	 * @brief Get a task that is ready for execution.
	 * @return A shared pointer to the ready task.
	 */
	std::shared_ptr<Task> getReadyTask() override { return nullptr; }

	/**
	 * @brief task scheduler for each worker thread
	 */
	void scheduler(const size_t _tidx) override;

	/**
	 * @brief Initialize the task manager after worker threads are launched.
	 */
	void init() override;

	ThreadManagerV1<TFriend>* getThreadManager() const override {
		return dynamic_cast<ThreadManagerV1<TFriend>*>(this->threadManager);
	}

	void setPendingInboundRequests() { this->pendingInboundRequests = true; }
	void clearPendingInboundRequests() { this->pendingInboundRequests = false; }

protected:
	/**
	 * @brief task priority queue sorted by the next execution time
	 */
	UpdateablePriorityQueue<TaskV1> taskQueue;

	/**
	 * @brief mutex for the taskQueue
	 */
	std::mutex taskQueueMutex;

	/**
	 * @brief mutex for number of finished threads
	 */
	std::mutex nFinishedThreadsMutex;

	/**
	 * @brief number of wakeup threads
	 */
	int nWakeupThreads;

	/**
	 * @brief mutex for task scheduler & control thread synchronization
	 */
	std::mutex taskAvailableMutex;

	/**
	 * @brief conditional variable for finished threads
	 */
	std::condition_variable cvFinishedThreads;

	std::condition_variable workerThreadsDoneCondVar;
	std::atomic<bool>       allThreadsDone;

	std::condition_variable_any newTaskAvailableCondVar;
	std::atomic<bool>           startPhase1;

	// Condition variable for worker threads to wait until ThreadManager is running
	std::condition_variable runningCondVar;
	std::mutex              runningMutex;

	// Counter for threads currently sleeping/waiting on newTaskAvailableCondVar
	// Used to optimize notify_all -> notify_one to reduce thundering herd
	std::atomic<int> nSleepingThreads{0};

	// flag for pending inbound requests in the framework
	bool pendingInboundRequests;

private:
	ThreadManagerV1<TFriend>* getThreadManager() { return static_cast<ThreadManagerV1<TFriend>*>(this->threadManager); }
};

}  // end of namespace acalsim

#include "sim/ThreadManagerV1/TaskManagerV1.inl"
