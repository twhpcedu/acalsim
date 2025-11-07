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

#include <barrier>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

// ACALSim
#include "common/PriorityQueue.hh"
#include "sim/TaskManager.hh"
#include "sim/ThreadManager.hh"

namespace acalsim {

// class definition forwarding
class SimBase;

template <typename T>
class ThreadManagerV3;

template <typename TFriend>
class TaskManagerV3 : public TaskManager {
	friend class ThreadManagerV3<TFriend>;

public:
	/**
	 * @class ThreadManager::TaskList
	 * @brief A class that manages a list of tasks to be executed by threads, with synchronization mechanisms.
	 */
	class TaskList {
		friend class TaskManagerV3<TFriend>;

	public:
		using SharedPtr = std::shared_ptr<TaskList>;

	public:
		/**
		 * @brief Constructs a TaskList object.
		 * @param _nWaitThread The number of threads to wait for synchronization. Default is 1.
		 */
		TaskList(size_t _nWaitThread = 1) : barrierPtr(new std::barrier<void (*)(void)>(_nWaitThread + 1, []() { ; })) {
			;
		}

		~TaskList() { delete this->barrierPtr; }

		/**************************************************
		 *                                                *
		 *           Methods for Thread Runners           *
		 *                                                *
		 **************************************************/

		void preProcessRoutine();

		/**
		 * @brief Acquires a task from the task list for execution.
		 * @return A function object representing the task to be executed.
		 */
		virtual std::function<void(void)> acquire();

		/**
		 * @brief Executes the routine after a task is finished.
		 */
		void finish();

		/**
		 * @brief Checks if the task list is empty.
		 * @return True if there are no tasks to be acquired, false otherwise.
		 */
		bool empty() const;

		/**
		 * @brief Get the amount of tasks in this task list.
		 * @return The size of the task list.
		 */
		size_t size() const;

		/**
		 * @brief Checks if the task list has no available tasks.
		 * @return True if there are no tasks can be executed, false otherwise.
		 */
		bool hasNoAvailTasks() const;

		/**************************************************
		 *                                                *
		 *        Methods for Task List Producers         *
		 *                                                *
		 **************************************************/

		/**
		 * @brief Inserts a new task into the task list.
		 * @param _task A function object representing the task to be inserted.
		 */
		void insert(std::function<void(void)> _task);

		/**
		 * @brief Clears all tasks from the task list.
		 */
		void clear();

		/**
		 * @brief Waits for all tasks to be completed.
		 */
		void wait();

	private:
		std::vector<std::function<void(void)>> tasks;

		std::atomic<size_t> dispatchedIdx = 0;
		std::atomic<size_t> finishedCnt   = 0;

		std::barrier<void (*)(void)>* barrierPtr;
	};

	class TaskV3 : public Task {
	public:
		using SharedPtr = std::shared_ptr<TaskV3>;

	public:
		TaskV3(typename TaskList::SharedPtr _list, std::function<void(void)> _task) : list(_list), task(_task) { ; }

		typename TaskList::SharedPtr list;
		std::function<void(void)>    task;
	};

public:
	TaskManagerV3(std::string _name) : TaskManager(_name) { ; }

	/**
	 * @brief Terminate a thread and Increment the number of finished threads
	 */
	void terminateThread() override;

	/**
	 * @note Doesn't support the execution frequency and period analysis of simulators for the time being
	 */
	void init() override { this->TaskManager::init(); }

	/**
	 * @brief Add a Task object to the TaskManagerV3.
	 * @param task The shared pointer to the Task object to be added.
	 */
	void addTask(const std::shared_ptr<Task>& task) override { ; }

	/**
	 * @brief Get the next simulation tick.
	 * @return The next execution cycle tick.
	 */
	Tick getNextSimTick() override;

	/**
	 * @brief Get a task that is ready for execution.
	 * @return A shared pointer to the ready task.
	 */
	std::shared_ptr<Task> getReadyTask() override;

	/**
	 * @brief Checks if there is no available tasks.
	 * @return True if there are no tasks can be executed, false otherwise.
	 */
	bool noReadyTask() const;

	/*
	 * @brief task scheduler for each worker thread
	 */
	void scheduler(const size_t _tidx) override;

	ThreadManagerV3<TFriend>* getThreadManager() const override {
		return dynamic_cast<ThreadManagerV3<TFriend>*>(this->threadManager);
	}

	/**
	 * @brief Submits a list of tasks to be executed by the thread pool.
	 * @param _tasks A shared pointer to a TaskList containing the tasks to be executed.
	 */
	void submitTasks(typename TaskList::SharedPtr _tasks);

	/**
	 * @brief Sets the next tick for the given simulator.
	 *
	 * @param _sim Pointer to the simulator instance.
	 * @param _tick The tick value indicating the next scheduled time for the simulator.
	 */
	void setSimulatorNextTick(SimBase* _sim, Tick _tick) {
		// std::lock_guard<std::mutex> lock(this->simulatorNextTickQueueMutex);
		this->simulatorNextTickQueue.insert(_sim, _tick);
	}

protected:
	std::queue<typename TaskList::SharedPtr> taskListQueue;
	mutable std::mutex                       taskListQueueMutex;

	std::condition_variable_any runnerCv;

	PriorityQueue<Tick, SimBase*> simulatorNextTickQueue;
	std::mutex                    simulatorNextTickQueueMutex;
};

}  // end of namespace acalsim

#include "sim/ThreadManagerV3/TaskManagerV3.inl"
