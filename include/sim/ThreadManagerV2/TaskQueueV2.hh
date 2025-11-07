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

/**
 * @file TaskQueueV2.hh
 * @brief Defines the TaskQueueV2 class, which manages a collection of tasks based on their execution cycle and ID.
 *
 * This file provides functionalities to add, remove, update, and access tasks in a priority queue manner.
 */

#pragma once

#include <memory>
#include <set>
#include <unordered_map>

#include "sim/ThreadManagerV2/TaskV2.hh"

namespace acalsim {

/**
 * @brief The TaskQueueV2 class manages a collection of tasks.
 *
 * This class provides functionality to add, remove, update, and access tasks
 * based on their execution cycle and ID.
 */
class TaskQueueV2 {
public:
	TaskQueueV2() {}
	~TaskQueueV2() {}

	/**
	 * @brief Adds a task to the queue.
	 *
	 * @param task A shared pointer to the task to be added.
	 */
	void add(std::shared_ptr<TaskV2> task) {
		this->s.emplace(std::make_pair(task->next_execution_cycle, task->id));
		this->m.emplace(task->id, task);
	}

	/**
	 * @brief Adds a task to the queue based on its execution cycle and ID.
	 *
	 * @param task A shared pointer to the task to be added.
	 */
	void push(std::shared_ptr<TaskV2> task) { this->s.insert(std::make_pair(task->next_execution_cycle, task->id)); }

	/**
	 * @brief Returns the task with the highest priority.
	 *
	 * @return A shared pointer to the highest priority task.
	 */
	std::shared_ptr<TaskV2> top() { return this->m[this->s.begin()->second]; }

	/**
	 * @brief Removes a specific task from the queue.
	 *
	 * @param task A shared pointer to the task to be removed.
	 */
	void remove(std::shared_ptr<TaskV2> task) {
		this->s.erase(this->s.find(std::make_pair(task->next_execution_cycle, task->id)));
		this->m.erase(task->id);
	}

	/**
	 * @brief Update a specific task's execution cycle in the queue.
	 *
	 * @param task A shared pointer to the task to be updated.
	 * @param tick The new execution cycle for the task.
	 */
	void update(std::shared_ptr<TaskV2> task, Tick tick) {
		this->s.erase(this->s.find(std::make_pair(task->next_execution_cycle, task->id)));
		task->next_execution_cycle = tick;
		this->push(task);
	}

	/// @brief Removes the highest priority task from the queue.
	void pop() { this->s.erase(this->s.begin()); }

	/**
	 * @brief Checks if the queue is empty.
	 *
	 * @return bool : True if the queue is empty, false otherwise.
	 */
	bool empty() { return this->s.empty(); }

	/**
	 * @brief Returns the number of tasks in the queue.
	 *
	 * @return size_t : The number of tasks in the queue.
	 */
	size_t size() { return this->s.size(); }

private:
	std::set<std::pair<Tick, int>>                   s;  // A set to store tasks sorted by execution cycle and ID.
	std::unordered_map<int, std::shared_ptr<TaskV2>> m;  // An unordered_map to store tasks and maintain order by ID.
};

}  // end of namespace acalsim
