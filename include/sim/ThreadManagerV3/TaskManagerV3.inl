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

#include <chrono>
#include <mutex>

#include "profiling/Synchronization.hh"
#include "profiling/Utils.hh"
#include "sim/SimTop.hh"
#include "sim/ThreadManager.hh"
#include "sim/ThreadManagerV3/TaskManagerV3.hh"

namespace acalsim {

/**********************************
 *                                *
 *    TaskManagerV3::TaskList     *
 *                                *
 **********************************/

template <typename TFriend>
void TaskManagerV3<TFriend>::TaskList::preProcessRoutine() {
	this->dispatchedIdx = 0;
	this->finishedCnt   = 0;
}

template <typename TFriend>
std::function<void(void)> TaskManagerV3<TFriend>::TaskList::acquire() {
	if (size_t task_idx = this->dispatchedIdx++; task_idx < this->tasks.size()) {
		return this->tasks[task_idx];
	} else {
		return nullptr;
	}
}

template <typename TFriend>
void TaskManagerV3<TFriend>::TaskList::finish() {
	// Enter the barrier if all tasks of the task list are finished
	if (++this->finishedCnt == this->tasks.size()) { std::ignore = this->barrierPtr->arrive(); }
}

template <typename TFriend>
bool TaskManagerV3<TFriend>::TaskList::empty() const {
	return this->tasks.empty();
}

template <typename TFriend>
size_t TaskManagerV3<TFriend>::TaskList::size() const {
	return this->tasks.size();
}

template <typename TFriend>
bool TaskManagerV3<TFriend>::TaskList::hasNoAvailTasks() const {
	return this->dispatchedIdx >= this->tasks.size();
}

template <typename TFriend>
void TaskManagerV3<TFriend>::TaskList::insert(std::function<void(void)> _task) {
	this->tasks.emplace_back(_task);
}

template <typename TFriend>
void TaskManagerV3<TFriend>::TaskList::clear() {
	this->tasks.clear();
}

template <typename TFriend>
void TaskManagerV3<TFriend>::TaskList::wait() {
	this->barrierPtr->arrive_and_wait();
}

/**********************************
 *                                *
 *         TaskManagerV3          *
 *                                *
 **********************************/

template <typename TFriend>
void TaskManagerV3<TFriend>::terminateThread() {
	this->nFinishedThreads++;
}

template <typename TFriend>
Tick TaskManagerV3<TFriend>::getNextSimTick() {
	return this->simulatorNextTickQueue.getTopPriority();
}

template <typename TFriend>
std::shared_ptr<Task> TaskManagerV3<TFriend>::getReadyTask() {
	ProfiledLock<"TaskManagerV3-TaskQueue-Phase1", std::lock_guard<std::mutex>, ProfileMode::ACALSIM_STATISTICS_FLAG>
	    lock(this->taskListQueueMutex);

	std::shared_ptr<TaskV3> task_info{nullptr};

	MEASURE_TIME_MICROSECONDS(
	    /* var_name */ task_queue,
	    /* code_block*/ {
		    if (!this->taskListQueue.empty()) {
			    // Acquire a task from the taskListQueue
			    typename TaskList::SharedPtr& task_list = this->taskListQueue.front();
			    auto                          task_func = task_list->acquire();

			    if (task_func) { task_info = std::make_shared<TaskV3>(task_list, task_func); }

			    // Remove the task list from the taskListQueue if it becomes empty
			    if (task_list->hasNoAvailTasks()) [[unlikely]] { this->taskListQueue.pop(); }
		    }
	    });

#ifdef ACALSIM_STATISTICS
	this->getThreadManager()->tqRetrievalTimeStatistics.push(task_queue_lat);
#endif  // ACALSIM_STATISTICS

	return task_info;
}

template <typename TFriend>
bool TaskManagerV3<TFriend>::noReadyTask() const {
	ProfiledLock<"TaskManagerV3-TaskQueue-Phase1", std::lock_guard<std::mutex>, ProfileMode::ACALSIM_STATISTICS_FLAG>
	    lock(this->taskListQueueMutex);
	return this->taskListQueue.empty();
}

template <typename TFriend>
void TaskManagerV3<TFriend>::scheduler(const size_t _tidx) {
	const std::string thread_idx_str = std::to_string(_tidx);
	std::mutex        runner_cv_mutex;

#ifdef ACALSIM_STATISTICS
	auto   task_time_statistics_entry = this->getThreadManager()->taskExecTimeStatistics.getEntry(_tidx);
	double task_time_curr_iter        = 0;
#endif  // ACALSIM_STATISTICS

	do {
		typename TaskV3::SharedPtr task_pair = std::static_pointer_cast<TaskV3>(this->getReadyTask());

		if (task_pair) {  // Obtain a task
			MT_DEBUG_CLASS_INFO << "Thread " + thread_idx_str + " starts to execute a task.";

			double task_time = this->collectTaskExecStatistics(task_pair->task, "Anonymous");

#ifdef ACALSIM_STATISTICS
			task_time_curr_iter += task_time;
#endif  // ACALSIM_STATISTICS

			task_pair->list->finish();
		} else {  // There is no available task
#ifdef ACALSIM_STATISTICS
			this->getThreadManager()->phase1IdleTimer.enterSyncPoint(_tidx);
			task_time_statistics_entry->push(task_time_curr_iter);
			task_time_curr_iter = 0;
#endif  // ACALSIM_STATISTICS

			MT_DEBUG_CLASS_INFO << "Thread " + thread_idx_str + " enters TaskManagerV3<TFriend>::runnerCv for waiting.";

			bool pred;

			ProfiledLock<"TaskManagerV3-TaskAvailCv-Phase1", std::unique_lock<std::mutex>,
			             ProfileMode::ACALSIM_STATISTICS_FLAG>
			    lock(runner_cv_mutex);

			do {
				pred = this->runnerCv.wait_for(lock, std::chrono::milliseconds(10), [this]() {
					return (this->threadManager->isRunning() & !this->noReadyTask()) |
					       (top->isReadyToTerminate() & !this->threadManager->isRunning());
				});
			} while (!pred);
		}
	} while (this->threadManager->isRunning());

	MT_DEBUG_CLASS_INFO << "Thread " + thread_idx_str + " terminates.";
}

template <typename TFriend>
void TaskManagerV3<TFriend>::submitTasks(typename TaskList::SharedPtr _tasks) {
	if (!_tasks->empty()) {
		_tasks->preProcessRoutine();

		this->taskListQueueMutex.lock();
		this->taskListQueue.push(_tasks);
		this->taskListQueueMutex.unlock();

		this->runnerCv.notify_all();
	}
}

}  // namespace acalsim
