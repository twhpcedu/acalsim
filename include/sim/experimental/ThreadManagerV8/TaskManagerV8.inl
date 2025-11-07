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
#include <barrier>

#include "profiling/Utils.hh"
#include "sim/ThreadManager.hh"
#include "sim/experimental/ThreadManagerV8/TaskManagerV8.hh"
#include "utils/Logging.hh"

namespace acalsim {

/**********************************
 *                                *
 *    TaskManagerV8::TaskList     *
 *                                *
 **********************************/

template <typename TFriend>
void TaskManagerV8<TFriend>::TaskList::preProcessRoutine() {
	this->dispatchedIdx.store(0, std::memory_order_release);
}

template <typename TFriend>
std::function<void(void)> TaskManagerV8<TFriend>::TaskList::acquire() {
	if (size_t task_idx = this->dispatchedIdx.fetch_add(1, std::memory_order_relaxed); task_idx < this->tasks.size()) {
		return this->tasks[task_idx];
	} else {
		return nullptr;
	}
}

template <typename TFriend>
bool TaskManagerV8<TFriend>::TaskList::empty() const {
	return this->tasks.empty();
}

template <typename TFriend>
size_t TaskManagerV8<TFriend>::TaskList::size() const {
	return this->tasks.size();
}

template <typename TFriend>
bool TaskManagerV8<TFriend>::TaskList::hasNoAvailTasks() const {
	return this->dispatchedIdx.load(std::memory_order_acquire) >= this->tasks.size();
}

template <typename TFriend>
void TaskManagerV8<TFriend>::TaskList::insert(std::function<void(void)> _task) {
	this->tasks.emplace_back(_task);
}

template <typename TFriend>
void TaskManagerV8<TFriend>::TaskList::clear() {
	this->tasks.clear();
}

/**********************************
 *                                *
 *         TaskManagerV8          *
 *                                *
 **********************************/

template <typename TFriend>
TaskManagerV8<TFriend>::~TaskManagerV8() {
	if (this->producerBarrierPtr) delete this->producerBarrierPtr;
	if (this->consumerBarrierPtr) delete this->consumerBarrierPtr;
}

template <typename TFriend>
void TaskManagerV8<TFriend>::init() {
	this->TaskManager::init();
	this->consumerBarrierPtr =
	    new std::barrier<void (*)(void)>(this->getThreadManager()->getNumThreads() + 1, []() { ; });
	this->consumerBarrierAvail = true;
	this->producerBarrierPtr   = new std::barrier<void (*)(void)>(2, []() { ; });
}

template <typename TFriend>
void TaskManagerV8<TFriend>::terminateThread() {
	this->nFinishedThreads++;
}

template <typename TFriend>
Tick TaskManagerV8<TFriend>::getNextSimTick() {
	return this->simulatorNextTickQueue.getTopPriority();
}

template <typename TFriend>
std::shared_ptr<Task> TaskManagerV8<TFriend>::getReadyTask() {
	std::shared_ptr<TaskV3> task_info{nullptr};

	MEASURE_TIME_MICROSECONDS(
	    /* var_name */ task_queue,
	    /* code_block*/ {
		    auto task_func = this->currTaskList->acquire();
		    if (task_func) { task_info = std::make_shared<TaskV3>(this->currTaskList, task_func); }
	    });

#ifdef ACALSIM_STATISTICS
	this->getThreadManager()->tqRetrievalTimeStatistics.push(task_queue_lat);
#endif  // ACALSIM_STATISTICS

	return task_info;
}

template <typename TFriend>
bool TaskManagerV8<TFriend>::noReadyTask() const {
	return !this->currTaskList || this->currTaskList->hasNoAvailTasks();
}

template <typename TFriend>
void TaskManagerV8<TFriend>::scheduler(const size_t _tidx) {
	const std::string thread_idx_str = std::to_string(_tidx);

#ifdef ACALSIM_STATISTICS
	auto   task_time_statistics_entry = this->getThreadManager()->taskExecTimeStatistics.getEntry(_tidx);
	double task_time_curr_iter        = 0;
#endif  // ACALSIM_STATISTICS

	while (!this->consumerBarrierAvail) { ; }

	this->consumerBarrierPtr->arrive_and_wait();

	do {
		typename TaskV3::SharedPtr task_pair = std::static_pointer_cast<TaskV3>(this->getReadyTask());

		if (task_pair) {  // Obtain a task
			MT_DEBUG_CLASS_INFO << "Thread " + thread_idx_str + " starts to execute a task.";

			double task_time = this->collectTaskExecStatistics(task_pair->task, "Anonymous");

#ifdef ACALSIM_STATISTICS
			task_time_curr_iter += task_time;
#endif  // ACALSIM_STATISTICS

		} else {  // There is no available task
#ifdef ACALSIM_STATISTICS
			this->getThreadManager()->phase1IdleTimer.enterSyncPoint(_tidx);
			task_time_statistics_entry->push(task_time_curr_iter);
			task_time_curr_iter = 0;
#endif  // ACALSIM_STATISTICS

			MT_DEBUG_CLASS_INFO << "Thread " + thread_idx_str + " enters TaskManagerV8<TFriend>::runnerCv for waiting.";

			this->finish();
			this->consumerBarrierPtr->arrive_and_wait();
		}
	} while (this->threadManager->isRunning());

	MT_DEBUG_CLASS_INFO << "Thread " + thread_idx_str + " terminates.";
}

template <typename TFriend>
void TaskManagerV8<TFriend>::submitTasks(typename TaskList::SharedPtr _tasks) {
	LABELED_ASSERT(!_tasks->empty() && !this->currTaskList, "TaskManagerV8");

	_tasks->preProcessRoutine();
	this->finishedCnt  = 0;
	this->currTaskList = _tasks;

	(void)this->consumerBarrierPtr->arrive();
}

template <typename TFriend>
void TaskManagerV8<TFriend>::finish() {
	if ((this->finishedCnt.fetch_add(1, std::memory_order_relaxed) + 1) == this->getThreadManager()->getNumThreads()) {
		this->currTaskList = nullptr;
		(void)this->producerBarrierPtr->arrive();
	}
}

}  // namespace acalsim
