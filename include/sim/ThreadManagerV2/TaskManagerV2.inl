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

#include "profiling/Synchronization.hh"
#include "sim/SimTop.hh"
#include "sim/ThreadManagerV2/ThreadManagerV2.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Utils.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

template <typename T>
void TaskManagerV2<T>::terminateThread() {
	// increment the number of finished threads
	this->nFinishedThreads++;
	MT_DEBUG_CLASS_INFO << "thread " << std::this_thread::get_id()
	                    << " is terminated. nFinishedThreads=" + std::to_string(this->nFinishedThreads);
}

template <typename T>
void TaskManagerV2<T>::init() {
	this->TaskManager::init();

	// Initialize task queue. One task per simulator.
	for (auto& sim : this->getSimulators()) {
		std::shared_ptr<Task> pTask = std::make_shared<TaskV2>(sim, sim->getName());
		// add new task to the task queue
		// Wake up threads that waiting for new task insertion
		this->addTask(pTask);
		sim->setTask(pTask);
	}

	this->setNTasks(this->getSimulators().size());
}

template <typename T>
void TaskManagerV2<T>::addTask(const std::shared_ptr<Task>& task) {
	// ThreadManager set up the task queue in ThreadManager::startSimThreads()
	std::lock_guard<std::mutex> lock(this->taskQueueMutex);
	// Add a new task with its initial execution cycle
	std::static_pointer_cast<TaskV2>(task)->id = this->taskQueue.size();
	this->taskQueue.add(std::static_pointer_cast<TaskV2>(task));
}

template <typename T>
std::shared_ptr<Task> TaskManagerV2<T>::getReadyTask() {
	ProfiledLock<"TaskManagerV2-TaskQueue-Phase1", std::lock_guard<std::mutex>, ProfileMode::ACALSIM_STATISTICS_FLAG>
	    lock(this->taskQueueMutex);

	std::shared_ptr<TaskV2> task{nullptr};

	MEASURE_TIME_MICROSECONDS(/* var_name */ task_queue, /* code_block */ {
		if (!this->taskQueue.empty()) {
			auto top_task = this->taskQueue.top();
			if (top->isReadyToTerminate() || top_task->next_execution_cycle <= top->getGlobalTick()) {
				this->taskQueue.pop();
				task = top_task;
			}
		}
	});

#ifdef ACALSIM_STATISTICS
	this->getThreadManager()->tqManipTimeStatistics.push(task_queue_lat);
#endif  // ACALSIM_STATISTICS

	return task;
}

template <typename T>
void TaskManagerV2<T>::scheduler(const size_t _tidx) {
	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));

	// flag will be set in the last iteration
	bool readyToTerminate = false;

	// Wait until the thread manager enters the running state
	while (!this->getThreadManager()->isRunning()) { ; }

	// Set the thread status from the inActive state to the Ready state
	this->setWorkerStatus(tid, ThreadStatus::Ready);

	while (this->getWorkerStatus(tid) != ThreadStatus::Terminated) {
		readyToTerminate = top->isReadyToTerminate();
		this->execPhase1(readyToTerminate, _tidx);
		this->execPhase2(readyToTerminate, _tidx);
	}
}

template <typename T>
void TaskManagerV2<T>::promoteTaskToTop(std::shared_ptr<acalsim::TaskV2> pTask) {
	std::lock_guard<std::mutex> lock(this->taskQueueMutex);

	if (pTask->next_execution_cycle != top->getGlobalTick() + 1) {
		this->taskQueue.update(pTask, top->getGlobalTick() + 1);
	}

	auto top_task = this->taskQueue.top();
	VERBOSE_CLASS_INFO << top_task->functor.getSimBaseName() +
	                          " next_execution_cycle : " + std::to_string(top_task->next_execution_cycle);
}

template <typename T>
void TaskManagerV2<T>::execPhase1(bool readyToTerminate, uint64_t tidx) {
	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));

	this->setWorkerStatus(tid, ThreadStatus::Ready);

#ifdef ACALSIM_STATISTICS
	auto   task_time_statistics_entry = this->getThreadManager()->taskExecTimeStatistics.getEntry(tidx);
	double task_time_curr_iter        = 0;
#endif  // ACALSIM_STATISTICS

	// continue to schedule task
	while (this->getWorkerStatus(tid) != ThreadStatus::Sleep &&
	       this->getWorkerStatus(tid) != ThreadStatus::Terminated) {
		MT_DEBUG_CLASS_INFO << "thread " << std::this_thread::get_id()
		                    << " enter scheduler(), workerStatus = " +
		                           ThreadManagerBase::ThreadStatusString[this->getWorkerStatus(tid)];

		if (auto task = std::static_pointer_cast<TaskV2>(this->getReadyTask())) [[likely]] {
			// has Task Ready to execute

			double task_lat = this->collectTaskExecStatistics(*task, task->getSimBaseName());

#ifdef ACALSIM_STATISTICS
			task_time_curr_iter += task_lat;
#endif  // ACALSIM_STATISTICS

			MT_DEBUG_CLASS_INFO << "thread " << std::this_thread::get_id()
			                    << " path #1 [" + task->functor.getSimBaseName() +
			                           "] steps, next_execution_cycle=" + std::to_string(task->next_execution_cycle) +
			                           " taskQueue.size()=" + std::to_string(taskQueue.size()) +
			                           " readyToTerminate=" + std::to_string(readyToTerminate);

			if (!readyToTerminate) {
				ProfiledLock<"TaskManagerV2-TaskQueue-Phase1", std::lock_guard<std::mutex>,
				             ProfileMode::ACALSIM_STATISTICS_FLAG>
				    lock(this->taskQueueMutex);

				MEASURE_TIME_MICROSECONDS(/* var_name */ task_queue, /* code_block */ { this->taskQueue.push(task); });

#ifdef ACALSIM_STATISTICS
				this->getThreadManager()->tqManipTimeStatistics.push(task_queue_lat);
#endif  // ACALSIM_STATISTICS
			}
		} else {
			if (readyToTerminate) {
				MT_DEBUG_CLASS_INFO << "thread " << std::this_thread::get_id() << " path #2 break the scheduler loop";
				this->terminateThread();
				this->setWorkerStatus(tid, ThreadStatus::Terminated);
			} else {
				MT_DEBUG_CLASS_INFO << "thread " << std::this_thread::get_id()
				                    << " path #3 sleep for the rest of the iteration";
				this->setWorkerStatus(tid, ThreadStatus::Sleep);
			}
		}
	}

#ifdef ACALSIM_STATISTICS
	this->getThreadManager()->phase1IdleTimer.enterSyncPoint(tidx);
	task_time_statistics_entry->push(task_time_curr_iter);
#endif  // ACALSIM_STATISTICS

	this->getThreadManager()->getWorkerThreadsDoneBarrier()->arrive_and_wait();
}

template <typename T>
void TaskManagerV2<T>::execPhase2(bool readyToTerminate, uint64_t tidx) {
	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));

	if (this->getWorkerStatus(tid) != ThreadStatus::Terminated) {
		// Do someting
		this->getThreadManager()->getWorkerThreadsDoneBarrier()->arrive_and_wait();
	}
}

}  // namespace acalsim
