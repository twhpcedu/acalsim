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

#include <mutex>

#include "profiling/Synchronization.hh"
#include "profiling/Utils.hh"
#include "sim/SimTop.hh"
#include "sim/ThreadManagerV1/ThreadManagerV1.hh"

namespace acalsim {

template <typename T>
void TaskManagerV1<T>::promoteTaskToTop(int id) {
	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
	{
		std::lock_guard<std::mutex> lock(this->taskQueueMutex);
		MT_DEBUG_CLASS_INFO << this->taskQueue.dump();

		this->taskQueue.update(id, top->getGlobalTick() + 1);

		MT_DEBUG_CLASS_INFO << "thread " << tid << " update task next_execution_cycle to "
		                    << this->taskQueue.top().next_execution_cycle << " simID: " << id;
		MT_DEBUG_CLASS_INFO << this->taskQueue.dump();
	}
}

template <typename T>
void TaskManagerV1<T>::terminateThread() {
	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));

	std::unique_lock<std::mutex> lock(this->taskQueueMutex);

	// increment the number of finished threads
	this->nFinishedThreads++;
	MT_DEBUG_CLASS_INFO << "thread " << tid << " is terminated. nFinishedThreads=" << this->nFinishedThreads;
	if (this->nFinishedThreads == this->getThreadManager()->getNumThreads()) {
		// Set the flag to true if not already set
		this->allThreadsDone.store(true);
		lock.unlock();
		// Signal that all threads are done
		this->workerThreadsDoneCondVar.notify_one();
	}
}

template <typename T>
void TaskManagerV1<T>::addTask(const std::shared_ptr<Task>& task) {
	auto tid     = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
	auto task_v1 = std::static_pointer_cast<TaskV1>(task);

	// ThreadManager set up the task queue in ThreadManager::startSimThreads()
	std::lock_guard<std::mutex> lock(this->taskQueueMutex);
	// Add a new task with its initial execution cycle
	this->taskQueue.push(*task_v1, task_v1->next_execution_cycle);

	MT_DEBUG_CLASS_INFO << "thread " << tid << " Add a test to taskQueue " + this->taskQueue.dump();
}

template <typename T>
void TaskManagerV1<T>::scheduler(const size_t _tidx) {
	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));

#ifdef ACALSIM_STATISTICS
	double task_time_curr_iter         = 0;
	double tq_operation_time_curr_iter = 0;

	std::chrono::time_point<std::chrono::high_resolution_clock> tq_op_start, tq_op_end;
#endif  // ACALSIM_STATISTICS

	// flag will be set in the last iteration
	bool readyToTerminate = false;

	// Wait until ThreadManager signals that simulation is running
	// This replaces the busy-wait spin loop to reduce context switches
	{
		std::unique_lock<std::mutex> lock(this->runningMutex);
		this->runningCondVar.wait(lock, [this] { return this->getThreadManager()->isRunning(); });
	}

	// Set the thread status from the inActive state to the Ready state
	this->setWorkerStatus(tid, ThreadStatus::Ready);

	// continue to schedule task
	while (this->getWorkerStatus(tid) != ThreadStatus::Terminated) {
		bool gotoSleep = false;

		// aquire the lock to access taskQueue
		ProfiledLock<"TaskManagerV1-TaskQueue-Phase1", std::unique_lock<std::mutex>,
		             ProfileMode::ACALSIM_STATISTICS_FLAG>
		    lock(this->taskQueueMutex);

		MT_DEBUG_CLASS_INFO << "thread " << tid
		                    << " start scheduling, workerStatus = " +
		                           ThreadManager::ThreadStatusString[this->getWorkerStatus(tid)] + " nWakeupThreads="
		                    << this->nWakeupThreads << " nFinishedThreads=" << this->nFinishedThreads;
#ifdef ACALSIM_STATISTICS
		tq_op_start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

		if (!this->taskQueue.hasReadyTask(top->getGlobalTick()) && !readyToTerminate) {
#ifdef ACALSIM_STATISTICS
			tq_op_end = std::chrono::high_resolution_clock::now();
			tq_operation_time_curr_iter += (double)(tq_op_end - tq_op_start).count() / pow(10, 3);
#endif  // ACALSIM_STATISTICS

			// No Job is available in the taskQueue
			gotoSleep = true;
		} else [[likely]] {  // has Task ready to execute or readyToTerminate
#ifdef ACALSIM_STATISTICS
			tq_op_end = std::chrono::high_resolution_clock::now();
			tq_operation_time_curr_iter += (double)(tq_op_end - tq_op_start).count() / pow(10, 3);
#endif  // ACALSIM_STATISTICS

			if (readyToTerminate && this->taskQueue.empty()) {
				lock.unlock();
				MT_DEBUG_CLASS_INFO << "thread " << tid << " path #1  break the scheduler loop and terminate";
				this->terminateThread();
				return;
			} else [[likely]] {
#ifdef ACALSIM_STATISTICS
				tq_op_start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

				// Detach the task from the queue and capture it by value (avoids copy)
				auto task = this->taskQueue.top();
				this->taskQueue.pop();

#ifdef ACALSIM_STATISTICS
				tq_op_end = std::chrono::high_resolution_clock::now();
				tq_operation_time_curr_iter += (double)(tq_op_end - tq_op_start).count() / pow(10, 3);
#endif  // ACALSIM_STATISTICS

				lock.unlock();  // Release the lock immediately

				// execute task
				double task_time = this->collectTaskExecStatistics(task, task.getSimBaseName());

#ifdef ACALSIM_STATISTICS
				task_time_curr_iter += task_time;
#endif  // ACALSIM_STATISTICS

				if (readyToTerminate) { continue; }

				lock.lock();

				MT_DEBUG_CLASS_INFO << "thread " << tid
				                    << " path #3 [" + task.functor.getSimBaseName() + "] steps, next_execution_cycle="
				                    << task.next_execution_cycle << " taskQueue.size()=" << this->taskQueue.size()
				                    << " readyToTerminate=" << readyToTerminate;

#ifdef ACALSIM_STATISTICS
				tq_op_start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

				this->taskQueue.push(task, task.next_execution_cycle);

#ifdef ACALSIM_STATISTICS
				tq_op_end = std::chrono::high_resolution_clock::now();
				tq_operation_time_curr_iter += (double)(tq_op_end - tq_op_start).count() / pow(10, 3);
#endif  // ACALSIM_STATISTICS

				MT_DEBUG_CLASS_INFO << "thread " << tid << "nFinishedThreads = " << this->nFinishedThreads
				                    << " After push task back to taskQueue " + this->taskQueue.dump();

#ifdef ACALSIM_STATISTICS
				tq_op_start = std::chrono::high_resolution_clock::now();
#endif  // ACALSIM_STATISTICS

				if (!this->taskQueue.hasReadyTask(top->getGlobalTick())) { gotoSleep = true; }

#ifdef ACALSIM_STATISTICS
				tq_op_end = std::chrono::high_resolution_clock::now();
				tq_operation_time_curr_iter += (double)(tq_op_end - tq_op_start).count() / pow(10, 3);
#endif  // ACALSIM_STATISTICS
			}

		}  // taskQueue is non-empty

		lock.unlock();

		MT_DEBUG_CLASS_INFO << "thread " << tid << " finish one iteration gotoSleep=" + std::to_string(gotoSleep);

		if (gotoSleep) {
			ProfiledLock<"TaskManagerV1-TaskAvailCv-Phase1", std::unique_lock<std::mutex>,
			             ProfileMode::ACALSIM_STATISTICS_FLAG>
			    lock(this->taskAvailableMutex);

			if (this->startPhase1.load()) {
				// When first thread that completes the current iteration goes to sleep,
				// stop waking up threads
				this->startPhase1.store(false);
				MT_DEBUG_CLASS_INFO << "thread " + std::to_string(tid) +
				                           " stop waking up any more threads. nWakeupThreads=" +
				                           std::to_string(this->nWakeupThreads);
			}

			// A thread should go to sleep if there is no available task
			MT_DEBUG_CLASS_INFO << "thread " << tid
			                    << " path #2 sleep for the rest of the iteration nFinishedThreads = "
			                    << this->nFinishedThreads
			                    << " startPhase1=" + std::to_string(this->startPhase1.load()) + " nWakeupThreads="
			                    << this->nWakeupThreads;

			// increment the number of finished threads
			this->nFinishedThreads++;

			if (this->nFinishedThreads == this->nWakeupThreads) {
				MT_DEBUG_CLASS_INFO << "thread " << tid << " phase1Done/allThreadsDone is set";
				// When last thread that completes the current iteration goes to sleep,
				// send the notification to the control thread.
				// Make sure the last thread completes the current iteration and switch phases
				// Setting allThreadsDone, control thread notification and enter wait state need to be atomic
				// for many-to-1 synchronization

				// reset number of wakeup theads
				this->nWakeupThreads = 0;

				// Set the flag to true if not already set
				this->allThreadsDone.store(true);

				// Signal that all threads are done
				this->workerThreadsDoneCondVar.notify_one();
				// std::cout << "TaskManager send notification\n";
			}

#ifdef ACALSIM_STATISTICS
			this->getThreadManager()->phase1IdleTimer.enterSyncPoint(_tidx);

			this->getThreadManager()->taskExecTimeStatistics.getEntry(_tidx)->push(task_time_curr_iter);
			this->getThreadManager()->tqOperationTimeStatistics.getEntry(_tidx)->push(tq_operation_time_curr_iter);

			task_time_curr_iter         = 0;
			tq_operation_time_curr_iter = 0;
#endif  // ACALSIM_STATISTICS

			// Wait for a new iteration to start
			this->newTaskAvailableCondVar.wait(lock, [this] {
				// std::cout << "TaskManager send wait and check\n";
				//  startPhase1 is true when none of threads going to sleep
				return this->startPhase1.load() && !this->allThreadsDone.load();
			});

			// Thread Wakeup
			this->nWakeupThreads++;
			MT_DEBUG_CLASS_INFO << "thread " << tid
			                    << " startPhase1=" + std::to_string(this->startPhase1.load()) +
			                           " wake up nWakeupThreads="
			                    << this->nWakeupThreads;
			lock.unlock();
			readyToTerminate = top->isReadyToTerminate();
		}

	}  // end of while(threadManager->workerStatus[tid] != ThreadStatus::Terminated) loop
}

template <typename T>
void TaskManagerV1<T>::init() {
	this->TaskManager::init();

	// Initialize task queue. One task per simulator.
	for (auto& sim : this->getSimulators()) {
		std::shared_ptr<Task> pTask = std::make_shared<TaskV1>(sim, sim->getName());
		// add new task to the task queue
		// Wake up threads that waiting for new task insertion
		this->addTask(pTask);
		sim->setTask(pTask);
	}
	this->setNTasks(this->getSimulators().size());
	this->nWakeupThreads = this->threadManager->getNumThreads();
}

}  // namespace acalsim
