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

// ACALSim
#include <condition_variable>

#include "sim/SimTop.hh"
#include "sim/TaskManager.hh"
#include "sim/ThreadManager.hh"
#include "sim/experimental/ThreadManagerV5/TaskV5.hh"
#include "sim/utils/ConcurrentTaskQueue.hh"

namespace acalsim {

// class definition forwarding
template <typename T>
class ThreadManagerV5;

template <typename TFriend>
class TaskManagerV5 : public TaskManager {
	friend class ThreadManagerV5<TFriend>;

public:
	TaskManagerV5(std::string _name)
	    : TaskManager(_name),
	      allThreadsDone(false),
	      startPhase1(false),
	      nWakeupThreads(0),
	      nFinishedThreads(0),
	      pendingInboundRequests(false) {}

	virtual ~TaskManagerV5() {}

	void promoteTaskToTop(int simID) {
		auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));

		// No explicit lock needed - handled inside the queue methods
		MT_DEBUG_CLASS_INFO << this->taskQueue.dump();
		this->taskQueue.update(simID, top->getGlobalTick() + 1);
		MT_DEBUG_CLASS_INFO << "thread " << tid << " update task next_execution_cycle to "
		                    << this->taskQueue.top().next_execution_cycle << " simID: " << simID;
		MT_DEBUG_CLASS_INFO << this->taskQueue.dump();
	}

	/**
	 * @brief Terminate a thread and Increment the number of finished threads
	 */
	void terminateThread() override {
		auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
		{
			std::unique_lock<std::mutex> lock(this->taskAvailableMutex);
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
		MT_DEBUG_CLASS_INFO << "thread " << tid << " exiting terminateThread function";
	}

	/**
	 * @brief Add a Task object to the TaskManagerV5.
	 * @param task The shared pointer to the Task object to be added.
	 */
	void addTask(const std::shared_ptr<Task>& task) override {
		auto tid     = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
		auto task_v5 = std::static_pointer_cast<TaskV5>(task);

		// Add a new task with its initial execution cycle - locking handled internally
		this->taskQueue.push(*task_v5, task_v5->next_execution_cycle);
		MT_DEBUG_CLASS_INFO << "thread " << tid << " Add a test to taskQueue " + this->taskQueue.dump();
	}

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
	void scheduler(const size_t _tidx) override {
		auto   tid              = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
		bool   readyToTerminate = false;  // flag will be set in the last iteration
		TaskV5 task;                      // For holding extracted tasks

#ifdef ACALSIM_STATISTICS
		bool   has_task                   = false;
		double task_time_curr_iter        = 0;
		auto   task_time_statistics_entry = this->getThreadManager()->taskExecTimeStatistics.getEntry(_tidx);
#endif  // ACALSIM_STATISTICS

		while (!this->getThreadManager()->isRunning()) { ; }

		// Set the thread status from the inActive state to the Ready state
		this->setWorkerStatus(tid, ThreadStatus::Ready);

		// continue to schedule task
		while (this->getWorkerStatus(tid) != ThreadStatus::Terminated) {
			bool gotoSleep = false;

			MT_DEBUG_CLASS_INFO << "thread " << tid
			                    << " start scheduling, workerStatus = " +
			                           ThreadManager::ThreadStatusString[this->getWorkerStatus(tid)] +
			                           " nWakeupThreads="
			                    << this->nWakeupThreads << " nFinishedThreads=" << this->nFinishedThreads;

			// Check termination status at the beginning of each loop
			readyToTerminate = top->isReadyToTerminate();

			// Handle termination case first
			if (readyToTerminate) {
				if (this->taskQueue.empty()) {
					// No more tasks and ready to terminate
					MT_DEBUG_CLASS_INFO << "thread " << tid << " path #1 break the scheduler loop and terminate";
					this->terminateThread();
					return;
				}

				// Even if tasks remain, we should terminate them in termination mode
				// Try to get a task (any task, ready or not)
				if (this->taskQueue.tryGetAnyTask</* EnableProfiling */ true, /* PhaseName */ "Phase1">(task)) {
					// Execute any remaining task without pushing it back
					double task_lat = this->collectTaskExecStatistics(task, task.getSimBaseName());

#ifdef ACALSIM_STATISTICS
					has_task = true;
					task_time_curr_iter += task_lat;
#endif  // ACALSIM_STATISTICS

					MT_DEBUG_CLASS_INFO << "thread " << tid << " In termination mode processing task " << task.id
					                    << " without pushing back";

					// Continue with next iteration - don't push task back
					continue;
				} else {
					// No tasks left to process
					MT_DEBUG_CLASS_INFO << "thread " << tid << " No more tasks to process, terminating";
					this->terminateThread();
					return;
				}
			}

			// Normal operation (not terminating)
			if (this->taskQueue.tryGetReadyTask</* EnableProfiling */ true, /* PhaseName */ "Phase1">(
			        top->getGlobalTick(), task)) {
				// Got a ready task to execute

				// Execute task
				double task_lat = this->collectTaskExecStatistics(task, task.getSimBaseName());

#ifdef ACALSIM_STATISTICS
				has_task = true;
				task_time_curr_iter += task_lat;
#endif  // ACALSIM_STATISTICS

				// Push the task back with its next execution cycle
				this->taskQueue.push</* EnableProfiling */ true, /* PhaseName */ "Phase1">(task,
				                                                                           task.next_execution_cycle);

				MT_DEBUG_CLASS_INFO << "thread " << tid << "nFinishedThreads = " << this->nFinishedThreads
				                    << " After push task back to taskQueue " + this->taskQueue.dump();

				// Check if we should sleep - no more ready tasks
				if (!this->taskQueue.hasReadyTask</* EnableProfiling */ true, /* PhaseName */ "Phase1">(
				        top->getGlobalTick())) {
					gotoSleep = true;
				}
			} else {
				// No ready task, go to sleep
				gotoSleep = true;
			}

			MT_DEBUG_CLASS_INFO << "thread " << tid << " finish one iteration gotoSleep=" + std::to_string(gotoSleep);

			if (gotoSleep) {
				ProfiledLock<"TaskManagerV5-TaskAvailCv-Phase1", std::unique_lock<std::mutex>,
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

					// reset number of wakeup threads
					this->nWakeupThreads = 0;

					// Set the flag to true if not already set
					this->allThreadsDone.store(true);

					// Signal that all threads are done
					this->workerThreadsDoneCondVar.notify_one();
				}

#ifdef ACALSIM_STATISTICS
				if (has_task) {
					this->getThreadManager()->phase1IdleTimer.enterSyncPoint(_tidx);
					task_time_statistics_entry->push(task_time_curr_iter);
					task_time_curr_iter = 0;
					has_task            = false;
				}
#endif  // ACALSIM_STATISTICS

				// Add timeout to avoid indefinite waiting
				auto status = this->newTaskAvailableCondVar.wait_for(lock, std::chrono::milliseconds(100), [this] {
					return this->startPhase1.load() && !this->allThreadsDone.load();
				});

				// If timeout occurred, check termination condition again
				if (!status) {
					if (top->isReadyToTerminate()) {
						MT_DEBUG_CLASS_INFO << "thread " << tid << " timed out waiting and termination is ready";
						// Don't terminate here, let the next loop iteration handle it properly
					}
				}

				// Thread Wakeup
				this->nWakeupThreads++;
				MT_DEBUG_CLASS_INFO << "thread " << tid
				                    << " startPhase1=" + std::to_string(this->startPhase1.load()) +
				                           " wake up nWakeupThreads="
				                    << this->nWakeupThreads;

				// Re-check termination status after waking up
				readyToTerminate = top->isReadyToTerminate();
			}
		}  // end of while loop
	}

	/**
	 * @brief Initialize the task manager after worker threads are launched.
	 */
	void init() override {
		this->TaskManager::init();

		// Initialize task queue. One task per simulator.
		for (auto& sim : this->getSimulators()) {
			std::shared_ptr<Task> pTask = std::make_shared<TaskV5>(sim, sim->getName());
			// add new task to the task queue
			this->addTask(pTask);
			sim->setTask(pTask);
		}
		this->setNTasks(this->getSimulators().size());
		this->nWakeupThreads = this->threadManager->getNumThreads();
	}

	ThreadManagerV5<TFriend>* getThreadManager() const override {
		return dynamic_cast<ThreadManagerV5<TFriend>*>(this->threadManager);
	}

	void setPendingInboundRequests() { this->pendingInboundRequests = true; }
	void clearPendingInboundRequests() { this->pendingInboundRequests = false; }

protected:
	/**
	 * @brief task priority queue sorted by the next execution time
	 * Using concurrent queue implementation with fine-grained locking
	 */
	ConcurrentUpdateablePriorityQueue<TaskV5> taskQueue;

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

	/**
	 * @brief number of finished threads
	 */
	int nFinishedThreads;

	/**
	 * @brief number of wakeup threads
	 */
	int nWakeupThreads;

	// flag for pending inbound requests in the framework
	bool pendingInboundRequests;

private:
	ThreadManagerV5<TFriend>* getThreadManager() { return static_cast<ThreadManagerV5<TFriend>*>(this->threadManager); }
};

}  // end of namespace acalsim
