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

// ACALSim
#include <atomic>
#include <condition_variable>
#include <memory>  // for std::unique_ptr
#include <mutex>
#include <queue>
#include <vector>

#include "profiling/Synchronization.hh"
#include "profiling/Utils.hh"
#include "sim/SimTop.hh"
#include "sim/TaskManager.hh"
#include "sim/ThreadManager.hh"
#include "sim/ThreadManagerV6/TaskV6.hh"
#include "sim/utils/ConcurrentTaskQueue.hh"

namespace acalsim {

// class definition forwarding
template <typename T>
class ThreadManagerV6;

template <typename TFriend>
class TaskManagerV6 : public TaskManager {
	friend class ThreadManagerV6<TFriend>;

public:
	// Thread-local task queue for partitioning tasks per thread
	struct ThreadLocalQueue {
		std::queue<TaskV6> tasks;  // Queue for local tasks
		std::mutex         mutex;  // Mutex for thread synchronization
	};

	TaskManagerV6(std::string _name)
	    : TaskManager(_name),
	      allThreadsDone(false),
	      startPhase1(false),
	      nWakeupThreads(0),
	      nFinishedThreads(0),
	      pendingInboundRequests(false),
	      tasksPartitioned(false) {}

	virtual ~TaskManagerV6() {}

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
	 * @brief Add a Task object to the TaskManagerV6.
	 * @param task The shared pointer to the Task object to be added.
	 */
	void addTask(const std::shared_ptr<Task>& task) override {
		auto tid     = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
		auto task_v6 = std::static_pointer_cast<TaskV6>(task);

		// Add a new task with its initial execution cycle - locking handled internally
		this->taskQueue.push(*task_v6, task_v6->next_execution_cycle);
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
	 * @brief Partition the tasks among thread-local queues at the start of a phase
	 */
	void partitionTasks() {
		auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
		MT_DEBUG_CLASS_INFO << "thread " << tid << " partitioning tasks";

		// Clear all thread-local queues
		for (size_t i = 0; i < threadLocalQueues.size(); i++) {
			std::unique_lock<std::mutex> lock(threadLocalQueues[i]->mutex);
			while (!threadLocalQueues[i]->tasks.empty()) { threadLocalQueues[i]->tasks.pop(); }
		}

		// Extract ready tasks from the global queue and distribute them
		Tick currentTick = top->getGlobalTick();
		int  numThreads  = this->getThreadManager()->getNumThreads();
		int  threadIdx   = 0;

		TaskV6 task;
		while (this->taskQueue.tryGetReadyTask(currentTick, task)) {
			// Round-robin distribution
			std::unique_lock<std::mutex> lock(threadLocalQueues[threadIdx]->mutex);
			threadLocalQueues[threadIdx]->tasks.push(task);
			threadIdx = (threadIdx + 1) % numThreads;
		}

		// Signal that task partitioning is complete
		tasksPartitioned.store(true);
		taskPartitionCv.notify_all();

		MT_DEBUG_CLASS_INFO << "thread " << tid << " finished partitioning tasks";
	}

	/**
	 * @brief Consolidate all completed tasks back to the global queue at the end of a phase
	 */
	void consolidateTasks() {
		auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
		MT_DEBUG_CLASS_INFO << "thread " << tid << " consolidating tasks";

		std::lock_guard<std::mutex> lock(completedTasksMutex);
		for (auto& task : completedTasks) { this->taskQueue.push(task, task.next_execution_cycle); }
		completedTasks.clear();

		MT_DEBUG_CLASS_INFO << "thread " << tid << " completed consolidating tasks";
	}

	/**
	 * @brief task scheduler for each worker thread
	 */
	void scheduler(const size_t _tidx) override {
		auto   tid              = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
		bool   readyToTerminate = false;  // flag will be set in the last iteration
		TaskV6 task;                      // For holding extracted tasks
		size_t threadIdx = _tidx;         // Thread ID for accessing local queue

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
				// Handle termination - first check thread-local queue
				bool foundTask = false;
				{
					ProfiledLock<"TaskManagerV6-LocalTaskQueue-Phase1", std::lock_guard<std::mutex>,
					             ProfileMode::ACALSIM_STATISTICS_FLAG>
					    lock(threadLocalQueues[threadIdx]->mutex);

					MEASURE_TIME_MICROSECONDS(/* var_name */ local_tq, /* code_block */ {
						if (!threadLocalQueues[threadIdx]->tasks.empty()) {
							task = threadLocalQueues[threadIdx]->tasks.front();
							threadLocalQueues[threadIdx]->tasks.pop();
							foundTask = true;
						}
					});

#ifdef ACALSIM_STATISTICS
					this->getThreadManager()->localTqManipTimeStatistics.push(local_tq_lat);
#endif  // ACALSIM_STATISTICS
				}

				if (foundTask) {
					// Execute remaining task without pushing it back
					double task_lat = this->collectTaskExecStatistics(task, task.getSimBaseName());

#ifdef ACALSIM_STATISTICS
					has_task = true;
					task_time_curr_iter += task_lat;
#endif  // ACALSIM_STATISTICS

					continue;
				}

				// If no local tasks, try global queue as fallback
				if (this->taskQueue.tryGetAnyTask</* EnableProfiling */ true, /* PhaseName */ "Phase1">(task)) {
					double task_lat = this->collectTaskExecStatistics(task, task.getSimBaseName());

#ifdef ACALSIM_STATISTICS
					has_task = true;
					task_time_curr_iter += task_lat;
#endif  // ACALSIM_STATISTICS

					continue;
				}

				// No more tasks to process
				MT_DEBUG_CLASS_INFO << "thread " << tid << " No more tasks to process, terminating";
				this->terminateThread();
				return;
			}

			// Wait for task partitioning if needed
			if (!tasksPartitioned.load()) {
				MEASURE_TIME_MICROSECONDS(/*var_name*/ tq_wait, /*code_block*/ {
					std::unique_lock<std::mutex> lock(taskPartitionMutex);
					taskPartitionCv.wait(lock, [this] { return tasksPartitioned.load(); });
				});

#ifdef ACALSIM_STATISTICS
				this->getThreadManager()->localTqPrepareTimeStatistics.push(tq_wait_lat);
#endif  // ACALSIM_STATISTICS
			}

			// Normal operation - try to get a task from thread-local queue
			bool taskProcessed = false;
			{
				ProfiledLock<"TaskManagerV6-LocalTaskQueue-Phase1", std::lock_guard<std::mutex>,
				             ProfileMode::ACALSIM_STATISTICS_FLAG>
				    lock(threadLocalQueues[threadIdx]->mutex);
				MEASURE_TIME_MICROSECONDS(/* var_name */ local_tq, /* code_block */ {
					if (!threadLocalQueues[threadIdx]->tasks.empty()) {
						task = threadLocalQueues[threadIdx]->tasks.front();
						threadLocalQueues[threadIdx]->tasks.pop();
						taskProcessed = true;
					}
				});

#ifdef ACALSIM_STATISTICS
				this->getThreadManager()->localTqManipTimeStatistics.push(local_tq_lat);
#endif  // ACALSIM_STATISTICS
			}

			if (taskProcessed) {
				// Execute the task
				double task_lat = this->collectTaskExecStatistics(task, task.getSimBaseName());

#ifdef ACALSIM_STATISTICS
				has_task = true;
				task_time_curr_iter += task_lat;
#endif  // ACALSIM_STATISTICS

				// Save the task for later consolidation
				{
					ProfiledLock<"TaskManagerV6-LocalTaskQueue-Phase1", std::lock_guard<std::mutex>,
					             ProfileMode::ACALSIM_STATISTICS_FLAG>
					    lock(completedTasksMutex);

					MEASURE_TIME_MICROSECONDS(/* var_name */ complete_tq,
					                          /* code_block */ { completedTasks.push_back(task); });

#ifdef ACALSIM_STATISTICS
					this->getThreadManager()->localTqManipTimeStatistics.push(complete_tq_lat);
#endif  // ACALSIM_STATISTICS
				}
			} else {
				// Try to steal tasks from other threads
				bool stolenTask = false;
				for (size_t i = 0; i < threadLocalQueues.size() && !stolenTask; i++) {
					if (i == threadIdx) continue;  // Skip own queue

					ProfiledLock<"TaskManagerV6-LocalTaskQueue-Phase1", std::lock_guard<std::mutex>,
					             ProfileMode::ACALSIM_STATISTICS_FLAG>
					    lock(threadLocalQueues[i]->mutex);

					MEASURE_TIME_MICROSECONDS(/* var_name */ tq,
					                          /* code_block */ {
						                          if (!threadLocalQueues[i]->tasks.empty()) {
							                          task = threadLocalQueues[i]->tasks.front();
							                          threadLocalQueues[i]->tasks.pop();
							                          stolenTask = true;

							                          MT_DEBUG_CLASS_INFO << "thread " << tid
							                                              << " stole task from thread " << i;
						                          }
					                          });

#ifdef ACALSIM_STATISTICS
					this->getThreadManager()->localTqManipTimeStatistics.push(tq_lat);
#endif  // ACALSIM_STATISTICS
				}

				if (stolenTask) {
					// Execute stolen task
					double task_lat = this->collectTaskExecStatistics(task, task.getSimBaseName());

#ifdef ACALSIM_STATISTICS
					has_task = true;
					task_time_curr_iter += task_lat;
#endif  // ACALSIM_STATISTICS

					// Save for later consolidation
					{
						ProfiledLock<"TaskManagerV6-LocalTaskQueue-Phase1", std::lock_guard<std::mutex>,
						             ProfileMode::ACALSIM_STATISTICS_FLAG>
						    lock(completedTasksMutex);

						MEASURE_TIME_MICROSECONDS(/* var_name */ complete_tq,
						                          /* code_block */ { completedTasks.push_back(task); });

#ifdef ACALSIM_STATISTICS
						this->getThreadManager()->localTqManipTimeStatistics.push(complete_tq_lat);
#endif  // ACALSIM_STATISTICS
					}
				} else {
					// No tasks available, go to sleep
					gotoSleep = true;
				}
			}

			MT_DEBUG_CLASS_INFO << "thread " << tid << " finish one iteration gotoSleep=" + std::to_string(gotoSleep);

			if (gotoSleep) {
				ProfiledLock<"TaskManagerV6-TaskAvailCv-Phase1", std::unique_lock<std::mutex>,
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
			std::shared_ptr<Task> pTask = std::make_shared<TaskV6>(sim, sim->getName());
			// add new task to the task queue
			this->addTask(pTask);
			sim->setTask(pTask);
		}

		// Initialize thread-local queues with unique_ptr
		int numThreads = this->threadManager->getNumThreads();
		threadLocalQueues.clear();

		for (int i = 0; i < numThreads; i++) { threadLocalQueues.push_back(std::make_unique<ThreadLocalQueue>()); }

		this->setNTasks(this->getSimulators().size());
		this->nWakeupThreads = this->threadManager->getNumThreads();
	}

	ThreadManagerV6<TFriend>* getThreadManager() const override {
		return dynamic_cast<ThreadManagerV6<TFriend>*>(this->threadManager);
	}

	void setPendingInboundRequests() { this->pendingInboundRequests = true; }
	void clearPendingInboundRequests() { this->pendingInboundRequests = false; }

protected:
	/**
	 * @brief task priority queue sorted by the next execution time
	 * Using concurrent queue implementation with fine-grained locking
	 */
	ConcurrentUpdateablePriorityQueue<TaskV6> taskQueue;

	/**
	 * @brief Thread-local task queues for each worker thread
	 * Using unique_ptr to avoid copy/move issues with mutex
	 */
	std::vector<std::unique_ptr<ThreadLocalQueue>> threadLocalQueues;

	/**
	 * @brief Completed tasks that will be consolidated back to global queue
	 */
	std::vector<TaskV6> completedTasks;
	std::mutex          completedTasksMutex;

	/**
	 * @brief Task partitioning synchronization
	 */
	std::mutex              taskPartitionMutex;
	std::condition_variable taskPartitionCv;
	std::atomic<bool>       tasksPartitioned;

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
	ThreadManagerV6<TFriend>* getThreadManager() { return static_cast<ThreadManagerV6<TFriend>*>(this->threadManager); }
};

}  // end of namespace acalsim
