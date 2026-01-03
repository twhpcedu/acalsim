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

#include "profiling/Utils.hh"
#include "sim/SimTop.hh"
#include "sim/ThreadManagerV6/TaskManagerV6.hh"
#include "sim/ThreadManagerV6/ThreadManagerV6.hh"

namespace acalsim {

template <typename T>
void ThreadManagerV6<T>::runInterIterationUpdate() {
	for (auto& sim : this->simulators) {
		if (sim->interIterationUpdate()) {
			// The simulator has pending inbound requests in the SimPort
			// The corresponding task might not be the top of the taskQueue
			// Need to promote the task to the top
			this->getTaskManager()->setPendingInboundRequests();
			this->getTaskManager()->promoteTaskToTop(sim->getID());
		}
	}
}

template <typename T>
void ThreadManagerV6<T>::terminateAllThreads() {
	// Allow child threads to terminate itself
	this->running = false;

	MT_DEBUG_CLASS_INFO << "thread " << std::this_thread::get_id()
	                    << " terminateAllThreads() nFinishedThreads=" << this->getTaskManager()->nFinishedThreads;
	for (size_t i = 0; i < this->workers.size(); i++) {
		MT_DEBUG_CLASS_INFO << "Thread " << i << " joinable: " << (this->workers[i]->joinable() ? "yes" : "no");
	}
	for (auto& worker : this->workers) {
		if (worker->joinable()) worker->join();
	}
}

template <typename T>
void ThreadManagerV6<T>::startPhase1() {
	// In Phase 1, all simulators will execute one iteration
	// The control thread (SimTop) executes the SimTop::control_thread_step() function
	// This function is for the control thread to do something in the beginner of Phase 1

	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));

	this->collectBeforePhase1Statistics();

	// Partition tasks at the start of Phase 1
	this->getTaskManager()->tasksPartitioned.store(false);
	MEASURE_TIME_MICROSECONDS(/*var_name*/ tq_partition, /*code_block*/ { this->getTaskManager()->partitionTasks(); });

	if (this->getTaskManager()->taskQueue.empty())
		MT_DEBUG_CLASS_INFO << "Control thread " << tid << " wake up all worker threads. taskQueue is empty";
	else
		MT_DEBUG_CLASS_INFO << "Control thread " << tid
		                    << " wake up all worker threads. taskQueue->top().next_execution_cycle="
		                    << this->getTaskManager()->taskQueue.top().next_execution_cycle;

	this->getTaskManager()->nFinishedThreads = 0;
	this->getTaskManager()->startPhase1.store(true);
	// Reset allThreadsDone before starting a new iteration (explicit reset)
	this->getTaskManager()->allThreadsDone.store(false);

#ifdef ACALSIM_STATISTICS
	this->globalTqPartitionTimeStatistics.push(tq_partition_lat);
	this->phase1IdleTimer.startStage();
#endif  // ACALSIM_STATISTICS

	this->getTaskManager()->newTaskAvailableCondVar.notify_all();
}

template <typename T>
void ThreadManagerV6<T>::finishPhase1() {
	// In Phase 1, all simulators will execute one iteration
	// The control thread (SimTop) executes the SimTop::control_thread_step() function
	// This function is to sync the control thread with all the simulators in the end of Phase 1

	// Wait until all worker threads are done
	std::unique_lock<std::mutex> lock(this->getTaskManager()->taskAvailableMutex);
	this->getTaskManager()->workerThreadsDoneCondVar.wait(
	    lock, [this] { return this->getTaskManager()->allThreadsDone.load(); });

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.endStage();
#endif  // ACALSIM_STATISTICS

	// Consolidate tasks after Phase 1 is complete
	MEASURE_TIME_MICROSECONDS(/*var_name*/ tq_partition,
	                          /*code_block*/ { this->getTaskManager()->consolidateTasks(); });

	this->collectAfterPhase1Statistics(top->getGlobalTick() == 0, top->isReadyToTerminate());

#ifdef ACALSIM_STATISTICS
	this->globalTqPartitionTimeStatistics.push(tq_partition_lat);
#endif  // ACALSIM_STATISTICS

	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
	MT_DEBUG_CLASS_INFO << "Control thread " + std::to_string(tid) + " pass the allThreadsDone condition check";
}

template <typename T>
void ThreadManagerV6<T>::startPhase2() {
	// In Phase 2, all simulators are paused and do nothing
	// The control thread (SimTop) does all the bookkeeping things
	// This function is for the control thread to do something in the beginner of Phase 2
}

template <typename T>
void ThreadManagerV6<T>::finishPhase2() {
	// In Phase 2, all simulators are paused and do nothing
	// The control thread (SimTop) does all the bookkeeping things
	// This function is to sync the control thread with all the simulators in the end of Phase 2
}

#ifdef ACALSIM_STATISTICS
template <typename T>
void ThreadManagerV6<T>::printSchedulingOverheads(double _total_time) const {
	double task_queue_lat = this->getTaskManager()->taskQueue.getUniqueLockWaitingTime() +
	                        this->getTaskManager()->taskQueue.getSharedLockWaitingTime() +
	                        NamedTimer<"TaskManagerV6-LocalTaskQueue-Phase1">::getTimerVal();
	double task_queue_manip_lat =
	    this->getTaskManager()->taskQueue.getTqManipTime() + this->localTqManipTimeStatistics.sum();
	double task_queue_prep_lat =
	    this->localTqPrepareTimeStatistics.sum() + this->globalTqPartitionTimeStatistics.sum() * this->nThreads;
	double task_avail_lock_cost_us = NamedTimer<"TaskManagerV6-TaskAvailCv-Phase1">::getTimerVal();

	LABELED_STATISTICS("ThreadManagerV6")
	    << "Scheduling Overheads: "
	    << "(1) Task Queue Lock: " << task_queue_lat / this->nThreads / _total_time * 100 << "% "
	    << "(2) Task Queue Waiting: " << task_queue_prep_lat / this->nThreads / _total_time * 100 << "% "
	    << "(3) Task Queue Manipulation: " << task_queue_manip_lat / this->nThreads / _total_time * 100 << "% "
	    << "(4) Task Avail Lock: " << task_avail_lock_cost_us / this->nThreads / _total_time * 100 << "%.";

	LABELED_STATISTICS("ThreadManagerV6")
	    << "Scheduling Overheads (per thread): "
	    << "(1) Task Queue Lock: " << task_queue_lat / this->nThreads << " us, "
	    << "(2) Task Queue Waiting: " << task_queue_prep_lat / this->nThreads << " us, "
	    << "(3) Task Queue Manipulation: " << task_queue_manip_lat / this->nThreads << " us, "
	    << "(4) Task Avail Lock: " << task_avail_lock_cost_us / this->nThreads << " us.";
}
#endif  // ACALSIM_STATISTICS

}  // namespace acalsim
