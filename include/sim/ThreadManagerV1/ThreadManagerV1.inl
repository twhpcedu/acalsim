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

#include "sim/SimTop.hh"
#include "sim/ThreadManagerV1/TaskManagerV1.hh"
#include "sim/ThreadManagerV1/ThreadManagerV1.hh"
#include "utils/Logging.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Utils.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

template <typename T>
void ThreadManagerV1<T>::startSimThreads() {
	this->T::startSimThreads();

#ifdef ACALSIM_STATISTICS
	for (size_t tid = 0; tid < this->nThreads; ++tid) { this->tqOperationTimeStatistics.addEntry(tid); }
#endif  // ACALSIM_STATISTICS
}

template <typename T>
void ThreadManagerV1<T>::startRunning() {
	// Set the running flag first (via base class)
	this->running = true;

	// Notify all waiting worker threads that simulation is now running
	// This wakes threads from the condition variable wait (replaces busy-wait spin loop)
	{ std::lock_guard<std::mutex> lock(this->getTaskManager()->runningMutex); }
	this->getTaskManager()->runningCondVar.notify_all();
}

template <typename T>
void ThreadManagerV1<T>::runInterIterationUpdate() {
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
void ThreadManagerV1<T>::terminateAllThreads() {
	// Allow child threads to terminate itself
	this->running = false;

	MT_DEBUG_CLASS_INFO << "thread " << std::this_thread::get_id()
	                    << " terminateAllThreads() nFinishedThreads=" << this->getTaskManager()->nFinishedThreads;
	for (auto& worker : this->workers) {
		if (worker->joinable()) worker->join();
	}
}

template <typename T>
void ThreadManagerV1<T>::startPhase1() {
	// In Phase 1, all simulators will execute one iteration
	// The control thread (SimTop) executes the SimTop::control_thread_step() function
	// This function is for the control thread to do something in the beginner of Phase 1

	// wake up worker threads

	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));

	// std::lock_guard<std::mutex> lock(task_manager->taskQueueMutex);

	this->collectBeforePhase1Statistics();

	if (this->getTaskManager()->taskQueue.empty())
		MT_DEBUG_CLASS_INFO << "Control thread " << tid << " wake up all worker threads. taskQueu is empty";
	else
		MT_DEBUG_CLASS_INFO << "Control thread " << tid
		                    << " wake up all worker threads. taskQueue->top().next_execution_cycle="
		                    << this->getTaskManager()->taskQueue.top().next_execution_cycle;

	this->getTaskManager()->nFinishedThreads = 0;
	this->getTaskManager()->startPhase1.store(true);
	// Reset allThreadsDone before starting a new iteration (explicit reset)
	this->getTaskManager()->allThreadsDone.store(false);

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.startStage();
#endif  // ACALSIM_STATISTICS

	// Wake all waiting threads to start the phase
	// Note: notify_one() optimization was attempted but causes issues when
	// threads haven't entered the wait state yet on the first iteration
	this->getTaskManager()->newTaskAvailableCondVar.notify_all();
}

template <typename T>
void ThreadManagerV1<T>::finishPhase1() {
	// In Phase 1, all simulators will execute one iteration
	// The control thread (SimTop) executes the SimTop::control_thread_step() function
	// This function is to sync the control thread with all the simulators in the end of Phase 1

	// Wait until all worker threads are done
	std::unique_lock<std::mutex> lock(this->getTaskManager()->taskAvailableMutex);
	this->getTaskManager()->workerThreadsDoneCondVar.wait(lock, [this] {
		// std::cout << "ThreadManager wait and check\n";
		return this->getTaskManager()->allThreadsDone.load();
	});

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.endStage();
#endif  // ACALSIM_STATISTICS

	this->collectAfterPhase1Statistics(top->getGlobalTick() == 0, top->isReadyToTerminate());

	auto tid = static_cast<uint64_t>(std::hash<std::thread::id>()(std::this_thread::get_id()));
	MT_DEBUG_CLASS_INFO << "Control thread " + std::to_string(tid) + " pass the allThreadsDone condition check";
}

template <typename T>
void ThreadManagerV1<T>::startPhase2() {
	// In Phase 2, all simulators are paused and do nothing
	// The control thread (SimTop) does all the bookkeeping things
	// This function is for the control thread to do something in the beginner of Phase 2
}

template <typename T>
void ThreadManagerV1<T>::finishPhase2() {
	// In Phase 2, all simulators are paused and do nothing
	// The control thread (SimTop) does all the bookkeeping things
	// This function is to sync the control thread with all the simulators in the end of Phase 2
}

#ifdef ACALSIM_STATISTICS
template <typename T>
void ThreadManagerV1<T>::printSchedulingOverheads(double _total_time) const {
	double taskq_lock_cost_us      = NamedTimer<"TaskManagerV1-TaskQueue-Phase1">::getTimerVal();
	double task_avail_lock_cost_us = NamedTimer<"TaskManagerV1-TaskAvailCv-Phase1">::getTimerVal();

	LABELED_STATISTICS("ThreadManagerV1")
	    << "Scheduling Overheads: "
	    << "(1) Task Queue Mutex: " << taskq_lock_cost_us / this->nThreads / _total_time * 100 << "% "
	    << "(2) Task Available Mutex: " << task_avail_lock_cost_us / this->nThreads / _total_time * 100 << "% "
	    << "(3) Task Queue Manipulation: " << this->tqOperationTimeStatistics.sum() / this->nThreads / _total_time * 100
	    << "%.";
	LABELED_STATISTICS("ThreadManagerV1")
	    << "Scheduling Overheads (per thread): "
	    << "(1) Task Queue Mutex: " << taskq_lock_cost_us / this->nThreads << " us, "
	    << "(2) Task Available Mutex: " << task_avail_lock_cost_us / this->nThreads << " us, "
	    << "(3) Task Queue Manipulation: " << this->tqOperationTimeStatistics.sum() / this->nThreads << " us.";
}

#endif  // ACALSIM_STATISTICS

}  // namespace acalsim
