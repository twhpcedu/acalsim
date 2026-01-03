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
#include "sim/ThreadManagerV2/TaskManagerV2.hh"

namespace acalsim {

template <typename T>
void ThreadManagerV2<T>::preSimInit() {
	this->nThreads = (this->nThreads > this->simulators.size() ? this->simulators.size() : this->nThreads);
	// initialize the barrier
	this->workerThreadsDoneBarrier = new std::barrier<void (*)(void)>(this->nThreads + 1, []() {});

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.startStage();
#endif  // ACALSIM_STATISTICS
}

template <typename T>
void ThreadManagerV2<T>::runInterIterationUpdate() {
	for (auto& sim : this->simulators) {
		if (sim->interIterationUpdate()) {
			// The simulator has pending inbound requests in the SimPort
			// The corresponding task might not be the top of the taskQueue
			// Need to promote the task to the top
			this->getTaskManager()->promoteTaskToTop(std::static_pointer_cast<TaskV2>(sim->getTask()));
		}
	}
}

template <typename T>
void ThreadManagerV2<T>::terminateAllThreads() {
	// Allow child threads to terminate itself
	this->running = false;

	MT_DEBUG_CLASS_INFO << "thread " << std::this_thread::get_id()
	                    << " terminateAllThreads() nFinishedThreads=" << this->getTaskManager()->nFinishedThreads;

	for (auto& worker : this->workers) { worker->join(); }
}

template <typename T>
void ThreadManagerV2<T>::startPhase1() {
	// In Phase 1, all simulators will execute one iteration
	// The control thread (SimTop) executes the SimTop::control_thread_step() function
	// This function is for the control thread to do something in the beginner of Phase 1

	this->collectBeforePhase1Statistics();
}

template <typename T>
void ThreadManagerV2<T>::finishPhase1() {
	// In Phase 1, all simulators will execute one iteration
	// The control thread (SimTop) executes the SimTop::control_thread_step() function
	// This function is to sync the control thread with all the simulators in the end of Phase 1
	this->getWorkerThreadsDoneBarrier()->arrive_and_wait();

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.endStage();
#endif  // ACALSIM_STATISTICS

	this->collectAfterPhase1Statistics(top->getGlobalTick() == 0, top->isReadyToTerminate());
}

template <typename T>
void ThreadManagerV2<T>::startPhase2() {
	// In Phase 2, all simulators are paused and do nothing
	// The control thread (SimTop) does all the bookkeeping things
	// This function is for the control thread to do something in the beginner of Phase 2
}

template <typename T>
void ThreadManagerV2<T>::finishPhase2() {
	// In Phase 2, all simulators are paused and do nothing
	// The control thread (SimTop) does all the bookkeeping things
	// This function is to sync the control thread with all the simulators in the end of Phase 2

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.startStage();
#endif  // ACALSIM_STATISTICS

	this->getWorkerThreadsDoneBarrier()->arrive_and_wait();
}

#ifdef ACALSIM_STATISTICS
template <typename TClass>
void ThreadManagerV2<TClass>::printSchedulingOverheads(double _total_time) const {
	double taskq_lock_cost_us = NamedTimer<"TaskManagerV2-TaskQueue-Phase1">::getTimerVal();

	LABELED_STATISTICS("ThreadManagerV2")
	    << "Scheduling Overheads: "
	    << "(1) Task Queue Mutex: " << taskq_lock_cost_us / this->nThreads / _total_time * 100 << "% "
	    << "(2) Task Queue Manipulation: " << this->tqManipTimeStatistics.sum() / this->nThreads / _total_time * 100
	    << "%.";

	LABELED_STATISTICS("ThreadManagerV2")
	    << "Scheduling Overheads (per thread): "
	    << "(1) Task Queue Mutex: " << taskq_lock_cost_us / this->nThreads << " us, "
	    << "(2) Task Queue Manipulation: " << this->tqManipTimeStatistics.sum() / this->nThreads << " us.";
}
#endif  // ACALSIM_STATISTICS

}  // namespace acalsim
