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
#include "sim/SimBase.hh"
#include "sim/SimTop.hh"
#include "sim/ThreadManager.hh"
#include "sim/experimental/ThreadManagerV8/TaskManagerV8.hh"
#include "sim/experimental/ThreadManagerV8/ThreadManagerV8.hh"

namespace acalsim {

template <typename TClass>
ThreadManagerV8<TClass>::ThreadManagerV8(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable)
    : TClass(_name, _nThreads, _nThreadsAdjustable) {
	this->stepTaskList                 = std::make_shared<typename TaskManagerV8<TClass>::TaskList>();
	this->interIterationUpdateTaskList = std::make_shared<typename TaskManagerV8<TClass>::TaskList>();
}

template <typename TClass>
void ThreadManagerV8<TClass>::preSimInit() {
	// Note: The following operations prepare ThreadManagerV8<TClass>::interIterationUpdateTaskList
	// for ThreadManagerV8<TClass>::runInterIterationUpdate(). However, due to performance
	// considerations, this task list is not currently used.

	auto task_manager = this->getTaskManager();

	for (auto& sim : this->simulators) {
		this->interIterationUpdateTaskList->insert([task_manager, sim]() {
			sim->interIterationUpdate();
			Tick next_tick = sim->getSimNextTick();
			if (next_tick != -1) { task_manager->setSimulatorNextTick(sim, next_tick); }
		});
	}
}

template <typename TClass>
void ThreadManagerV8<TClass>::startPhase1() {
	// In Phase 1, all simulators will execute one iteration
	// The control thread (SimTop) executes the SimTop::control_thread_step() function
	// This function is for the control thread to do something in the beginner of Phase 1

	this->collectBeforePhase1Statistics();

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.startStage();
#endif  // ACALSIM_STATISTICS

	MEASURE_TIME_MICROSECONDS(task_list_preparation, {
		if (top->getGlobalTick() == 0 || top->isReadyToTerminate()) {
			for (auto& sim : this->simulators) {
				this->stepTaskList->insert([sim]() { sim->stepWrapperBase(); });
			}
		} else [[likely]] {
			if (top->getGlobalTick() == this->getFastForwardCycles()) {
				this->getTaskManager()->simulatorNextTickQueue.getTopElements([this](SimBase* const& _sim) {
					// Add the step wrappers of simulators that should be executed in the next tick to the task list
					this->stepTaskList->insert([_sim]() { _sim->stepWrapperBase(); });
				});
			}
		}

		this->getTaskManager()->submitTasks(this->stepTaskList);
	});

#ifdef ACALSIM_STATISTICS
	this->tqPreparationTimeStatistics.push(task_list_preparation_lat);
#endif  // ACALSIM_STATISTICS
}

template <typename TClass>
void ThreadManagerV8<TClass>::finishPhase1() {
	// In Phase 1, all simulators will execute one iteration
	// The control thread (SimTop) executes the SimTop::control_thread_step() function
	// This function is to sync the control thread with all the simulators in the end of Phase 1

	this->getTaskManager()->wait();

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.endStage();
#endif  // ACALSIM_STATISTICS

	MEASURE_TIME_MICROSECONDS(task_list_clear, { this->stepTaskList->clear(); });

#ifdef ACALSIM_STATISTICS
	this->tqPreparationTimeStatistics.push(task_list_clear_lat);
#endif  // ACALSIM_STATISTICS

	this->collectAfterPhase1Statistics(top->getGlobalTick() == 0, top->isReadyToTerminate());
}

template <typename TClass>
void ThreadManagerV8<TClass>::startPhase2() {
	// In Phase 2, all simulators are paused and do nothing
	// The control thread (SimTop) does all the bookkeeping things
	// This function is for the control thread to do something in the beginner of Phase 2
}

template <typename TClass>
void ThreadManagerV8<TClass>::finishPhase2() {
	// In Phase 2, all simulators are paused and do nothing
	// The control thread (SimTop) does all the bookkeeping things
	// This function is to sync the control thread with all the simulators in the end of Phase 2
}

template <typename TClass>
void ThreadManagerV8<TClass>::runInterIterationUpdate() {
	// Note: The most intuitive way to execute SimBase::interIterationUpdate() for all simulators
	// would be to submit ThreadManagerV8<TClass>::interIterationUpdateTaskList to TaskManagerV8.
	// However, experiments show this approach results in slower performance. This suggests that
	// the execution time of SimBase::interIterationUpdate() is minimal compared to the overhead
	// of dynamic thread allocation in TaskManagerV8<TClass>::scheduler().

	for (auto& sim : this->simulators) {
		sim->interIterationUpdate();
		Tick next_tick = sim->getSimNextTick();
		if (next_tick != -1) { this->getTaskManager()->setSimulatorNextTick(sim, next_tick); }
	}

	// this->getTaskManager()->submitTasks(this->interIterationUpdateTaskList);
	// this->getTaskManager()->wait();
}

template <typename TClass>
void ThreadManagerV8<TClass>::terminateAllThreads() {
	this->running = false;
	(void)this->getTaskManager()->consumerBarrierPtr->arrive();

	for (auto& thread : this->workers) { thread->join(); }
}

#ifdef ACALSIM_STATISTICS
template <typename TClass>
void ThreadManagerV8<TClass>::printSchedulingOverheads(double _total_time) const {
	double taskq_lock_cost_us     = NamedTimer<"TaskManagerV8-TaskQueue-Phase1">::getTimerVal();
	double runner_cv_lock_cost_us = NamedTimer<"TaskManagerV8-TaskAvailCv-Phase1">::getTimerVal();

	LABELED_STATISTICS("ThreadManagerV8")
	    << "Scheduling Overheads: "
	    << "(1) Task Queue Mutex: " << taskq_lock_cost_us / this->nThreads / _total_time * 100 << "% "
	    << "(2) Runner CV Mutex: " << runner_cv_lock_cost_us / this->nThreads / _total_time * 100 << "% "
	    << "(3) Task List Preparation: " << this->tqPreparationTimeStatistics.sum() / _total_time * 100 << "% "
	    << "(4) Task Queue Manipulation: " << this->tqRetrievalTimeStatistics.sum() / this->nThreads / _total_time * 100
	    << "%.";

	LABELED_STATISTICS("ThreadManagerV8")
	    << "Scheduling Overheads (per thread): "
	    << "(1) Task Queue Mutex: " << taskq_lock_cost_us / this->nThreads << " us, "
	    << "(2) Runner CV Mutex: " << runner_cv_lock_cost_us / this->nThreads << " us, "
	    << "(3) Task List Preparation: " << this->tqPreparationTimeStatistics.sum() << "us, "
	    << "(4) Task Queue Manipulation: " << this->tqRetrievalTimeStatistics.sum() / this->nThreads << " us.";
}
#endif  // ACALSIM_STATISTICS

}  // namespace acalsim
