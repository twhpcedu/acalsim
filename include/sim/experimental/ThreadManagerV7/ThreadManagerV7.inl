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

#include <memory>

#include "common/PriorityQueue.hh"
#include "sim/SimBase.hh"
#include "sim/SimTop.hh"
#include "sim/experimental/ThreadManagerV7/ThreadManagerV7.hh"

namespace acalsim {

template <typename TClass>
ThreadManagerV7<TClass>::ThreadManagerV7(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable)
    : TClass(_name, _nThreads, _nThreadsAdjustable) {}

template <typename TClass>
void ThreadManagerV7<TClass>::startSimThreads() {
	this->TClass::startSimThreads();
	this->startPhase1BarrierPtr =
	    std::make_shared<std::barrier<std::function<void(void)>>>(this->getNumThreads() + 1, []() {});
	this->finishPhase1BarrierPtr =
	    std::make_shared<std::barrier<std::function<void(void)>>>(this->getNumThreads() + 1, []() {});
}

template <typename TClass>
void ThreadManagerV7<TClass>::startPhase1() {
	this->collectBeforePhase1Statistics();

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.startStage();
#endif  // ACALSIM_STATISTICS

	(void)this->startPhase1BarrierPtr->arrive();
}

template <typename TClass>
void ThreadManagerV7<TClass>::finishPhase1() {
	this->finishPhase1BarrierPtr->arrive_and_wait();

#ifdef ACALSIM_STATISTICS
	this->phase1IdleTimer.endStage();
#endif  // ACALSIM_STATISTICS

	this->collectAfterPhase1Statistics(top->getGlobalTick() == 0, top->isReadyToTerminate());
}

template <typename TClass>
void ThreadManagerV7<TClass>::runInterIterationUpdate() {
	for (size_t tid = 0; tid < this->getNumThreads(); ++tid) {
		const std::vector<SimBase*>& sim_vec = this->getTaskManager()->localSimBaseVec[tid];

		for (SimBase* const& sim : sim_vec) { sim->interIterationUpdate(); }
	}

	this->updateTaskQueue();
}

template <typename TClass>
void ThreadManagerV7<TClass>::issueExitEvent(Tick _t) {
	MT_DEBUG_CLASS_INFO << "ThreadManagerV7<TClass>::issueExitEvent(Tick) starts.";

	this->TClass::issueExitEvent(_t);
	this->updateTaskQueue();
}

template <typename TClass>
void ThreadManagerV7<TClass>::terminateAllThreads() {
	MT_DEBUG_CLASS_INFO << "ThreadManagerV7<TClass>::terminateAllThreads() starts.";

	this->running = false;

	for (auto& thread : this->workers) { thread->join(); }
}

template <typename TClass>
void ThreadManagerV7<TClass>::updateTaskQueue() {
	for (size_t tid = 0; tid < this->getNumThreads(); ++tid) {
		const std::vector<SimBase*>&   sim_vec    = this->getTaskManager()->localSimBaseVec[tid];
		PriorityQueue<Tick, SimBase*>& task_queue = this->getTaskManager()->localTaskQueue[tid];

		for (SimBase* const& sim : sim_vec) {
			Tick next_tick = sim->getSimNextTick();
			if (next_tick != -1) { task_queue.insert(sim, next_tick); }
		}
	}
}

#ifdef ACALSIM_STATISTICS
template <typename TClass>
void ThreadManagerV7<TClass>::printSchedulingOverheads(double) const {}
#endif

}  // namespace acalsim
