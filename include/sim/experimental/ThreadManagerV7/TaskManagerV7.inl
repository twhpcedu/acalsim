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

#include <limits>
#include <memory>
#include <ranges>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "sim/SimBase.hh"
#include "sim/experimental/ThreadManagerV7/TaskManagerV7.hh"
#include "utils/Logging.hh"

#ifdef ACALSIM_STATISTICS
#include "utils/HostPlatform.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

template <typename TFriend>
void TaskManagerV7<TFriend>::init() {
	MT_DEBUG_CLASS_INFO << "TaskManagerV7<TFriend>::init() starts.";

	this->TaskManager::init();

	// Initialize the local task queue of all worker threads
	const size_t n_threads = this->getThreadManager()->getNumThreads();
	this->localSimBaseVec.resize(n_threads);
	this->localTaskQueue.resize(n_threads);

	// Categorize all SimBAse instances based on their type hash values
	std::unordered_map<size_t, std::shared_ptr<std::vector<SimBase*>>> sim_type_map;
	for (const auto& sim : this->getThreadManager()->getAllSimulators()) {
		size_t type_hash = sim->getTypeHash();

		if (sim_type_map.contains(type_hash)) {
			sim_type_map[type_hash]->push_back(sim);
		} else {
			sim_type_map.emplace(std::make_pair(type_hash, new std::vector<SimBase*>{sim}));
		}
	}

	// Sort all SimBase vectors based on their size
	struct SimBaseVecPtrComparator {
		bool operator()(std::shared_ptr<std::vector<SimBase*>> _a, std::shared_ptr<std::vector<SimBase*>> _b) const {
			return _a->size() < _b->size();
		}
	};

	std::multiset<std::shared_ptr<std::vector<SimBase*>>, SimBaseVecPtrComparator> sim_vec_set;
	for (const auto& [type_hash, vec] : sim_type_map) { sim_vec_set.emplace(vec); }

	// Assign SimBase instances to worker threads
	size_t thread_idx = 0;

	for (const auto& vec : std::ranges::reverse_view{sim_vec_set}) {
		for (const auto& sim : *vec) {
			MT_DEBUG_CLASS_INFO << "Assigns SimBase \"" << sim->getName() << "\" to worker thread " << thread_idx
			                    << ".";

			this->localSimBaseVec[thread_idx].push_back(sim);
			this->localTaskQueue[thread_idx].insert(sim, Tick{0});
			thread_idx = (thread_idx + 1) % n_threads;
		}
	}

#ifdef ACALSIM_STATISTICS
	size_t unaligned_simbases = 0;

	for (const auto& sim : this->getThreadManager()->getAllSimulators()) {
		if (alignof(std::remove_reference_t<decltype(*sim)>) != kCacheLineSize ||
		    reinterpret_cast<uintptr_t>(sim) % kCacheLineSize != 0) {
			unaligned_simbases++;
		}
	}

	LABELED_STATISTICS("TaskManagerV7") << "Unaligned SimBase objects: " << unaligned_simbases << ".";
#endif  // ACALSIM_STATISTICS

	MT_DEBUG_CLASS_INFO << "TaskManagerV7<TFriend>::init() completes.";
}

template <typename TFriend>
void TaskManagerV7<TFriend>::scheduler(const size_t _tidx) {
	const std::string thread_idx_str = std::to_string(_tidx);

	std::unordered_set<SimBase*> task_set;

#ifdef ACALSIM_STATISTICS
	auto   task_time_statistics_entry = this->getThreadManager()->taskExecTimeStatistics.getEntry(_tidx);
	double task_time_curr_iter        = 0;
#endif  // ACALSIM_STATISTICS

	while (!this->getThreadManager()->isRunning()) { ; }

	MT_DEBUG_CLASS_INFO << "Thread " << thread_idx_str << " starts running.";

	PriorityQueue<Tick, SimBase*>& task_queue = this->localTaskQueue[_tidx];

	do {
		MT_DEBUG_CLASS_INFO << "Thread " << thread_idx_str << " waits for starting an iteration.";

		// Wait for starting the next execution phase
		this->getThreadManager()->startPhase1BarrierPtr->arrive_and_wait();

		// Check if the local task queue has expired tasks
		if (task_queue.getTopPriority() == top->getGlobalTick()) {
			// Get all local expired tasks via task_set
			task_queue.getTopElements([&task_set](std::unordered_set<SimBase*>& _set) { task_set.swap(_set); });

			for (auto& sim : task_set) {
				MT_DEBUG_CLASS_INFO << "Thread " + thread_idx_str + " starts to execute a task.";
				double task_time = this->collectTaskExecStatistics(&SimBase::stepWrapperBase, sim, sim->getName());

#ifdef ACALSIM_STATISTICS
				task_time_curr_iter += task_time;
#endif  // ACALSIM_STATISTICS
			}
		}

#ifdef ACALSIM_STATISTICS
		this->getThreadManager()->phase1IdleTimer.enterSyncPoint(_tidx);
		task_time_statistics_entry->push(task_time_curr_iter);
		task_time_curr_iter = 0;
#endif  // ACALSIM_STATISTICS

		MT_DEBUG_CLASS_INFO << "Thread " << thread_idx_str
		                    << " arrives at ThreadManagerV7<TFriend>::finishPhase1BarrierPtr.";

		// Enter the synchronization point of this execution phase
		(void)this->getThreadManager()->finishPhase1BarrierPtr->arrive();
	} while (this->threadManager->isRunning() && !top->isReadyToTerminate());

	MT_DEBUG_CLASS_INFO << "Thread " + thread_idx_str + " terminates.";
}

template <typename TFriend>
Tick TaskManagerV7<TFriend>::getNextSimTick() {
	Tick tick = std::numeric_limits<Tick>::max();

	for (const auto& queue : this->localTaskQueue) {
		Tick local_tick = queue.getTopPriority();
		if (local_tick < tick) { tick = local_tick; }
	}

	return tick;
}

}  // namespace acalsim
