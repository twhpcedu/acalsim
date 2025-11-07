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

/**
 * @file ThreadManagerV8.hh
 * @warning EXPERIMENTAL: This version was not validated in published research.
 *          Not recommended for production use.
 *
 * @note For production use, see:
 *       - ThreadManagerV1 (PriorityQueue): Default, sparse activation patterns
 *       - ThreadManagerV3 (PrebuiltTaskList): Memory-intensive workloads
 *       - ThreadManagerV6 (LocalTaskQueue): Lock-optimized V1
 *
 * @details ThreadManagerV8 is an experimental refinement of V3 with pre-built task lists.
 *          This design was explored but not included in published research.
 */

#pragma once

#include <string>

// ACALSim
#include "sim/experimental/ThreadManagerV8/TaskManagerV8.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

template <typename TClass>
class ThreadManagerV8 : public TClass {
	friend class TaskManagerV8<TClass>;

public:
	ThreadManagerV8(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable);

	~ThreadManagerV8() { ; }

	// Pre-Simulation Initialization
	void preSimInit() override;

	// Performance inter-iteration updates (similar to the register update in hardware modeling)
	void runInterIterationUpdate() override;

	// Terminate all threads
	void terminateAllThreads() override;

	// end of Phase 1 synchronization
	void startPhase1() override;

	// end of Phase 2 synchronization
	void startPhase2() override;

	// end of Phase 1 synchronization
	void finishPhase1() override;

	// end of Phase 2 synchronization
	void finishPhase2() override;

	// Post-Simulation Cleanup
	void postSimCleanup() override { ; }

	TaskManagerV8<TClass>* getTaskManager() const { return static_cast<TaskManagerV8<TClass>*>(this->taskManager); }

private:
	std::shared_ptr<typename TaskManagerV8<TClass>::TaskList> stepTaskList                 = nullptr;
	std::shared_ptr<typename TaskManagerV8<TClass>::TaskList> interIterationUpdateTaskList = nullptr;

#ifdef ACALSIM_STATISTICS
public:
	void printSchedulingOverheads(double _total_time) const override;

private:
	// The time spent preparing the task queue
	Statistics<double, StatisticsMode::Accumulator, true> tqPreparationTimeStatistics;

	// The time spent retrieving the task queue
	Statistics<double, StatisticsMode::Accumulator, true> tqRetrievalTimeStatistics;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim

#include "sim/experimental/ThreadManagerV8/ThreadManagerV8.inl"
