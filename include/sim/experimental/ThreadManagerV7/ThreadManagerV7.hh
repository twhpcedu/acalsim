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

/**
 * @file ThreadManagerV7.hh
 * @warning EXPERIMENTAL: This version was not validated in published research.
 *          Not recommended for production use.
 *
 * @note For production use, see:
 *       - ThreadManagerV1 (PriorityQueue): Default, sparse activation patterns
 *       - ThreadManagerV3 (PrebuiltTaskList): Memory-intensive workloads
 *       - ThreadManagerV6 (LocalTaskQueue): Lock-optimized V1
 *
 * @details ThreadManagerV7 uses C++20 barriers with pre-built task lists (hybrid approach).
 *          This design was explored but not included in published research.
 */

#pragma once

#include <barrier>
#include <functional>
#include <memory>

#include "external/gem5/Event.hh"

#ifdef ACALSIM_STATISTICS
#include "profiling/Statistics.hh"
#endif  // ACALSIM_STATISTICS

namespace acalsim {

template <typename T>
class TaskManagerV7;

template <typename TClass>
class ThreadManagerV7 : public TClass {
	friend class TaskManagerV7<TClass>;

public:
	ThreadManagerV7(const std::string& _name, unsigned int _nThreads, bool _nThreadsAdjustable);

	~ThreadManagerV7() { ; }

	// Pre-Simulation Initialization
	void preSimInit() override { ; }

	void startSimThreads() override;

	// end of Phase 1 synchronization
	void startPhase1() override;

	// end of Phase 2 synchronization
	void startPhase2() override {}

	// end of Phase 1 synchronization
	void finishPhase1() override;

	// end of Phase 2 synchronization
	void finishPhase2() override {}

	// Performance inter-iteration updates (similar to the register update in hardware modeling)
	void runInterIterationUpdate() override;

	void issueExitEvent(Tick _t) override;

	// Terminate all threads
	void terminateAllThreads() override;

	// Post-Simulation Cleanup
	void postSimCleanup() override { ; }

	TaskManagerV7<TClass>* getTaskManager() const { return static_cast<TaskManagerV7<TClass>*>(this->taskManager); }

private:
	void updateTaskQueue();

private:
	std::shared_ptr<std::barrier<std::function<void(void)>>> startPhase1BarrierPtr;
	std::shared_ptr<std::barrier<std::function<void(void)>>> finishPhase1BarrierPtr;

#ifdef ACALSIM_STATISTICS
public:
	void printSchedulingOverheads(double _total_time) const override;

private:
	// The time spent retrieving the task queue
	Statistics<double, StatisticsMode::Accumulator, true> tqRetrievalTimeStatistics;
#endif  // ACALSIM_STATISTICS
};

}  // namespace acalsim

#include "sim/experimental/ThreadManagerV7/ThreadManagerV7.inl"
